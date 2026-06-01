# Llama-3.1-8B-Instruct: vLLM vs Megatron MXFP8 numeric mismatch — initial report

This document is the **starting baseline** for the MXFP8 cross-engine numerical
matching effort. It captures the result of the first vLLM↔Megatron MXFP8
comparison after all known BF16 patches were ported over, plus the
infrastructure changes required to get a Megatron MXFP8 forward to run at all
in our test harness.

The sister document
[`llama3_8b_numeric_mismatch.md`](llama3_8b_numeric_mismatch.md) covers the
BF16 work that landed bit-identical end-to-end parity. The proposed
batch-invariant MXFP8 matmul kernel that would close the remaining MXFP8 gap
is sketched in [`mxfp8_bi_matmul_design.md`](mxfp8_bi_matmul_design.md).

---

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct.
  - vLLM consumes a pre-quantized MXFP8 ckpt produced by
    `my_script/convert_hf_bf16_ckpt_to_mxfp8.py` (1×32 block scaling, E8M0
    scales). vLLM detects the quantization from `config.json`'s
    `quantization_config = {"quant_algo": "MXFP8", "quant_method": "modelopt"}`.
  - Megatron consumes the BF16 HF ckpt and quantizes activations + weights
    on the fly via Transformer Engine's `MXFP8BlockScaling` recipe
    (`provider.fp8 = "e4m3"`, `provider.fp8_recipe = "mxfp8"`).
- **Prompt**: `"The quick brown fox jumps over the lazy dog."` (11 tokens
  including BOS).
- **Hardware**: single GB200 (Blackwell, SM10.0).
- **Activations**: BF16 on both sides (only the GEMM weights/activations get
  quantized to FP8 at the kernel boundary).
- **Capture script invocations**:
  ```bash
  uv run --extra vllm python my_script/vllm_forward.py \
      --mxfp8 \
      --model /lustre/fsw/coreai_dlalgo_llm/users/guyueh/checkpoints/meta-llama--Llama-3.1-8B-Instruct.mxfp8 \
      --tokenizer meta-llama/Llama-3.1-8B-Instruct \
      --batch-invariant --capture-layers 0

  uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py \
      --mxfp8 --batch-invariant --split-fused \
      --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm \
      --capture-layers 0
  ```

---

## Infrastructure changes required to get a Megatron MXFP8 forward working

Three back-to-back TE / BI-mode failures had to be resolved before any
numerical comparison was possible. Each is documented here so future MXFP8
debugging doesn't re-hit them.

### 1. Sequence-length padding to 32

**Failure**:
```
ValueError: FP8 execution requires the product of all dimensions except the
last to be divisible by 8 and the last dimension to be divisible by 16, but
got tensor with dims=[11, 1, 4096] (product of leading dims = 11, last dim
= 4096)
```
Then, after padding to 16:
```
RuntimeError: MXFP8 requires tensor dims that are divisible by 32 (got
shape=(16,1,4096))
```

**Root cause**: TE's MXFP8 quantizer requires every dimension of the
quantized activation tensor to be a multiple of the MXFP8 block size (32).
With batch=1, this forces `seq_len % 32 == 0`. The base prompt is 11 tokens.

**Fix**: when `--mxfp8` is set, `megatron_forward.py` pads `token_ids` to the
next multiple of 32 (32 for our 11-token prompt) using the EOS token:

```python
# my_script/megatron_forward.py
real_seq_len = len(token_ids_list)
if args.mxfp8 and real_seq_len % 32 != 0:
    padded_len = ((real_seq_len + 31) // 32) * 32
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    token_ids_list = token_ids_list + [pad_id] * (padded_len - real_seq_len)
```

Causal attention makes the padded positions invisible to positions 0–10, so
the captured tensors at positions 0–10 are unaffected by the pad. `compare.py`
already truncates flat-comparison length to `min(vllm_n, mcore_n)`, so
positions 11–31 are dropped on the Megatron side automatically. vLLM is
left at 11 tokens — no padding needed there (vLLM's MXFP8 kernel accepts
non-multiple-of-32 prefill length).

### 2. BI GEMM patch can't ingest MXFP8 tensors

**Failure**:
```
AttributeError: 'MXFP8TensorStorage' object has no attribute 'is_cuda'
  at batch_invariant_kernels.py:849 in _te_general_gemm_patched:
    if (not A.is_cuda) or (not B.is_cuda):
```

**Root cause**: Megatron's `enable_batch_invariant_mode()` patches TE's
`general_gemm` to route every linear's GEMM through `_te_general_gemm_patched`
(which dispatches to the BF16 BI Triton matmul). Under MXFP8, the GEMM
weights are wrapped in `MXFP8TensorStorage`, which does not expose `is_cuda`.
The BF16 BI matmul has no way to consume an MXFP8 tensor anyway — fp8 +
block-scaling support is exactly what
[`mxfp8_bi_matmul_design.md`](mxfp8_bi_matmul_design.md) sets out to add.

**Fix (interim)**: when `--mxfp8 --batch-invariant` are both on, wrap the BI
patch so that any GEMM call whose inputs aren't regular fp32/fp16/bf16 CUDA
tensors falls through to TE's original `general_gemm`. BF16 GEMMs (none of
those exist in MXFP8-Llama, but the path is preserved for safety) still go
through the BI Triton matmul. New helper in `megatron_forward.py`:

```python
def install_mxfp8_passthrough_for_bi_gemm():
    """Wrap _te_general_gemm_patched so MXFP8 tensors route to TE's
    original general_gemm; BF16 GEMMs still hit BI Triton."""
    # ... see source for full impl
```

Called from `main()` immediately after `enable_batch_invariant_mode()` when
`args.mxfp8` is set. This mirrors what vLLM already does for MXFP8: vLLM's
BI patches intercept `aten::{mm, addmm, matmul, linear}` for plain bf16, but
the ModelOpt MXFP8 path bypasses aten and so vLLM's BI dispatcher doesn't
apply to MXFP8 GEMMs either.

### 3. Tokenizer source decoupled from model path

vLLM's MXFP8 ckpt directory ships only weights + config + safetensors index;
no tokenizer files. The consolidated `vllm_forward.py` now exposes a
`--tokenizer` flag (defaulting to `--model`) so the MXFP8 case can pin
tokenization to the canonical BF16 HF id (`meta-llama/Llama-3.1-8B-Instruct`)
while pointing `--model` at the MXFP8 checkpoint directory.

---

## Result: L0 module-level diffs

| Module pair (positions 0–10) | max_abs | mean_abs | Bit-identical? |
|---|---|---|---|
| `input_layernorm` input (embedding) | 0 | 0 | ✓ |
| `linear_qkv.linear` input (post first RMSNorm) | **0** | **0** | ✓ |
| `linear_proj` input (post-SDPA) | **2.44e-4** | 8.83e-8 | ✗ — first MXFP8 divergence |
| `linear_fc1.linear` input (post second RMSNorm) | 3.91e-3 | 3.86e-5 | ✗ |
| `linear_fc2` input (post-SwiGLU) | 3.91e-3 | 1.25e-5 | ✗ |

`post_attention_layernorm ↔ pre_mlp_layernorm` shows the known
module-boundary semantic mismatch (vLLM captures `attn_out` pre-add,
Megatron captures the post-add residual stream — not a real numerical
divergence).

## Result: per-layer residual stream drift (positions 0–10)

| Layer | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| 0 | 0.00146 | 4.98e-5 | 1.0 |
| 5 | 0.0625 | 3.27e-3 | 1.0 |
| 10 | 0.125 | 7.21e-3 | 1.0 |
| 20 | 0.250 | 1.21e-2 | 0.99999 |
| 30 | 0.500 | 3.84e-2 | 0.99981 |
| 31 | **2.0** | **5.35e-2** | 0.99809 |

For reference, the corresponding BF16 run with the same set of patches
achieves max_abs = 0 across all 32 layers and the final logits.

## Result: final logits — comparison method is invalid (and needs a fix in compare.py)

`compare.py`'s current `m_last = m["logits"][0, -1]` heuristic returns the
**last padded position** (position 31 = an EOS pad token) when the
Megatron-side seq is padded for MXFP8. The reported `max_abs_diff = 24.66`
and `cos_sim = -0.36` are an artefact of comparing position 31 (padded) on
Megatron against position 10 (real last prompt token) on vLLM.

→ **Open todo**: extend `compare.py` (and the capture payload) to record the
real seq_len so the last-position comparison takes
`m["logits"][0, real_seq_len - 1]`. The per-layer residual table above is
correctly truncated to positions 0–10 via the existing `min(...)` length
heuristic, so it does not suffer from this bug.

---

## Diagnosis

### What matches

- **Embedding** (bit-identical, as expected — both engines read the same
  BF16 embedding weight and the BF16 ckpt is the same on both sides).
- **First RMSNorm** (bit-identical). Both engines route through vLLM's
  `rms_norm_batch_invariant` Triton kernel (`_rms_norm_kernel`) — same
  kernel binary, same inputs, same outputs. This confirms the
  `install_vllm_style_rmsnorm` patch works as designed even under MXFP8;
  RMSNorm input/output is BF16 (FP8 only appears at the GEMM boundary).

### What doesn't match — and why

The divergence enters at the **`linear_qkv` MXFP8 GEMM** (L0's first
linear). Its input is bit-identical (post-norm BF16 activation), but the
post-SDPA tensor (`linear_proj` input) already shows ≈ 1 bf16 ULP of drift.
The chain through the L0 attention block has three candidate sources of
divergence between the two engines:

1. **The QKV MXFP8 GEMM itself.** vLLM dispatches to the ModelOpt /
   flashinfer MXFP8 kernel; Megatron/TE dispatches to TE's cuBLASLt-backed
   `general_gemm` under the `MXFP8BlockScaling` recipe. The two kernels
   apply the same math in principle but pick different tile shapes, K
   reduction trees, and on-line activation-quantization paths. Given
   identical BF16 inputs and identical MXFP8 weights they will *not*
   produce bit-identical fp32 accumulators.
2. **On-line activation quantization.** Both engines quantize the BF16
   activation to MXFP8 (E4M3 + per-32-element E8M0 scales) inside / just
   before the GEMM. The vLLM ModelOpt path and TE quantizer may pick
   slightly different scaling rules at the same input (e.g., rounding of
   the E8M0 exponent at block-amax boundaries).
3. **RoPE / SDPA path.** Already aligned in BF16 via the
   `install_vllm_style_rope` and `install_vllm_style_sdpa` patches.
   Confirmed bit-identical on the BF16 path; under MXFP8 the Q/K/V
   *inputs* to those ops already differ from (1) above, so any further
   divergence at SDPA is downstream noise, not new divergence.

The 0 max_abs at `linear_qkv.linear` input and 2.44e-4 max_abs at
`linear_proj` input localise the first divergence to **the QKV GEMM and
its on-line activation quantization**, before SDPA. The drift then
compounds layer-by-layer through every subsequent linear, ending at
`max_abs = 2.0` at L31.

### Why this matters

Closing the L0 QKV-GEMM gap is the entire MXFP8 numerical-matching task.
The same divergence appears at every linear in every layer (it's the same
two kernels), so a single fix — a batch-invariant MXFP8 matmul kernel
shared by both engines — would close all of them at once.

---

## Next steps

1. **Fix `compare.py` for padded seq_len.** Have both forward scripts emit
   `real_seq_len` in the payload; have `compare.py` take
   `m["logits"][0, real_seq_len - 1]` when present. Until this lands the
   final-logits row is misleading.
2. **Build the batch-invariant MXFP8 matmul kernel.** The design is in
   [`mxfp8_bi_matmul_design.md`](mxfp8_bi_matmul_design.md). Headline:
   Triton kernel with `tl.dot(fp8, fp8, fp32_acc)` on 32-element K
   sub-blocks, per-block scale multiply factored as
   `acc += partial * (a_scale * b_scale)`. Same tile shape (BM=128, BN=128,
   BK=64) as the BF16 BI matmul so the dispatcher diff is small.
3. **Wire the new BI MXFP8 matmul into both engines' BI dispatchers.**
   - vLLM: extend `mm_batch_invariant` / `addmm_batch_invariant` /
     `matmul_batch_invariant` / `linear_batch_invariant` to detect
     MXFP8-quantized weight tensors and route to the new kernel.
   - Megatron: replace `install_mxfp8_passthrough_for_bi_gemm()` (which
     currently bypasses BI for FP8) with the actual MXFP8 BI dispatch.
4. **Verify**: re-run the L0 capture + compare. Expected outcome — same
   sequence as the BF16 run: bit-identical at the GEMM output, with
   residual drift dropping to whatever the activation-quantization-rounding
   floor allows (potentially 0 if both engines quantize the same input
   identically).
5. **Probe activation-quantization parity.** Even with a shared BI matmul
   kernel, the input-quantization step (BF16 → MXFP8 + E8M0 scales) is
   done before the kernel. If the two engines pick different scaling
   rules, the kernel still sees different inputs and the L0 divergence
   persists. Worth a unit test: hand the same BF16 activation tensor to
   both quantizers, compare the resulting (fp8_data, e8m0_scale) tuples.

---

## File and patch inventory

| Artefact | Path | Purpose |
|---|---|---|
| Pre-quantized MXFP8 ckpt | `/lustre/fsw/coreai_dlalgo_llm/users/guyueh/checkpoints/meta-llama--Llama-3.1-8B-Instruct.mxfp8` | vLLM input |
| BF16 → MXFP8 conversion script | `my_script/convert_hf_bf16_ckpt_to_mxfp8.py` | One-time conversion (written separately) |
| Consolidated vLLM forward | `my_script/vllm_forward.py` (`--mxfp8`, `--tokenizer`) | Both BF16 and MXFP8 captures |
| Consolidated Megatron forward | `my_script/megatron_forward.py` (`--mxfp8`, `--fp8-format`) | Both BF16 and MXFP8 captures |
| Seq-pad logic | `my_script/megatron_forward.py` (pad to multiple of 32 under `--mxfp8`) | TE MXFP8 dim constraint |
| BI GEMM FP8 passthrough | `my_script/megatron_forward.py::install_mxfp8_passthrough_for_bi_gemm` | Route MXFP8 GEMMs around the BF16 BI matmul |
| vLLM MXFP8 capture | `my_script/vllm_capture_mxfp8_bi.pt` | Baseline |
| Megatron MXFP8 capture | `my_script/megatron_capture_mxfp8_split_vllmrope_vllmswiglu_vllmsdpa_vllmrmsnorm_bi.pt` | Baseline |
| Compare output | `my_script/compare_mxfp8_bi_L0.log` | Per-module + per-layer diffs |
