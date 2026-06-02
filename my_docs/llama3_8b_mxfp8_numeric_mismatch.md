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

---

## Update 2026-06-01: dequant-to-bf16 + BF16 BI matmul approach

After the initial baseline, switched both engines to a different MXFP8 BI
strategy: **dequant both GEMM inputs to bf16 and reuse the existing BF16
batch-invariant matmul kernel**. The motivation is simplicity — instead of
writing a new MXFP8 BI matmul kernel (per
[`mxfp8_bi_matmul_design.md`](mxfp8_bi_matmul_design.md)), reuse the kernel
that's already bit-identical between vLLM and Megatron in the BF16 work.

### Implementation

**vLLM** — new `BatchInvariantMxfp8LinearKernel` at
`3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py`:

```python
def apply_weights(self, layer, x, bias=None):
    weight_bf16 = dequant_mxfp8_to_bf16(layer.weight, weight_scale)  # [N, K] bf16
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = matmul_persistent(x_2d, weight_bf16.t())               # BF16 BI matmul
    if bias is not None: out_2d = out_2d + bias
    return out_2d.reshape(*x.shape[:-1], -1).to(x.dtype)
```

The activation `x` is BF16; we don't quantize it. Registered first in
`_POSSIBLE_MXFP8_KERNELS[CUDA]` so it wins automatically under
`VLLM_BATCH_INVARIANT=1`.

**Megatron** — three new patches in `my_script/megatron_forward.py`, all
auto-applied when `--mxfp8 --batch-invariant` are both set:

1. `install_mxfp8_compact_scales()` — replaces
   `MXFP8Quantizer.optimize_for_gemm` with a `property` that always reads
   `False` and silently swallows writes. Required because TE's
   `basic_linear.py:353`, `forward_grouped_mlp.py:287`, and
   `backward_grouped_mlp.py:352` explicitly set
   `quantizer.optimize_for_gemm = True` just before each `tex.quantize`
   call; otherwise the resulting MXFP8 tensors carry swizzled scales and
   `tex.dequantize` refuses them with
   `Assertion failed: !input.with_gemm_swizzled_scales`.
2. `install_mxfp8_dequant_for_bi_gemm()` — wraps the BI `general_gemm`
   hook (TE has four binding points: `te_cpp.general_gemm`,
   `te_linear_mod.general_gemm`,
   `te_layernorm_linear_mod.general_gemm`, and Megatron's
   `meg_te.general_gemm`). For each TE quantised input, calls
   `.dequantize(dtype=torch.bfloat16)` to get back a bf16 tensor, then
   feeds the bf16 result into `BatchInvariantTEGemmFn` (the BF16 BI
   matmul path). Replaces the earlier no-op
   `install_mxfp8_passthrough_for_bi_gemm`.
3. Same `--mxfp8` seq-pad-to-32 logic stays in place.

### Result with this approach (L0 module-level)

Configuration:
`uv run --extra vllm  python    my_script/vllm_forward.py     --mxfp8 --batch-invariant --capture-layers 0`
followed by
`uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py --mxfp8 --batch-invariant --split-fused --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm --capture-layers 0`
followed by
`compare.py`.

| L0 tensor | max_abs | Bit-identical? |
|---|---|---|
| `input_layernorm` input (embedding) | 0 | ✓ |
| `linear_qkv.linear` input (post first RMSNorm) | 0 | ✓ |
| `linear_proj` input (post-SDPA) | **0.00586** | ✗ first MXFP8 divergence |
| `linear_fc1.linear` input (post second RMSNorm) | 0.0313 | ✗ |
| `linear_fc2` input (post-SwiGLU) | 0.125 | ✗ |

Per-layer residual drift (positions 0–10): L0=0.047, L5=0.250, L10=0.109,
L20=0.250, **L31=12.0** (compared to L31=2.0 in the prior FlashInfer-vs-TE
baseline). The deeper-layer drift grew under this approach.

Why is the new approach *worse* at depth? At the time we didn't know; we
hypothesised differing weight quantizers between vLLM (converter / disk
ckpt) and Megatron (TE on-the-fly quantization at GEMM time). The
diagnostic below disproves that.

### Diagnostic: are vLLM's and Megatron's MXFP8 quantizers byte-identical?

`my_script/compare_mxfp8_quant.py` (new) loads
`model.layers.0.self_attn.q_proj.weight` (4096×4096 bf16) from the HF
ckpt and runs three quantizers on it in the same Python process (mcore
venv has both TE and vLLM importable):

1. **vLLM converter** (already on disk in the MXFP8 ckpt produced by
   `convert_hf_bf16_ckpt_to_mxfp8.py`). Formula:
   `exponent = ceil(log2(amax / 448)); biased = clamp(exponent, -127, 127) + 127`.
2. **TE `MXFP8Quantizer.quantize_impl(w)`**. Formula on Blackwell:
   single PTX `cvt.rp.satfinite.ue8m0x2.f32` (round-up to E8M0)
   applied to `amax / 448` — equivalent to `ceil(log2(amax/448)) + 127`.
3. **vLLM `mxfp8_e4m3_quantize(w)`** — runtime quantizer, dispatches to
   FlashInfer's `mxfp8_quantize` CUDA kernel on Blackwell.

Result (byte-wise comparison of the `(uint8 scale, fp8 data)` tuples):

```
(converter) vs (TE):           scale-byte-identical: True  data-byte-identical: True
(converter) vs (vLLM runtime): scale-byte-identical: True  data-byte-identical: True
(TE)        vs (vLLM runtime): scale-byte-identical: True  data-byte-identical: True
```

**All three quantizers produce byte-identical output for the same bf16
input.** First-8-of-row-0 of every scale tensor is
`[114, 114, 114, 114, 114, 114, 114, 114]` — identical across all three.

So:

- The weight quantizer **is not the source of the L0 drift.** The
  formula reads `ceil(log2(amax/448)) + 127` everywhere; on Blackwell the
  hardware PTX op implements it directly.
- Per Section "Implementation" in
  [`mxfp8_bi_matmul_design.md`](mxfp8_bi_matmul_design.md), the
  *converter* uses the explicit `ceil(log2(amax/desc_max))` while the
  Python *runtime fallback* in `mxfp8_utils.py:_mxfp8_e4m3_quantize_torch`
  uses `floor(log2(amax))` — but the runtime fallback is only used on
  non-Blackwell hardware. On Blackwell (our target), FlashInfer's CUDA
  kernel is selected and matches TE byte-for-byte.

⚠️ Caveat: my first diagnostic run *appeared* to show a 1.24 max_abs diff
between TE-quantized and converter-quantized dequanted tensors. That was
a script bug: TE returns `_rowwise_data` as `torch.uint8` (not
`torch.float8_e4m3fn`), and my dequant did
`.to(torch.float32)` directly on the uint8, interpreting bytes as
integers 0–255 rather than as fp8 floats. Once
`.view(torch.float8_e4m3fn)` is applied first, the dequanted bf16 is
byte-identical too. The underlying conclusion is unchanged: the bytes
match.

### Updated diagnosis: activation quantization is the asymmetry

The L0 drift at `linear_proj` input therefore cannot come from weight
quantization. Re-reading the two paths:

| | vLLM `BatchInvariantMxfp8LinearKernel.apply_weights` | Megatron `install_mxfp8_dequant_for_bi_gemm` |
|---|---|---|
| Weight | MXFP8 → dequant → bf16 | bf16 (HF ckpt) → TE on-the-fly quant → MXFP8 → our wrapper dequant → bf16 |
| Activation | bf16 (untouched) | bf16 → TE on-the-fly quant → MXFP8 → our wrapper dequant → bf16 |

The two engines' weight handling is now numerically equivalent
(quantizers byte-identical, both dequant via the same formula).

But **Megatron quantizes the activation on the fly via `fp8_autocast`**;
vLLM does not. Megatron's activation goes through a *lossy* MXFP8
round-trip (bf16 → fp8 → bf16) before reaching the BF16 BI matmul, while
vLLM's activation stays in bf16. This lossy round-trip is the source of
the L0 `linear_proj` input drift, which compounds layer-by-layer.

### Recommended next step

Skip activation quantization on the Megatron side. Two implementation
candidates:

1. **Replace `MXFP8Quantizer.quantize_impl` to short-circuit for the
   activation path.** Detect "input activation" calls (e.g. by a flag on
   the quantizer, or by tensor identity) and return a wrapper holding the
   original bf16 directly; the wrapper's `.dequantize()` returns the
   bf16 unchanged. Megatron's weight quantization continues to round-trip
   through MXFP8 (so weights stay equivalent to vLLM's).
2. **Disable `fp8_autocast` entirely + lossy-round-trip the weights
   once at model-load time.** Pre-bake each linear weight via
   `dequant(quant(w_bf16))` using TE's quantizer (which, per the
   diagnostic, matches vLLM's converter byte-for-byte). Skip
   `provider.fp8` and `provider.fp8_recipe`. Megatron then runs as a
   plain BF16 model with lossy weights; activations are untouched.

Option 2 is cleaner experimentally (no need to instrument TE's
quantizer with conditional logic) and matches vLLM's behavior exactly
(both engines: lossy bf16 weight × bf16 activation → BF16 BI matmul).
Option 1 keeps the `fp8_autocast` plumbing alive (potentially useful for
training-mode debugging later).

### File and patch inventory (updated)

| Artefact | Path | Purpose |
|---|---|---|
| BI MXFP8 kernel | `3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py` | vLLM dequant + BF16 BI matmul |
| BI MXFP8 registration | `3rdparty/vllm/vllm/model_executor/kernels/linear/__init__.py` | First in `_POSSIBLE_MXFP8_KERNELS[CUDA]` |
| Compact-scales patch | `my_script/megatron_forward.py::install_mxfp8_compact_scales` | Property-based override of `MXFP8Quantizer.optimize_for_gemm` |
| Dequant-for-BI hook | `my_script/megatron_forward.py::install_mxfp8_dequant_for_bi_gemm` | Dequant TE MXFP8 → bf16 at GEMM hook, route to BF16 BI matmul |
| Quantizer diagnostic | `my_script/compare_mxfp8_quant.py` | Standalone vLLM ↔ TE quantizer comparison |
| Diagnostic log | `my_script/compare_mxfp8_quant.log` | Confirms quantizers are byte-identical |
| MXFP8 BI L0 capture (Megatron) | `my_script/megatron_capture_mxfp8_split_vllmrope_vllmswiglu_vllmsdpa_vllmrmsnorm_bi.pt` | With dequant + BF16 BI matmul |
| MXFP8 BI L0 capture (vLLM) | `my_script/vllm_capture_mxfp8_bi.pt` | With BatchInvariantMxfp8LinearKernel |
| L0 compare log | `my_script/compare_mxfp8_bi_dequant_L0.log` | Drift starts at `linear_proj` input (post-SDPA) |

---

## Update 2026-06-01 (continued): W8A8 activation round-trip in vLLM → L0–L19 bit-identical

Following the diagnostic that ruled out the weight quantizer as the source
of drift, modified vLLM's `BatchInvariantMxfp8LinearKernel.apply_weights`
to also lossy-round-trip the **activation** through MXFP8 — mirroring what
Megatron's `fp8_autocast` already does per linear call. Both engines now
do W8A8: bf16 weight quant→dequant + bf16 activation quant→dequant +
BF16 BI matmul.

### Patch (vLLM)

`3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py`:

```python
def _quant_dequant_bf16_via_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """Lossy bf16 -> MXFP8 -> bf16 round-trip on activations.
    Mirrors TE's per-call activation quantisation under fp8_autocast."""
    x_2d = x.reshape(-1, x.shape[-1])
    x_fp8, x_scale = mxfp8_e4m3_quantize(x_2d, is_sf_swizzled_layout=False)
    x_bf16 = dequant_mxfp8_to_bf16(x_fp8, x_scale)
    return x_bf16.reshape(x.shape)

# In apply_weights, after weight dequant and before matmul_persistent:
x_bf16 = _quant_dequant_bf16_via_mxfp8(x)
out_2d = matmul_persistent(x_bf16.reshape(-1, x_bf16.shape[-1]), weight_bf16.t())
```

The quantizer used (`mxfp8_e4m3_quantize` → FlashInfer on Blackwell) was
confirmed byte-identical to TE's `tex.quantize` in
`my_script/compare_mxfp8_quant.py`, so the lossy round-trip on the
activation produces the same bf16 value on both engines for the same
input.

### Result

Configuration unchanged on the Megatron side:
`uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py --mxfp8 --batch-invariant --split-fused --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm --capture-layers 0`.

| Pair | max_abs | mean_abs |
|---|---|---|
| Final logits (last position, padded — see caveat) | 23.2 | 3.42 (cos -0.29) |
| Per-layer residual stream L0 | **0** | 0 |
| Per-layer residual stream L1–L19 | **0** | 0 |
| Per-layer residual stream L20 | 0.0313 | 1.16e-3 |
| Per-layer residual stream L25 | 0.250 | 9.79e-3 |
| Per-layer residual stream L30 | 0.625 | 2.47e-2 |
| Per-layer residual stream L31 | **0.898** | 3.77e-2 |

L0 module-level (positions 0–10):

| Tensor | max_abs |
|---|---|
| `input_layernorm` input (embedding) | **0** |
| `linear_qkv.linear` input (post first RMSNorm) | **0** |
| `linear_proj` input (post-SDPA) | **0** |
| `linear_fc1.linear` input (post second RMSNorm) | **0** |
| `linear_fc2` input (post-SwiGLU) | **0** |

Every comparable L0 module input is bit-identical. The known module-
boundary semantic mismatch (`post_attention_layernorm ↔
pre_mlp_layernorm`) still shows max_abs ≈ 0.25 because vLLM captures
pre-add `attn_out` while Megatron captures the post-add residual — not a
real numerical divergence.

### Comparison vs prior MXFP8 attempts

| Attempt | L0 drift | L31 drift |
|---|---|---|
| Baseline (FlashInfer cutlass vs TE cuBLASLt) | max=0.00146 | max=2.0 |
| Dequant + BF16 BI (W8A16: weight only) | max=0.0469 | max=12.0 |
| Dequant + BF16 BI (**W8A8: weight + activation**) | **max=0** for L0–L19 | max=0.898 |

The W8A8 round-trip is the right pattern — it produces:
- Bit-identical residual stream for the first **20 layers** of the
  network.
- ~13× smaller L31 drift vs the W8A16 attempt and ~2× smaller than the
  original FlashInfer-cutlass-vs-TE-cuBLASLt baseline.

### Why drift enters at L20

Layers 0–19 are bit-identical because every step (RMSNorm, RoPE, SDPA,
attn output projection, residual add, MLP linears, SwiGLU) produces
byte-identical outputs given byte-identical inputs. At L20 the residual
stream first picks up ~1 bf16 ULP of drift. This is the same threshold-
sensitive pattern we saw in BF16 mode at L6 (variance reduction
divergence in `fused_add_rms_norm` cub-tree vs Triton tl.sum): a kernel
that *happens* to agree at smaller magnitudes diverges past some
threshold as the residual stream grows.

The L20 drift is small enough that the next investigation can localise
it the same way we did the BF16 L6 case — capture per-module inputs on
both L0 and L20 (`--capture-layers 0,20`) and look for the first module
whose input is bit-identical but output drifts.

The most likely candidate: vLLM's activation quantiser (FlashInfer
`mxfp8_quantize`) and TE's `tex.quantize` are byte-identical *for the
q_proj weight tensor* (verified) but may differ at specific
larger-magnitude activation distributions. A targeted diff of the
quantised activation at L20 between the two engines would settle this.

### Remaining caveats

- ⚠️ `compare.py`'s final-logits row is still misleading (compares
  `m["logits"][0, -1]` which is a padded position on Megatron). Doesn't
  affect the per-layer/module table.
- ⚠️ Without `--capture-layers N` for N ≥ 20, we can't yet localise which
  L20 submodule introduces the drift.

### Updated file/patch inventory (additions)

| Artefact | Path | Purpose |
|---|---|---|
| W8A8 round-trip helper | `3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py::_quant_dequant_bf16_via_mxfp8` | Activation lossy round-trip via FlashInfer mxfp8_quantize + dequant_mxfp8_to_bf16 |
| Updated kernel | `3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py` (W8A8) | Adds the activation round-trip before the BF16 BI matmul |
| L0 W8A8 compare log | `my_script/compare_mxfp8_bi_w8a8_L0.log` | All comparable L0 tensors bit-identical, residual stream bit-identical L0–L19 |
| vLLM W8A8 capture | `my_script/vllm_capture_mxfp8_bi.pt` (regenerated) | With activation round-trip |

### Recommended next step

Capture L20's per-module inputs on both engines (`--capture-layers 0,20`)
and run compare; the first module whose input is bit-identical but
output drifts is the kernel to fix next. This is the same workflow that
localised the BF16 L6 RMSNorm divergence.

---

## Update 2026-06-01 (final): `--split-all-fused` → bit-identical across all 32 layers

The L20 capture identified the divergence inside L20's attention chain
(post-SDPA `linear_proj` input drifted by 0.0039). Re-ran Megatron with
`--split-all-fused` (which unfuses TE's `LayerNormColumnParallelLinear`
at every layer rather than just layer 0) — same flag that closed BF16
parity in
[`llama3_8b_numeric_mismatch.md`](llama3_8b_numeric_mismatch.md). vLLM
capture unchanged.

### Result: all 32 layers bit-identical

```
layer | max_abs |  cos_sim  | |vllm|    | |mcore|
    0 | 0.0     |  0.999994 |  12.1958  |  12.1958
    1 | 0.0     |  1.000033 | 544.0219  | 544.0219
   ...
   19 | 0.0     |  1.000021 | 549.9084  | 549.9084
   20 | 0.0     |  1.000017 | 550.3365  | 550.3365   (was 0.031 with --split-fused)
   ...
   30 | 0.0     |  0.999996 | 564.8802  | 564.8802
   31 | 0.0     |  0.999996 | 260.4788  | 260.4788   (was 0.898 with --split-fused)
```

L0 and L20 per-module diffs (positions 0–10):

| Module pair | L0 max_abs | L20 max_abs |
|---|---|---|
| `linear_qkv.linear` input (post first RMSNorm) | **0** | **0** |
| `linear_proj` input (post-SDPA) | **0** | **0** |
| `linear_fc1.linear` input (post second RMSNorm) | **0** | **0** |
| `linear_fc2` input (post-SwiGLU) | **0** | **0** |

Every comparable module input is bit-identical at both layers. The
non-zero diffs that remain
(`post_attention_layernorm ↔ pre_mlp_layernorm` and `input_layernorm` at
L20) are the documented module-boundary semantic mismatch (vLLM captures
pre-add `attn_out`; Megatron captures post-add residual stream) — not
real numerical divergence.

### Why `--split-all-fused` matters for MXFP8 too

The same fusion-boundary effect from the BF16 work applies here. TE's
`LayerNormColumnParallelLinear` performs RMSNorm + Linear in one fused
kernel and **keeps the post-RMSNorm activation in fp32 registers** until
the GEMM consumes it. vLLM has standalone RMSNorm; the post-RMSNorm
tensor is materialised in bf16 and re-read by `qkv_proj` /
`gate_up_proj` — i.e., it goes through a bf16 round-trip.

For BF16 GEMMs the difference was ~1 bf16 ULP per layer. For MXFP8
GEMMs the difference is amplified at the **activation-quantiser
boundary**: the quantiser sees fp32-precision input on the fused side
and bf16-precision input on the unfused (vLLM) side, so the per-block
amax differs slightly → the E8M0 scale lands on a different bucket for
some blocks → the resulting (fp8_data, scale) tuples disagree, and the
quant-then-dequant gives different bf16 outputs.

`--split-all-fused` forces Megatron to materialise the post-RMSNorm
bf16 tensor at every layer, matching vLLM's behaviour. Both engines'
quantisers then see byte-identical bf16 input, produce byte-identical
MXFP8 output, and the BF16 BI matmul yields byte-identical GEMM output.

### Complete recipe for MXFP8 BI cross-engine bit-identity

vLLM:
```bash
uv run --extra vllm python my_script/vllm_forward.py \
    --mxfp8 \
    --model /path/to/Llama-3.1-8B-Instruct-mxfp8 \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --batch-invariant --capture-layers 0,20
```

The vLLM patches required (all already in place from BF16 work + this
MXFP8 work):
- `BatchInvariantMxfp8LinearKernel` (W8A8: weight + activation lossy
  round-trip via `mxfp8_e4m3_quantize` + `dequant_mxfp8_to_bf16`, then
  BF16 BI matmul). Registered first in `_POSSIBLE_MXFP8_KERNELS[CUDA]`.
- The BF16-era patches inside `vllm/model_executor/layers/layernorm.py`
  (`RMSNorm.forward_cuda` BI-Triton routing under `VLLM_BATCH_INVARIANT`).

Megatron:
```bash
uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py \
    --mxfp8 --batch-invariant \
    --split-all-fused \
    --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm \
    --capture-layers 0,20
```

The Megatron patches required:
- `install_mxfp8_compact_scales()` (property override on
  `MXFP8Quantizer.optimize_for_gemm`) — auto-applied with
  `--mxfp8 --batch-invariant`.
- `install_mxfp8_dequant_for_bi_gemm()` (TE `general_gemm` hook: dequant
  → bf16 → `BatchInvariantTEGemmFn`) — auto-applied.
- `install_vllm_style_{rope,swiglu,sdpa,rmsnorm}` (BF16-era patches).
- `--split-all-fused` (unfuses LN+Linear at **every** layer, not just
  L0).
- Seq pad to 32 (auto-applied for MXFP8).

### Comparison vs prior MXFP8 attempts

| Attempt | L0 drift | L20 drift | L31 drift |
|---|---|---|---|
| Baseline (FlashInfer cutlass vs TE cuBLASLt, no BI patches) | 0.00146 | n/a | 2.0 |
| Dequant + BF16 BI (W8A16 — weight only) | 0.0469 | n/a | 12.0 |
| Dequant + BF16 BI (W8A8 — weight + activation, --split-fused) | 0 | 0.031 | 0.898 |
| **Dequant + BF16 BI (W8A8, --split-all-fused)** | **0** | **0** | **0** |

### Caveats

- `compare.py`'s final-logits row remains misleading under MXFP8 (Megatron
  pads seq 11 → 32 for the MXFP8 block-size constraint; `compare.py` takes
  `m["logits"][0, -1]` which is a padded position). The per-layer table
  is authoritative. Cleanup todo: capture `real_seq_len` in the payload
  and use it in `compare.py`.
- This parity is established for prefill of an 11-token prompt with TP=PP=1
  on a single GB200. Multi-GPU and multi-prompt batched generation
  haven't been verified in the BI MXFP8 path; per-block scale layouts
  may behave differently when activations cross block boundaries
  asymmetrically.

### Updated file/patch inventory (additions)

| Artefact | Path | Purpose |
|---|---|---|
| Final MXFP8 BI L0+L20 vLLM capture | `my_script/vllm_capture_mxfp8_bi.pt` | W8A8 round-trip; matches Megatron L0–L31 |
| Final MXFP8 BI L0+L20 Megatron capture | `my_script/megatron_capture_mxfp8_splitall_vllmrope_vllmswiglu_vllmsdpa_vllmrmsnorm_bi.pt` | --split-all-fused; all 32 layers bit-identical to vLLM |
| Final compare log | `my_script/compare_mxfp8_bi_splitall_L0L20.log` | Per-layer residual stream max_abs=0 for all 32 layers |

### Headline

> **MXFP8 cross-engine bit-identity achieved on Llama-3.1-8B-Instruct.**
> All 32 decoder-layer residual streams match exactly between vLLM
> (with `BatchInvariantMxfp8LinearKernel` doing W8A8 round-trip) and
> Megatron (with `--mxfp8 --batch-invariant --split-all-fused
> --vllm-{rope,swiglu,sdpa,rmsnorm}` + `install_mxfp8_{compact_scales,
> dequant_for_bi_gemm}`). The fix mirrors the BF16 result: align all
> kernel families *and* eliminate the TE fused LN+Linear's fp32
> register-only post-norm tensor by unfusing at every layer.
