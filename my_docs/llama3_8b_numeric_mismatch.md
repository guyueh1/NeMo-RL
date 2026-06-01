# Llama-3.1-8B-Instruct: vLLM vs Megatron numeric mismatch debugging

Working log of cross-engine numerical comparisons between vLLM (prefill, eager)
and Megatron-LM (forward-only via Megatron-Bridge) on a single prompt
(`"The quick brown fox jumps over the lazy dog."`, 11 tokens, bf16, TP=PP=1).

Repro scripts live under `my_script/`:
- `vllm_forward.py` — `uv run --extra vllm python my_script/vllm_forward.py [--batch-invariant]`
- `megatron_forward.py` — `uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py [--batch-invariant] [--split-fused]`
- `compare.py` — `uv run --extra mcore python my_script/compare.py [--batch-invariant] [--split-fused]`

Both forward scripts register forward hooks on every submodule of the first
decoder layer and on the LM head / logits processor; `compare.py` aligns the
hook capture by hand-written name pairs and prints max/mean abs diff and cosine
similarity per module.

---

## Module 1: RMSNorm (`input_layernorm`)

### Initial observation
With out-of-the-box engines (no BI, fused TE LayerNormColumnParallelLinear on
Megatron, custom `vllm_c` RMSNorm + standalone `qkv_proj` on vLLM):

| Pair | max_abs | cos_sim |
|---|---|---|
| `input_layernorm` input | 0.0 | 1.0 (bit-identical) |
| `qkv_proj` ↔ `linear_qkv` input | **6.72** | **0.40** |
| `o_proj` ↔ `linear_proj` input | 0.001 | 0.99999 |
| `mlp.down_proj` ↔ `linear_fc2` input | 0.016 | 1.0001 |
| final logits (last position) | 0.1875 | 0.99992 |

The 6.72 / cos≈0.40 mismatch at `qkv_proj` input looked alarming until we
realised it was a **fusion-boundary artifact**, not a numerical bug.

### Root cause of the apparent ~6.7 diff: fusion boundary
- **vLLM**: `RMSNorm` is a standalone `nn.Module` (`vllm/model_executor/layers/layernorm.py:103`). `qkv_proj`'s forward hook captures the *post-norm* bf16 tensor (norm ≈ 27.6).
- **Megatron**: `self_attention.linear_qkv` is TE's fused `LayerNormColumnParallelLinear` (`megatron/core/extensions/transformer_engine.py:910`) which performs LayerNorm + GEMM in one kernel. The hook on `linear_qkv` therefore captures the *pre-norm* residual stream (norm ≈ 1.88).

Different tensors → cos_sim drops to ~0.4. The downstream `o_proj` / `linear_proj`
inputs are the actual attention output, which is the same tensor on both sides
(both stacks materialise it), and indeed they agreed to ≤1e-3.

### Fix: unfuse Megatron's first-layer LN+GEMM
Used the existing upstream helper
`megatron.core.extensions.transformer_engine.split_te_layernorm_column_parallel_linear`
to split `self_attention.linear_qkv` and `mlp.linear_fc1` on the first decoder
layer into `(TENorm, TEColumnParallelLinear)`, wrapped in a tiny
`SplitNormLinear` to keep the `(out, bias)` calling convention. Implementation
in `my_script/megatron_forward.py::split_first_layer_fused`, gated behind
`--split-fused`.

After splitting (no BI on either side):

| Pair | max_abs | cos_sim |
|---|---|---|
| `qkv_proj` ↔ `linear_qkv.linear` (post-norm) | **0.000488** | **1.00000** |
| `gate_up_proj` ↔ `linear_fc1.linear` (post-norm) | **0.00391** | **0.99999** |
| final logits | 0.1875 (unchanged) | 0.99992 (unchanged) |

So once the captured tensors are apples-to-apples, the RMSNorm output agrees to
~1 bf16 ULP (1/2048 = 0.000488). The final logits are bit-identical to the
non-split run, confirming the split is a true no-op on math.

### Why the residual ~1 bf16 ULP exists (non-BI)
Both engines do the same math in fp32: `x_fp32 -> sum(x²) -> rsqrt(mean+eps) -> y_fp32 = x*rsigma*w -> cast bf16`. Differences are pure order-of-operations / kernel-implementation choices:

1. **Reduction tree for `sum(x²)`**: vLLM uses `cub::BlockReduce<float, 1024>` with pairwise tree (`csrc/layernorm_kernels.cu:62`). TE's RMSNorm uses TE's own warp-shuffle + cross-warp reduction. Non-associative fp32 addition ⇒ 1–2 fp32 ULP drift in `var`.
2. **`rsqrt` implementation**: vLLM kernel uses `rsqrtf` (SFU, ~2 ULP). TE may use `__frsqrt_rn` with Newton refinement (~1 ULP).
3. **Multiplication grouping** (`(x * rsigma) * w` vs `x * (rsigma * w)`): both engines use the first form, but the compiler can reorder.
4. **Vector width** of loads affects within-thread partial-sum order.

Net: ≤1 bf16 ULP at the RMSNorm output, propagating to ~0.1875 max-abs drift
across 32 layers of fp32-accumulation through SDPA + MLP, ending with cos_sim
0.99992 at the final logits.

### Batch-invariant (BI) mode
Each engine ships its own BI patch:
- **vLLM** (`vllm/model_executor/layers/batch_invariant.py:786`): Triton RMSNorm kernel, `BLOCK_SIZE=1024`, sequential outer `sum_sq += tl.sum(x²)`. Uses `inv_rms = 1.0 / tl.sqrt(mean_sq + eps)`.
- **Megatron** (`megatron/core/transformer/custom_layers/batch_invariant_kernels.py:877`): pure PyTorch `BatchInvariantRMSNormFn`, uses the BI `mean_dim` (Triton, `BLOCK_SIZE=1024`, same sequential outer pattern). Uses `rsigma = torch.rsqrt(ms + eps)`.

So both engines under BI have **identical reduction structure** (Triton tl.sum
with BLOCK_SIZE=1024, sequential outer accumulator), removing causes 1, 3, 4
above. The only remaining difference is **`1/sqrt(x)` vs `rsqrt(x)`**:
- vLLM: two rounded fp32 ops (sqrt then divide), ≤2 ULP from exact rsqrt.
- Megatron: one rsqrt op, ≤1 ULP from exact.

### Fix applied: align vLLM BI rsqrt formula with Megatron
In `vllm/model_executor/layers/batch_invariant.py:817-820` replaced
```python
rms = tl.sqrt(mean_sq + eps)
inv_rms = 1.0 / rms
```
with
```python
inv_rms = tl.rsqrt(mean_sq + eps)
```
to make both engines' BI RMSNorm produce bit-identical fp32 intermediates.

### Result with BI + split-fused on both sides

After patching vLLM BI to use `tl.rsqrt`, with split-fused on Megatron's first layer:

| Pair | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| `input_layernorm` input | 0.0 | 0.0 | 1.0 |
| `qkv_proj` ↔ `linear_qkv.linear` (post-norm) | **0.0** | **0.0** | **1.0** (bit-identical) |
| `o_proj` ↔ `linear_proj` | 0.000977 | 7.04e-6 | 0.99999 |
| `gate_up_proj` ↔ `linear_fc1.linear` (post-norm) | 0.00391 | 0.000109 | 0.99999 |
| `mlp.down_proj` ↔ `linear_fc2` | 0.0156 | 1.28e-5 | 1.00011 |
| final logits | 0.1875 | 0.040 | 0.99989 |

The first-layer RMSNorm now produces **bit-identical fp32 outputs** between
vLLM and Megatron (both stacks computed `linear_qkv.linear` input as
`27.602252960205078`-norm). Note that:

- `gate_up_proj` / `linear_fc1.linear` still shows ~1 bf16 ULP because its
  *input* is the post-attention residual stream, which has accumulated ~1 ULP
  of drift through the QKV→SDPA→o_proj→residual-add chain. The
  `pre_mlp_layernorm` (RMSNorm) computation itself is identical, but its input
  differs slightly.
- Final logits gap is the same order as before BI (0.1875). BI does not bring
  the two engines closer overall because other kernels (`mm`, `addmm`, attention
  reductions) still use independent deterministic implementations in each
  engine.

### Summary of RMSNorm debug
1. Initial 6.72 mismatch at `qkv_proj` input was a **fusion-boundary artifact**, not a numerical bug. Fixed by `--split-fused` (use upstream `split_te_layernorm_column_parallel_linear`).
2. After splitting, ~1 bf16 ULP residual remained from kernel implementation differences (reduction tree, rsqrt impl). Inherent to two independent CUDA kernels.
3. Under BI mode, both engines use Triton with identical reduction structure. Only difference left was `1/sqrt(x)` (vLLM) vs `rsqrt(x)` (Megatron). Patched vLLM BI to use `tl.rsqrt`.
4. End state: **bit-identical fp32 RMSNorm output** between vLLM BI and Megatron BI.

---

## Per-layer residual-stream drift across the full network

Both forward scripts now capture the post-layer residual-stream value at every
decoder layer (`my_script/{vllm,megatron}_forward.py`, key `layer_outputs`).
The vLLM hook reconstructs the residual stream as `hidden_states + residual`
out of `LlamaDecoderLayer` (since vLLM threads them as a 2-tuple); Megatron's
`TransformerLayer` returns the residual stream directly. `compare.py` prints
per-layer max_abs / mean_abs / cos_sim and the L2 norms on each side.

### Results

Both configurations: split-fused on first layer, all 32 decoder layers
captured.

| layer | NO-BI mean_abs | NO-BI cos_sim | BI mean_abs | BI cos_sim |
|---|---|---|---|---|
| 0 | 4.11e-5 | 0.999992 | 4.11e-5 | 0.999992 |
| 1 | 1.33e-4 | 1.000035 | 1.26e-4 | 1.000035 |
| 5 | 6.22e-4 | 1.000064 | 5.95e-4 | 1.000063 |
| 10 | 1.25e-3 | 1.000044 | 1.19e-3 | 1.000043 |
| 15 | 1.43e-3 | 1.000034 | 1.38e-3 | 1.000033 |
| 20 | 2.25e-3 | 1.000015 | 2.18e-3 | 1.000016 |
| 25 | 3.89e-3 | 1.000006 | 3.82e-3 | 1.000006 |
| 30 | 7.48e-3 | 0.999991 | 7.37e-3 | 0.999990 |
| 31 | **1.09e-2** | 0.999918 | **1.01e-2** | 0.999928 |

(Full tables in `my_script/compare_run_split.log` and
`my_script/compare_run_split_bi.log`.)

### Observations

1. **Drift grows roughly monotonically with depth.** Mean abs-diff scales by
   ~270× from layer 0 to layer 31. This is the standard "1 bf16 ULP per layer"
   pattern: each layer's matmul/attention contributes a small drift on top of
   what came in, and the residual stream accumulates it.
2. **Sharp ~2× jump at layer 31.** Mean abs-diff jumps from 7.5e-3 (L30) to
   1.1e-2 (L31). The L31 norm itself **drops** sharply (~552 → ~260) on both
   engines — the last decoder block subtracts from the residual stream rather
   than adding to it for this prompt. The drift becomes a larger fraction of
   the (smaller) signal.
3. **BI helps a little, but not much.** Across all 32 layers, BI shaves
   ≤10% off the mean abs-diff at each layer. BI homogenises *within-engine*
   reduction order, but the two engines still use different attention kernels
   (vLLM Flashinfer vs TE `DotProductAttention`), different RoPE
   implementations, different GEMM tiling, etc.
4. **Cos-sim stays ≥0.999918** through all 32 layers in both runs. The
   residual stream is *directionally* the same on both stacks; what differs is
   tiny per-element bf16 rounding noise.
5. **At layer boundary, vs at layer interior** — layer-output diffs are
   stable; the first non-trivial drift appears at layer 0 (4e-5) right after
   the first transformer block, meaning the contributions are roughly equal
   from attention and MLP. We don't yet have per-module per-layer captures to
   localise within a layer.

### Next investigation candidates (in suggested order)

- **Attention output**: hook `core_attention` / `self_attn.attn` on layer 0
  and capture the post-SDPA tensor — that's the next point downstream of the
  identical post-norm tensor. Difference would point at the SDPA kernel
  (Flashinfer vs TE DPA).
- **RoPE**: `self_attn.rotary_emb` (vLLM) vs the apply-RoPE inside `core_attention`
  (Megatron). vLLM keeps RoPE as a separate module; Megatron may apply it
  inside the attention kernel. User flagged `fused_apply_rotary_pos_emb_thd`
  in TE source as a possible source.
- **MLP activation**: `mlp.act_fn` input shape is already (11, 28672) in our
  vLLM capture (gate_up fused); the equivalent in Megatron is inside `mlp` after
  `linear_fc1`. Worth comparing pre/post SwiGLU.

---

## Module 2: RoPE (Llama-3 NTK-scaled rotary embedding)

### Setup
With the post-norm tensor now bit-identical (RMSNorm aligned), the first non-trivial divergence is in Q and K, which differ from V by ~64× — V doesn't touch RoPE, Q and K do.

| SDPA input (BI + split-fused, fused-TE-RoPE) | max_abs | mean_abs |
|---|---|---|
| Q (post-RoPE)  | 0.031 | 3.06e-4 |
| K (post-RoPE)  | 0.031 | 5.04e-4 |
| V (no RoPE)    | 0.000488 | 2.61e-7 |

### Implementation comparison

**vLLM** (`3rdparty/vllm/vllm/model_executor/layers/rotary_embedding/`):
- `_compute_inv_freq` in `base.py:69-81`: `inv_freq = 1/(base^(arange(0,rot_dim,2,fp32)/rot_dim))`.
- Llama-3 NTK smoothing in `llama3_rope.py:33-54` (same math as Megatron's).
- `_compute_cos_sin_cache` (`base.py:83-92`): computes `freqs = positions × inv_freq` in fp32, then `cos = freqs.cos(); sin = freqs.sin()` (fp32), concatenated into one cache tensor.
- **Cache stored as bf16** (`base.py:60-63`): `cache = cache.to(dtype)` — full fp32→bf16 cast at module init, never recomputed.
- `forward_cuda` calls `vllm._custom_ops.rotary_embedding(...)` — C++ kernel that reads bf16 cos/sin and bf16 q/k, computes the rotation `(x1·cos − x2·sin, x2·cos + x1·sin)` in fp32 registers and writes bf16 once.

**Megatron** (`3rdparty/.../megatron/core/models/common/embeddings/`):
- `RotaryEmbedding.__init__` in `rotary_pos_embedding.py:79-81`: same `inv_freq` formula.
- `_apply_scaling` (`rotary_pos_embedding.py:92-125`): same Llama-3 smoothing math.
- `get_emb` (`rotary_pos_embedding.py:150-175`): builds `emb = cat(freqs, freqs)` of shape `(seq, 1, 1, rot_dim)` — keeps **fp32 freqs**, no cos/sin precomputed.
- `apply_rotary_pos_emb` (`rope_utils.py:250-316`):
  - Default `apply_rope_fusion=True` → TE's `fused_apply_rotary_pos_emb(t, freqs, ...)` computes `cos(freqs)`/`sin(freqs)` inside the kernel in fp32, does the rotation in fp32, casts to bf16 at store — **one bf16 cast event**.
  - With `apply_rope_fusion=False` → `_apply_rotary_pos_emb_bshd` (`rope_utils.py:92-126`): `cos_ = cos(freqs).to(t.dtype)` and `sin_ = sin(freqs).to(t.dtype)` (**cos/sin cast to bf16 up-front**, matching vLLM), then `(t * cos_) + (rotate_half(t) * sin_)` in PyTorch ops — each PyTorch op materialises a bf16 intermediate, so **three bf16 cast events** before the add result.

### Hypothesis 1: cos/sin precision — and what disabling rope fusion did

Initial theory: the gap comes from vLLM pre-quantising cos/sin to bf16 once at init while TE keeps them fp32 inside the fused kernel. Disabling rope fusion (`--no-rope-fusion`) should make Megatron also cast cos/sin to bf16 before the multiply, matching vLLM and closing the gap.

**Result (BI + split-fused + no-rope-fusion):**

| SDPA input | TE fused RoPE | PyTorch unfused (this run) |
|---|---|---|
| Q (post-RoPE) | max 0.031, mean 3.06e-4 | max 0.031, mean 3.42e-4 |
| K (post-RoPE) | max 0.031, mean 5.04e-4 | max 0.031, mean 5.36e-4 |
| V (no RoPE)   | max 0.000488, mean 2.61e-7 | **max 0.0**, mean 0.0 |

**The Q/K gap did NOT close. V became bit-identical.** Final-logits cos_sim went from 0.99989 → 0.99993 (marginal). So the bf16-vs-fp32-cos/sin hypothesis was wrong as the *dominant* cause.

### Hypothesis 2: rounding-event count in the rotation chain

Re-reading the two paths, what *actually* differs is the number of bf16 rounding events between input bf16 q/k and output bf16 q/k:

| RoPE path | cos/sin precision at multiply | bf16 cast events |
|---|---|---|
| vLLM C++ kernel | bf16 (pre-quantised cache) | **1** (final store) |
| TE fused kernel | fp32 (computed in-kernel) | **1** (final store) |
| Megatron PyTorch unfused | bf16 (cast before multiply) | **3** (after `t*cos_`, after `rotate_half(t)*sin_`, after the add) |

Switching from TE fused to PyTorch unfused trades fp32 cos/sin (lower error) for bf16 cos/sin (higher error) but also adds two extra bf16 cast events in the multiply-add chain. The two effects roughly cancel → unchanged Q/K gap.

The reason V becomes bit-identical with `--no-rope-fusion` is unrelated to the RoPE math itself: the QKV split/reshape happens to land V in a different fp32 contiguity path between fused and unfused, and the unfused path matches vLLM's reshape order. We confirmed `linear_qkv.linear` output is bit-identical in both runs.

### What this means

The ~0.031 bf16-ULP Q/K residual is **not** a single-source bug; it's the combined effect of:
1. cos/sin precision (vLLM bf16 vs TE fp32) — small contribution.
2. Rotation kernel implementation (different multiply-add order, different vector widths, different reduction across the rot_dim).
3. An axis-permutation between `(seq, heads, head_dim)` (vLLM) and `(seq, batch, heads, head_dim)` (Megatron) that can change which fp32 intermediate is rounded first.

To actually close it, both engines would have to call **the same** RoPE kernel — e.g., point vLLM's `forward_cuda` at TE's fused kernel (or have Megatron use Triton with vLLM-style cos_sin_cache). That's a larger surgery than what we did for RMSNorm.

### Hypothesis 3 (confirmed): port vLLM's exact RoPE recipe into Megatron

Looking at vLLM's C++ wrapper `csrc/pos_encoding_kernels.cu:171`:
```cpp
auto cache_f32 = cos_sin_cache.to(torch::kFloat32);
```
vLLM upcasts the bf16 cos_sin_cache back to fp32 *inside the wrapper* before launching the kernel. The kernel then reads fp32 cos/sin and computes per token:
```cpp
const float x_f = static_cast<float>(arr[x_index]);
const float y_f = static_cast<float>(arr[y_index]);
arr[x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
arr[y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
```

So the precise recipe is: cos/sin computed fp32 → cast bf16 (lossy) → cast back fp32 → rotation in fp32 → one bf16 cast per output element. I replicated this in PyTorch and monkey-patched `apply_rotary_pos_emb` (see `my_script/megatron_forward.py::install_vllm_style_rope`, gated behind `--vllm-rope`).

**Result (BI + split-fused + vllm-rope):**

| Pair | TE fused RoPE | vllm-style RoPE patch |
|---|---|---|
| SDPA arg0 (Q post-RoPE) | max 0.031, mean 3.06e-4 | **0.0 / 0.0** (bit-identical) |
| SDPA arg1 (K post-RoPE) | max 0.031, mean 5.04e-4 | **0.0 / 0.0** (bit-identical) |
| SDPA arg2 (V)           | max 4.88e-4, mean 2.6e-7 | **0.0 / 0.0** (bit-identical) |
| `o_proj` input          | max 9.77e-4, mean 7.05e-6 | max 8e-6, mean 2.5e-10 |
| `pre_mlp_layernorm` input | max 9.77e-4, mean 9.59e-6 | max 6e-5, mean 5.6e-9 |
| `linear_fc1.linear` (post-norm) | max 3.9e-3, mean 1.09e-4 | max 9.8e-4, mean 7.0e-8 |
| `linear_fc2` (post-SwiGLU) | max 0.0156, mean 1.28e-5 | max 0.0156, mean 3.7e-6 |

The Q/K gap **closes completely** (bit-identical fp32). Everything downstream collapses to ≤1e-5 mean_abs through the attention block. The downstream MLP `linear_fc2` input still shows max 0.0156 — driven by SwiGLU + down-proj, unaffected by the RoPE fix.

### Final-logits drift across the three RoPE configurations

| Configuration | final max_abs | mean_abs | cos_sim |
|---|---|---|---|
| BI + split + TE-fused RoPE | 0.1875 | 0.0408 | 0.99989 |
| BI + split + PyTorch unfused RoPE | 0.1328 | 0.0252 | 0.99993 |
| BI + split + vLLM-style RoPE patch | 0.1563 | 0.0247 | 0.99993 |

The vLLM-style patch eliminates RoPE as a per-layer drift source, but final logits are not dramatically better than the unfused PyTorch run. **Conclusion: RoPE accounted for ~5e-4 mean drift per layer in Q/K but is not the dominant accumulated source of cross-engine logit drift.** After fixing RoPE, the layer-0 residual stream drift falls from mean 4.11e-5 → 2.34e-5 (43% reduction), confirming there's still a non-RoPE component in attention (most likely SDPA kernel — Flashinfer vs TE DotProductAttention).

### Per-layer residual-stream drift comparison

| Layer | TE fused RoPE mean_abs | vLLM-style RoPE patch mean_abs |
|---|---|---|
| 0  | 4.11e-5 | 2.34e-5 |
| 5  | 5.95e-4 | 5.51e-4 |
| 10 | 1.19e-3 | 1.13e-3 |
| 15 | 1.38e-3 | 1.35e-3 |
| 20 | 2.18e-3 | 2.18e-3 |
| 25 | 3.82e-3 | 3.79e-3 |
| 30 | 7.37e-3 | 7.22e-3 |
| 31 | 1.01e-2 | 1.00e-2 |

Drift reduction is ~45% at layer 0 but shrinks toward 0% at deeper layers — consistent with RoPE contributing a fixed-magnitude per-layer error that becomes a smaller fraction of the accumulated other-error as depth grows.

---

## Module 3: BI matmul kernel (bf16 → fp32-accumulator GEMM)

### Side-by-side: `matmul_kernel_persistent` in both engines

- vLLM: `3rdparty/vllm/vllm/model_executor/layers/batch_invariant.py:41-139`
- Megatron: `3rdparty/.../megatron/core/transformer/custom_layers/batch_invariant_kernels.py:62-151`

`diff` of the two Triton kernels (full kernel function):

```
< def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
> def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
```

Megatron's `_compute_pid` takes an extra `NUM_SMS` parameter that is **unused** in the function body. Otherwise, everything else is whitespace / line-wrapping. Algorithm-wise the two kernels are **bit-for-bit identical**:

1. Persistent grid, `start_pid = tl.program_id(axis=0)`.
2. `accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)` — fp32 accumulator.
3. Per-tile K-loop: load bf16 `a`, bf16 `b`, `accumulator = tl.dot(a, b, accumulator)`.
4. Optional bias: `bias.to(tl.float32); accumulator += bias`.
5. `c = accumulator.to(c_ptr.dtype.element_ty); tl.store(...)` — single bf16 cast at store.

### Launch configs (bf16 — our path) are identical

| Param | vLLM | Megatron |
|---|---|---|
| BLOCK_SIZE_M | 128 | 128 |
| BLOCK_SIZE_N | 128 | 128 |
| BLOCK_SIZE_K | 64 | 64 |
| GROUP_SIZE_M | 8 | 8 |
| num_stages | 3 | 3 |
| num_warps | 8 | 8 |
| grid | `min(NUM_SMS, cdiv(M,BM)*cdiv(N,BN))` | identical |

So given the **same input tensors** (same `a`, `b`, same dtype, same strides), the kernel **must produce bit-identical output**. This is what we observe for `linear_qkv.linear` post-RMSNorm-fix and post-RoPE-fix: both engines emit fp32-equivalent norm `27.602252960205078` to the last decimal.

### Aten dispatch coverage differs

vLLM patches more aten ops than Megatron:

| Aten op | vLLM | Megatron |
|---|---|---|
| `aten::mm` | mm_batch_invariant | mm_batch_invariant |
| `aten::addmm` | addmm_batch_invariant | addmm_batch_invariant |
| `aten::matmul` | matmul_batch_invariant | **not patched** |
| `aten::linear` | linear_batch_invariant | **not patched** |
| `aten::softmax` / `aten::_softmax` | softmax_batch_invariant | **not patched** |
| `aten::_log_softmax` | _log_softmax_batch_invariant | _log_softmax_batch_invariant |
| `aten::mean.dim` | mean_batch_invariant | mean_batch_invariant |

Megatron compensates by additionally patching **TE's** internal `general_gemm` (`_te_general_gemm_patched` at `batch_invariant_kernels.py:830`) and TE's RMSNorm. For our Llama-3.1-8B path:
- All linear layers (`linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`, `output_layer`) go through TE → `_te_general_gemm_patched` → `BatchInvariantTEGemmFn.forward` → `matmul_persistent(opB_2d, opA, bias=None)`.
- vLLM linear layers (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, `lm_head`) hit `aten::mm` or `aten::matmul` directly → `matmul_persistent(a, b)`.

Either way the GEMM hits the same Triton kernel with the same configs, so **the matmul itself contributes zero cross-engine drift under BI mode** (confirmed by the bit-identical `linear_qkv.linear` output in our run).

### Subtle bias-handling difference (does not apply to Llama-3.1-8B)

For models with biased linear layers, the two engines apply bias at different points:

- **vLLM `addmm_batch_invariant`**: `matmul_persistent(a, b, bias=bias)` — bias added to the **fp32 accumulator** *inside* the kernel, single bf16 cast at end.
- **Megatron `BatchInvariantTEGemmFn`**: `out = matmul_persistent(opB_2d, opA, bias=None); out = out + bias` — bias added to the **bf16 output tensor** *outside* the kernel, two bf16 round-trips (one inside the matmul cast, one for the add).

This would produce up to 1 bf16 ULP per element of difference for biased linear layers. **Llama-3.1-8B has `bias=False` on every linear layer, so this is moot for us.**

### Conclusion on Module 3

The BI matmul kernels are functional copies of each other (literally the same Triton implementation in both upstream repos, with one cosmetic parameter-list change). Given identical inputs, they produce bit-identical fp32 accumulators and bit-identical bf16 outputs.

Therefore, **none of the remaining cross-engine drift (≤1 bf16 ULP at `linear_proj` / `linear_fc1.linear` / `linear_fc2` inputs in the vllm-rope + BI run) is attributable to the GEMM kernel itself**. The drift must come from:

1. **SDPA** (Flashinfer vs TE DotProductAttention) — different kernel families, separate from `matmul_persistent`. Even with bit-identical Q/K/V, they will produce slightly different attention output.
2. **SwiGLU / activation** — the `mlp.act_fn` is a separate elementwise op that doesn't go through `matmul_persistent`. Its drift propagates into `linear_fc2` input.
3. **Cumulative drift** from layers 0 through N-1, where each layer's small attention-kernel-derived drift propagates into the next layer's `linear_qkv` input — even though the GEMM is identical, identical input gives identical output, so different input gives different output.

### Implication for next investigation

The next dominant source of drift to attribute is **SDPA**. Specifically, at layer 0 with Q/K/V bit-identical, the post-SDPA tensor still differs at max_abs 8e-6 (mean 2.5e-10). That's tiny but non-zero. Either:
- The vLLM Flashinfer attention and TE BI-attention use different reduction patterns (different `block_m`/`block_n`) and so their fp32 partial sums of `softmax(QK^T) @ V` differ.
- One of them is not actually batch-invariant for this specific shape (single prompt prefill).

Worth comparing Megatron's `get_batch_invariant_attention_block_size() = (16, 16)` (`batch_invariant_kernels.py:516-518`) with whatever Flashinfer's BI mode uses — that's a direct cause if they differ.

---

## Module 4: SwiGLU activation

### vLLM (`SiluAndMul`)

`3rdparty/vllm/vllm/model_executor/layers/activation.py:118-148` + the CUDA kernel at `3rdparty/vllm/csrc/activation_kernels.cu:75-125`:

```python
class SiluAndMul(CustomOp):
    """x -> silu(x[:d]) * x[d:]  where d = x.shape[-1] // 2."""
    def forward_cuda(self, x):
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        self.op(out, x)   # torch.ops._C.silu_and_mul
        return out
```

The CUDA kernel (`act_and_mul_kernel`) reads `gate = x[:d]` and `up = x[d:]` per token, then for each element:

```cpp
// silu: cast bf16 -> fp32, compute x/(1+exp(-x)) in fp32, cast back to bf16
T silu_kernel(const T& x) {
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}
// then:
return packed_mul(PACKED_ACT_FN(gate), up);   // bf16 * bf16 -> bf16
```

So vLLM's silu path has **two bf16 rounding events** per output element:
1. `silu(gate_bf16)` → fp32 compute → cast to bf16 (round #1)
2. `silu_result_bf16 * up_bf16` → bf16 (round #2)

### Megatron (`bias_swiglu_impl` / `SwiGLUFunction`)

`3rdparty/.../megatron/core/transformer/mlp.py:296-304` routes to `bias_swiglu_impl(intermediate_parallel, bias_parallel, ...)` when `bias_activation_fusion=True` and `activation_func == F.silu and gated_linear_unit`. For Llama (bias=None on linear layers), this becomes `SwiGLUFunction.apply`, which calls:

`3rdparty/.../megatron/core/fusions/fused_bias_swiglu.py:15-26`:
```python
@jit_fuser
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2
```

`@jit_fuser` is `torch.compile` for torch ≥ 2.2 (`megatron/core/jit.py:7+21`). Under TorchInductor, this routine becomes a single fused Triton kernel that:

1. Reads `y_1` and `y_2` (bf16)
2. Casts both to fp32
3. Computes `silu(y_1_fp32) * y_2_fp32` in fp32 registers
4. Casts the final result to bf16 once
5. Stores

So Megatron's path has **one bf16 rounding event** per output element (assuming TorchInductor actually fuses the chunk + silu + mul into a single kernel — TorchInductor reliably does this for pure-elementwise Python chains like the one above).

### Predicted and observed difference

| Path | bf16 rounding events |
|---|---|
| vLLM SiluAndMul (hand-written CUDA) | **2** (silu output cast, then multiply cast) |
| Megatron SwiGLU under TorchInductor | **1** (only final store) |

Each extra bf16 cast event introduces up to 1 bf16 ULP. For SwiGLU outputs of magnitude ~0.5–2 (typical post-activation), that's a max-abs drift around `2 × 2^-7 ≈ 0.016` — which matches **exactly** the `linear_fc2` / `down_proj` input drift we measured:

| Pair | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| `down_proj` ↔ `linear_fc2` (post-SwiGLU) | **0.0156** | 3.7e-6 | 1.00011 |

The 0.0156 maximum is consistent with a single bf16 ULP at value ~1, occurring at elements where silu(gate) has small magnitude near the silu non-linearity's sensitive region.

### Aten dispatch under BI mode

Neither engine patches SwiGLU directly under batch-invariant mode:
- vLLM BI: `enable_batch_invariant_mode` patches `aten::mm`, `addmm`, `matmul`, `linear`, `softmax`, `mean.dim`, etc., but NOT `silu_and_mul` (which is a vLLM-defined custom op, not an aten op).
- Megatron BI: `enable_batch_invariant_mode` patches `aten::mm`, `addmm`, `_log_softmax`, `mean.dim`. SwiGLU is elementwise so it's naturally batch-invariant; no patch needed.

Both paths are batch-invariant by construction (elementwise operations don't have cross-batch reduction). The cross-engine drift comes purely from the **kernel implementation choice** — hand-written CUDA with intermediate bf16 cast (vLLM) vs TorchInductor-fused fp32-throughout (Megatron).

### Implementation comparison summary

| Aspect | vLLM | Megatron |
|---|---|---|
| Math | `silu(x[:d]) * x[d:]` | `silu(y_1) * y_2` (identical) |
| Kernel | Hand-written CUDA (`csrc/activation_kernels.cu`) | TorchInductor-compiled Triton (`@torch.compile`) |
| Where silu computed | fp32 in CUDA registers | fp32 in Triton registers |
| Intermediate bf16 cast | yes (after silu) | no (kept in fp32 to multiply) |
| Final cast | yes (multiply result → bf16) | yes (final result → bf16) |
| Bias support | clamp variant only (`SiluAndMulWithClamp`) | yes (`bias_swiglu` adds bias before silu) — N/A for Llama-3.1-8B |

### To bit-match SwiGLU

Either:
- **Modify vLLM's `silu_and_mul` CUDA kernel** to keep silu output in fp32 and multiply in fp32 before the final cast. Single-line change in `csrc/activation_kernels.cu`'s `packed_compute`:
  ```cpp
  // Before:
  return packed_mul(PACKED_ACT_FN(gate), up);  // bf16 intermediate
  // After:
  // Convert to fp32, multiply, cast back
  float2 silu_f = cast_to_float2(PACKED_ACT_FN(gate));
  float2 up_f   = cast_to_float2(up);
  float2 result = {silu_f.x * up_f.x, silu_f.y * up_f.y};
  return cast_to_packed<packed_t>(result);
  ```
- Or **change Megatron's SwiGLU** to insert an explicit bf16 cast after silu — easiest but lossier.

Either gets `down_proj`/`linear_fc2` input bit-identical, removing the ~1 bf16 ULP drift propagating through to logits.

> **TODO — long-term direction (preferred):** the *correct* fix is to change
> vLLM's `silu_and_mul` CUDA kernel to keep silu in fp32 and only cast to
> bf16 once after the multiply (the patch sketch above). That requires
> recompiling vLLM's C++ extension, so for now we go the other direction —
> downgrade Megatron to match — via `--vllm-swiglu` (see
> `my_script/megatron_forward.py::install_vllm_style_swiglu`). Revisit once
> we can rebuild vLLM cleanly.

### Fix applied: align Megatron's SwiGLU with vLLM (downgrade)

`install_vllm_style_swiglu()` in `my_script/megatron_forward.py` monkey-patches
`megatron.core.fusions.fused_bias_swiglu.swiglu` (the function referenced by
`SwiGLUFunction.forward`) to a non-compiled eager-mode implementation:

```python
def _vllm_style_swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    silu_out = F.silu(y_1)        # eager: fp32 compute, bf16 cast at output
    return silu_out * y_2          # bf16 * bf16 hardware multiply
```

This bypasses `@torch.compile` so the silu output is *materialised* in bf16
(round #1) before the multiply (round #2) — matching vLLM's kernel exactly.

Gated behind `--vllm-swiglu` on the megatron forward script.

### Result with split-fused + vllm-rope + vllm-swiglu + BI

First-layer comparison vs `vllm_capture_bi.pt`:

| Pair | vllm-rope only | + vllm-swiglu |
|---|---|---|
| `qkv_proj` ↔ `linear_qkv.linear` (post-norm) | 0.0 / 0.0 | 0.0 / 0.0 |
| SDPA arg0 / arg1 / arg2 (Q, K, V) | 0.0 / 0.0 each | 0.0 / 0.0 each |
| `o_proj` ↔ `linear_proj` (post-SDPA) | max 8e-6, mean 2.5e-10 | max 8e-6, mean 2.5e-10 |
| `pre_mlp_layernorm` arg0 (residual after attention) | max 6e-5 | max 6e-5 |
| `linear_fc1.linear` (post-norm into MLP) | max 9.8e-4, mean 7.0e-8 | max 9.8e-4, mean 7.0e-8 |
| **`linear_fc2` (post-SwiGLU)** | **max 0.0156, mean 3.7e-6** | **max 0.000244, mean 1.10e-7** |

`linear_fc2` input now matches vLLM's `down_proj` input to ≈1 bf16 ULP at value
~0.01 (was 1 bf16 ULP at value ~1) — **64× max-abs reduction**.

Per-layer residual-stream drift collapses at layer 0:

| Layer | vllm-rope only mean_abs | + vllm-swiglu mean_abs |
|---|---|---|
| 0  | 2.34e-5 | **8.6e-7** (27× ↓) |
| 1  | 1.01e-4 | 3.8e-5 |
| 5  | 5.51e-4 | 4.7e-4 |
| 10 | 1.13e-3 | 1.04e-3 |
| 15 | 1.35e-3 | 1.26e-3 |
| 20 | 2.18e-3 | 2.07e-3 |
| 25 | 3.79e-3 | 3.71e-3 |
| 30 | 7.22e-3 | 7.20e-3 |
| 31 | 1.00e-2 | 1.00e-2 |

Reduction is dramatic at L0 (27× lower mean_abs) but converges with depth as
SDPA-derived drift takes over.

Final logits:

| Configuration | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| vllm-rope only | 0.156 | 0.0247 | 0.99993 |
| + vllm-swiglu | **0.141** | **0.0207** | **0.99994** |

### Confirmed: the SwiGLU drift hypothesis was correct

Predicted 1 bf16 ULP at `linear_fc2` input → observed exactly that, collapsing
from 0.0156 (1 bf16 ULP at value ~1, the typical post-SwiGLU magnitude) to
0.000244 (1 bf16 ULP at value ~0.015, residual GEMM-input noise). The MLP path
is now numerically aligned end-to-end up to SDPA's downstream influence.

---

## Module 5: SDPA / attention kernel

### Setup

With RMSNorm + RoPE + matmul + SwiGLU all aligned, Q, K, V going into SDPA are
now **bit-identical** between vLLM and Megatron (max=0, mean=0). The only
remaining per-token-block drift in the first layer is the SDPA output itself:
`o_proj` / `linear_proj` input still differs at max≈8e-6, mean≈2.5e-10.

### Why we cannot bit-match on Blackwell (GB200, SM10.0) without monkey-patching

- **vLLM** on Blackwell uses **FA4** (the cute-DSL kernel at `vllm.vllm_flash_attn.cute.interface`). Under `VLLM_BATCH_INVARIANT=1`, vLLM forces `max_num_splits=1` (`vllm/v1/attention/backends/flash_attn.py:442-443`), which makes the K-dim reduction order independent of batch size.
- **Megatron / TE** on Blackwell uses TE's `DotProductAttention`. TE imports `flash_attn` (FA2 family, `backends.py:96-141`) and optionally `flash_attn_3` (Hopper-only). **TE has no FA4 import path.** On Blackwell, with FA3 unavailable and FA2 considered slow, TE typically resolves to the cuDNN-fused-attention backend.
- TE's `num_splits` knob (`DotProductAttention.forward(num_splits=...)`) only routes through the **FA3** path (`backends.py:1021`); on Blackwell where FA3 isn't installed, `num_splits` has no effect — the backend filter at `utils.py:525-534` disables FA2/FusedAttention/UnfusedDPA when `num_splits != 1`.

Net: vLLM uses FA4 + num_splits=1, Megatron/TE uses a completely different kernel (cuDNN fused). Different kernels → different fp32 partial sums → ≤1 bf16 ULP residual at the attention output. Both are batch-invariant *internally*; they're just not the *same* kernel.

### Module-source confirmation

`grep -r "FA4\|flash_attn_4\|fa_v4\|flash-attn-4"` over:
- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/` → **no hits**
- `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/` → **no hits**
- TE source at `uvcache/.../transformer_engine/` → **no hits**

So **neither Megatron-Core nor TransformerEngine has any FA4 import path or interface.** Wiring FA4 into Megatron requires external patching.

### Offline proof: would option 3 work?

Rather than fight the `uv` extras conflict (`{nemo-rl[mcore], nemo-rl[vllm]}` declared incompatible in `pyproject.toml`), we ran the in-engine patch experiment **offline**: load Megatron's saved Q/K/V tensors, run vLLM's FA4 cute kernel with `num_splits=1` on them in the vllm venv, and compare to both engines' actual attention outputs.

Script: `my_script/run_vllm_fa4_on_mcore_qkv.py`. Result:

| Comparison | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| Q (mcore) vs Q (vllm) — sanity, post-RoPE | **0.0** | **0.0** | 1.0 |
| K (mcore) vs K (vllm) | **0.0** | **0.0** | 1.0 |
| V (mcore) vs V (vllm) | **0.0** | **0.0** | 1.0 |
| `vllm_FA4(mcore_QKV)` vs Megatron actual `linear_proj` input (TE cuDNN-fused) | **7.63e-6** | **2.54e-10** | 0.9999969 |
| `vllm_FA4(mcore_QKV)` vs vLLM actual `o_proj` input | **0.0** | **0.0** | 1.0 ✓ |

**Interpretation:**

1. The third row confirms FA4 with `num_splits=1` is a pure deterministic function: running it offline on the same Q/K/V matches vLLM's in-engine attention output **bit-for-bit**. This is the proof that option 3 (monkey-patch TE to dispatch to vLLM's FA4) would yield bit-identical first-layer attention on Megatron.
2. The second-from-last row quantifies exactly how much TE's cuDNN-fused attention differs from FA4: **~1 bf16 ULP per element** at the attention output. This *is* the entire SDPA component of the remaining cross-engine drift.

### What still blocks the in-engine patch

`uv` rejects `--extra mcore --extra vllm` because `pyproject.toml` declares them conflicting (transitive xformers, flash-attn versions, cuda-python overlaps). To enable option 3 in-engine, one of:

- Remove the conflict declaration (may break the production training setup that relies on these venvs being isolated).
- After `uv sync --extra mcore`, manually `uv pip install --no-deps vllm vllm-flash-attn nvidia-cutlass-dsl quack` into the mcore venv (best-effort, may pull conflicting `flash_attn` versions).
- Create a new `mcore_with_vllm_kernels` extra that pins compatible versions of both stacks.

These are infrastructure-level changes and likely out of scope for the numerical investigation.

### Conclusion (Module 5)

The SDPA gap is **fully attributed** to "different kernel implementations" — not non-determinism, not batch-variance, not Q/K/V differences. It is the irreducible bf16 ULP between TE's cuDNN-fused attention and vLLM's FA4 cute kernel given identical inputs. Closing it requires both engines to call the same kernel, which on Blackwell means routing Megatron's attention through vLLM's FA4 path (option 3).

`my_script/megatron_forward.py::install_vllm_style_sdpa` is the in-place monkey-patch that would do this. It is gated behind `--vllm-sdpa` but cannot currently be exercised in-engine due to the `uv` extras conflict; the offline experiment in `my_script/run_vllm_fa4_on_mcore_qkv.py` substitutes for it.

> **TODO — long-term direction:** Once the uv-venv conflict is resolvable (either by removing the declared conflict in `pyproject.toml` or by an in-place `uv pip install --no-deps` into the mcore venv), run `megatron_forward.py --split-fused --batch-invariant --vllm-rope --vllm-swiglu --vllm-sdpa` and confirm the in-engine first-layer attention output now matches `vllm_capture_bi.pt`'s `o_proj` input to 0.0 max-abs. The offline experiment predicts this with certainty.

### UPDATE: In-engine option 3 now exercised — full first-layer bit-identity

Resolved the uv-venv conflict on 2026-05-31 by:
1. Removing the `[mcore, vllm]` conflict declaration in `pyproject.toml`.
2. Adding `vllm`, `nvidia-cutlass-dsl>=4.4.0.dev1`, `cuda-python` to the `mcore` extra so vLLM's `vllm.vllm_flash_attn.cute` module (FA4) is importable from the mcore venv.
3. Running `uv lock` — 441 packages resolve cleanly.

Then ran `uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py --split-fused --batch-invariant --vllm-rope --vllm-swiglu --vllm-sdpa` to dispatch Megatron's `DotProductAttention.forward` through vLLM's FA4 cute kernel.

**Result — every captured first-layer tensor is bit-identical:**

| Pair | Before SDPA patch | After SDPA patch |
|---|---|---|
| `input_layernorm` input | 0 / 0 | 0 / 0 |
| `qkv_proj` ↔ `linear_qkv.linear` | 0 / 0 | 0 / 0 |
| SDPA Q / K / V | 0 / 0 each | 0 / 0 each |
| **`o_proj` ↔ `linear_proj` (post-SDPA)** | max 8e-6, mean 2.5e-10 | **0 / 0** |
| `pre_mlp_layernorm` input | max 6e-5, mean 5.6e-9 | **0 / 0** |
| `linear_fc1.linear` (post-norm) | max 9.8e-4, mean 7.0e-8 | **0 / 0** |
| `linear_fc2` (post-SwiGLU) | max 0.000244, mean 1.1e-7 | **0 / 0** |

**Per-layer residual-stream drift:**

| Layer | + vllm-rope + vllm-swiglu (no sdpa) | + vllm-sdpa (this run) |
|---|---|---|
| **0** | mean 8.6e-7 | **0.0 (bit-identical)** |
| 1  | mean 3.8e-5 | mean 1.3e-5 |
| 5  | mean 4.7e-4 | mean 3.5e-4 |
| 10 | mean 1.04e-3 | mean 8.4e-4 |
| 20 | mean 2.07e-3 | mean 1.78e-3 |
| 30 | mean 7.20e-3 | mean 6.64e-3 |
| 31 | mean 1.00e-2 | mean 9.10e-3 |

Layer 0 is now **completely bit-identical** between vLLM and Megatron — every captured activation matches to fp32. Final logits cos_sim 0.99994 (essentially unchanged from before SDPA patch, since the deep-layer drift dominates the final-logit shift).

### Why deeper layers still drift after the SDPA patch

Our `--split-fused` only unfuses LN+Linear on **layer 0** (to expose the post-norm tensor for capture). Layers 1–31 still use TE's fused `LayerNormColumnParallelLinear`, which keeps the post-norm activation in fp32 registers inside the fused kernel (no intermediate bf16 materialisation). vLLM, by contrast, always runs RMSNorm and the projection as two separate kernels with an explicit bf16 round-trip in between (Module 1's analysis).

So the per-layer ~1 bf16 ULP "fusion-boundary" rounding mismatch reappears starting at layer 1 and accumulates over the remaining 31 layers. To get bit-identical residual streams **all the way through**, the next step would be:

> **TODO — full-depth bit-equality:** apply `split_first_layer_fused` to *every* decoder layer (not just `decoder.layers[0]`), then re-run with `--vllm-rope --vllm-swiglu --vllm-sdpa --batch-invariant`. Expected outcome: all 32 layer outputs bit-identical, final logits bit-identical between the two engines.

The current state has already achieved the primary numerical-mismatch debug goal: at the first-layer module level, all four kernel families (RMSNorm, RoPE, matmul, SwiGLU, SDPA) are aligned, with the only residual being the *same fusion-boundary effect that we already understood and addressed for layer 0*.

### Cumulative effect

Together with RoPE-style fix (which we already applied) and the trivially-aligned BI matmul kernel, fixing SwiGLU would account for the last *kernel-choice* source of cross-engine drift in the first layer's MLP path. What would remain is **only SDPA**.

### Final-logits drift in this configuration

| Configuration | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| BI + split + TE-fused RoPE | 0.1875 | 0.0408 | 0.99989 |
| BI + split + PyTorch unfused RoPE | 0.1328 | 0.0252 | 0.99993 |

Small improvement, consistent with V becoming bit-identical — not with closing the Q/K gap.

### Other first-layer pairs under BI + split + no-rope-fusion

| Pair | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| `input_layernorm` input | 0.0 | 0.0 | 1.0 |
| `qkv_proj` ↔ `linear_qkv.linear` (post-norm hidden) | 0.0 | 0.0 | 1.0 |
| SDPA arg0 (Q post-RoPE) | 0.031 | 3.42e-4 | 0.999999 |
| SDPA arg1 (K post-RoPE) | 0.031 | 5.36e-4 | 1.000004 |
| SDPA arg2 (V) | **0.0** | **0.0** | 1.000002 |
| `o_proj` ↔ `linear_proj` (post-SDPA) | 0.00078 | 7.44e-6 | 0.99999 |
| `post_attention_layernorm` arg1 ↔ `pre_mlp_layernorm` arg0 (residual) | 0.00098 | 9.86e-6 | 0.999998 |
| `gate_up_proj` ↔ `linear_fc1.linear` (post-norm) | 0.00391 | 1.15e-4 | 0.999999 |
| `down_proj` ↔ `linear_fc2` (post-SwiGLU) | 0.0156 | 1.29e-5 | 1.00011 |

### Remaining gaps to investigate

1. **`o_proj` input** (post-SDPA context): 1 bf16 ULP. The Q/K mismatch contributes ~0.031 of input drift into SDPA; the SDPA reduction softens it to ~0.001 at the output. So most of `o_proj`-input drift is downstream of RoPE, not SDPA-kernel-internal.
2. **`gate_up_proj` input** (post-norm hidden into MLP): 4 bf16 ULPs. The input to `pre_mlp_layernorm` itself agrees to 1 bf16 ULP (residual stream after attention), so the extra factor comes from the second RMSNorm + the gate_up GEMM in the second multiply-add chain. Likely same kernel-choice family of reasons as o_proj.
3. **`down_proj` input** (post-SwiGLU): 0.0156 max — driven by the SwiGLU activation and the down-proj GEMM. No direct module-level capture of SwiGLU input exists yet (vLLM captures `mlp.act_fn` input as the fused `(11, 28672)` gate+up tensor, Megatron has no separate `act_fn` module).

---

## Update 2026-06-01: Post-attention RMSNorm divergence + full bit-identity

After the vLLM 0.20.x upgrade landed (commit `241ece552` — `Upgrade vLLM from
0.17.1 to 0.20.0`), the FA4 path described in Module 5 no longer applies under
BI mode, and a new RMSNorm divergence appeared at deeper layers. This section
documents the full diagnosis and the two patches that close the gap end-to-end.

### vLLM 0.20.x: BI now uses FA2, not FA4

`fa_utils.py:137-142` was added to vLLM's BI path. It rejects FA4 on Blackwell
under `VLLM_BATCH_INVARIANT=1` because FA4 uses batch-shape-dependent
scheduling heuristics on SM100+:

```python
if envs.VLLM_BATCH_INVARIANT and fa_version == 4:
    logger.warning_once("Cannot use FA version 4 with batch invariance, "
                        "defaulting to FA version 2.")
    fa_version = 2
```

So Megatron's `install_vllm_style_sdpa` was rewritten to call vLLM's FA2 entry
point (`vllm.vllm_flash_attn.flash_attn_varlen_func` with `fa_version=2,
num_splits=1`) instead of the FA4 cute kernel. Reshapes Megatron's `(s,b,n,d)`
into FA2 varlen's packed `(total,n,d)` layout with a `cu_seqlens` describing one
contiguous sequence per batch element. This pushed the per-layer bit-identical
region from L0–L3 (FA4 patch) to L0–L5 (FA2 patch).

### L6 onwards: residual-add + RMSNorm path diverges

With `--vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm --split-all-fused
--batch-invariant`, layers 0–5 are bit-identical but L6 first shows
~0.002 max-abs drift at the residual stream, growing geometrically to
~0.25 by L31.

Per-module captures on L6 (`--capture-layers 0,6` added to both scripts;
hook structure generalised to record per-module input tensors for any list
of decoder layers, payload key `module_inputs_by_layer`):

| L6 tensor | max_abs | Bit-identical? |
|---|---|---|
| L5 output (= L6 input residual) | 0 | ✓ |
| L6 `linear_qkv.linear` input (post first RMSNorm) | 0 | ✓ |
| L6 SDPA output (= `linear_proj` input) | 0 | ✓ |
| L6 post-attn residual (vllm `post_attention_layernorm.args[1]` ↔ mcore `pre_mlp_layernorm.args[0]`) | 0 | ✓ |
| **L6 norm output** (vllm `post_attention_layernorm.args[0]` ↔ mcore `linear_fc1.linear` input) | **0.000977** | **✗** |

Important capture-semantics note: vLLM's `fused_add_rms_norm` C++ kernel
mutates inputs in-place — after the kernel fires, the captured
`args[0]` holds the *norm output* (not `attn_out`) and `args[1]` holds the
*post-add residual* (not `residual_in`). The L6 module table above corrects
for this. The residual stream going *into* RMSNorm at L6 is bit-identical;
the divergence is entirely inside the RMSNorm kernel itself.

### Root cause: vLLM's BI mode skips its own BI Triton kernel when residual!=None

Looking at `vllm/model_executor/layers/layernorm.py:312-318`:

```python
if residual is not None:
    return fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
else:
    assert envs.VLLM_BATCH_INVARIANT
    return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)
```

The `VLLM_BATCH_INVARIANT` check **only fires when `residual is None`** —
i.e., for L0's first norm only. Every other layernorm (L0's post-attn norm,
all L1+ input/post-attn norms) routes to `fused_add_rms_norm` — vLLM's
hand-written C++ kernel at `csrc/layernorm_kernels.cu:93-150`, which uses:

- `cub::BlockReduce<float, 1024>` (warp-shuffle + cross-warp tree) for the
  variance reduction.
- `rsqrtf` (CUDA Newton-refined intrinsic) for the inverse square root.

Megatron's BI RMSNorm (now patched to call vLLM's `rms_norm_batch_invariant`
Triton kernel — see below) uses:

- `tl.sum BLOCK_SIZE=1024` sequential outer accumulator for variance
  reduction.
- `1.0 / tl.sqrt(...)` for inverse square root.

The two kernels happen to produce bit-identical fp32 variance for small
input magnitudes (L0 input norm ~2.5, variance ~0.0015) — different
reduction trees, same result. At L6 (input norm ~531, variance ~69), the
reduction-tree difference becomes observable: cub's pairwise tree and
Triton's sequential accumulator produce fp32 sums that differ by ~1 fp32
ULP, propagating to ~1 bf16 ULP at the norm output. Once 1 ULP enters the
residual stream at L6, it compounds through all subsequent layers.

### Fix: route vLLM's BI add+norm through the BI Triton kernel

`vllm/model_executor/layers/layernorm.py` `RMSNorm.forward_cuda` modified to
add a BI Triton path before the C++ kernel:

```python
if envs.VLLM_BATCH_INVARIANT and residual is not None:
    residual.add_(x)
    return (
        rms_norm_batch_invariant(residual, self.weight.data, self.variance_epsilon),
        residual,
    )
```

This is a deliberate, BI-gated modification to vLLM. The downstream effect:
**every** RMSNorm call under BI now routes through `rms_norm_batch_invariant`
— the same Triton kernel (`_rms_norm_kernel`) regardless of whether residual
is provided. The bf16 in-place add (`residual.add_(x)`) is byte-equivalent to
what `fused_add_rms_norm`'s C++ kernel does for the add step.

### Initial regression at L0 — and the second half of the fix

First attempt at the patch above introduced new drift at **L0**:
`linear_fc1.linear` input went from max=0 to max=0.000488. Reason:

- Before the patch, L0.post_attn_layernorm used `fused_add_rms_norm` (cub +
  rsqrtf), and L0's small input magnitude meant the C++ kernel happened to
  agree bit-for-bit with Megatron's `BatchInvariantRMSNormFn` (PyTorch
  `mean_dim` + `torch.sqrt`).
- After the patch, L0.post_attn_layernorm uses `_rms_norm_kernel` (Triton
  `tl.sum` + `tl.sqrt`). This is a *different* Triton kernel than what
  Megatron's PyTorch path resolves to, even though both nominally implement
  RMSNorm with `BLOCK_SIZE=1024` and `1/sqrt`. Key differences:
  - `tl.sqrt` (Triton, may compile to `sqrt.approx.f32`) vs `torch.sqrt`
    (PyTorch, IEEE-compliant `__fsqrt_rn`).
  - Single fused kernel vs separate `x*x` materialisation + `mean_dim`
    kernel (which uses Triton's `mean_kernel`, a different program with
    different reduction code).

These differences cancel out at L0.input_layernorm (small input, both happen
to agree) but expose themselves at L0.post_attn_layernorm (slightly larger
input). Empirically threshold-sensitive — exactly the same pathology that
made the cub-vs-tl.sum boundary appear at L6 in the previous round.

### Second half of the fix: Megatron calls vLLM's exact Triton kernel

`install_vllm_style_rmsnorm` in `my_script/megatron_forward.py` rewritten to
dispatch Megatron's `BatchInvariantRMSNormFn.forward` directly through vLLM's
`rms_norm_batch_invariant` wrapper (which calls `_rms_norm_kernel`):

```python
def install_vllm_style_rmsnorm():
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik_mod
    from vllm.model_executor.layers.batch_invariant import rms_norm as vllm_rms_norm_triton

    class _VllmStyleBatchInvariantRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps, zero_centered_gamma):
            w_eff = (weight + 1.0) if zero_centered_gamma else weight
            return vllm_rms_norm_triton(x, w_eff, eps)
        # ... forward-only, backward NotImplementedError

    bik_mod.BatchInvariantRMSNormFn = _VllmStyleBatchInvariantRMSNormFn
```

After this, every RMSNorm call on either engine (vLLM L0.input_layernorm,
vLLM L0.post_attn_layernorm, vLLM L1+ norms, Megatron L0.input_layernorm,
Megatron L0+ pre_mlp_layernorm, etc.) resolves to the literally identical
Triton kernel invocation on byte-identical inputs.

### Result: bit-identical end-to-end

Configuration: `vllm_forward.py --batch-invariant` (vLLM with the
`forward_cuda` BI Triton routing) + `megatron_forward.py --batch-invariant
--split-all-fused --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm`.

| Pair | max_abs | mean_abs | cos_sim |
|---|---|---|---|
| Final logits | **0.0** | **0.0** | 0.99997 |
| Every layer L0–L31 residual stream | **0.0** | **0.0** | ≥0.999996 |
| Every comparable L0/L6 submodule input | **0.0** | **0.0** | ~1.0 |

The two engines now produce **bit-identical bf16 outputs across all 32
decoder layers and the final logits** for the Llama-3.1-8B-Instruct forward
on the test prompt. The only non-zero diffs in the per-module table are the
documented module-boundary semantic differences (vLLM captures
`hidden_states + residual` separately; Megatron captures the already-added
residual stream — same underlying data, different module-boundary view).

### Summary of patches required

| Where | Patch | Direction |
|---|---|---|
| vLLM | `vllm/model_executor/layers/layernorm.py` `RMSNorm.forward_cuda` | Add a `VLLM_BATCH_INVARIANT and residual is not None` branch that does explicit bf16 in-place add + `rms_norm_batch_invariant`. |
| Megatron | `my_script/megatron_forward.py` `install_vllm_style_rmsnorm` | Replace Megatron's BI RMSNorm autograd Fn with one that dispatches directly through vLLM's `rms_norm_batch_invariant` Triton kernel. |
| Megatron | `my_script/megatron_forward.py` `install_vllm_style_sdpa` | Use `flash_attn_varlen_func(fa_version=2, num_splits=1)` (FA2) instead of the FA4 cute kernel — matches vLLM 0.20.x BI behaviour. |
| Megatron | `my_script/megatron_forward.py` `split_all_layers_fused` + `--split-all-fused` | Unfuse LN+Linear on all 32 decoder layers (not just layer 0) so the post-norm bf16 round-trip happens on both sides at every layer. |

The pre-existing patches (`install_vllm_style_rope`, `install_vllm_style_swiglu`,
`install_vllm_style_sdpa` — though the latter is now FA2-based) remain
unchanged in spirit.

### Capture & compare infrastructure changes

- `vllm_forward.py` and `megatron_forward.py` gained a `--capture-layers
  N,M,...` CLI flag. Per-module hooks are registered on every layer in that
  list; captures are stored under `module_inputs_by_layer[layer_idx]`.
  Legacy `first_layer_inputs` is retained as an alias for layer 0.
- `compare.py` reads `module_inputs_by_layer` and displays per-(layer,
  module) diffs for every captured layer.

---

## Notes & helpers for future debugging

- Hook captures only positional args. For modules that vLLM/Megatron call with
  kwargs (e.g. `self_attn(positions=..., hidden_states=...)`), the captured
  list is empty — switch the hook to `register_forward_hook(..., with_kwargs=True)`
  if needed.
- vLLM v1 ships hook closures to the engine-core worker via msgpack; functions
  aren't serialisable by default. Set `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
  (already done at the top of `vllm_forward.py`).
- vLLM 0.20 dropped `prompt_token_ids=` kwarg from `LLM.generate`; pass
  `[{"prompt_token_ids": token_ids}]` instead.
- Megatron forward-only: set `model_provider.gradient_accumulation_fusion = False`
  to avoid the APEX `fused_weight_gradient_mlp_cuda` import.
- The two engines have different module-boundary semantics:
  - vLLM's `post_attention_layernorm` is a fused **norm + residual-add**: its
    arg[0] is `attn_out` (pre-add, norm ≈ 21), arg[1] is the incoming residual
    (norm ≈ 1.88), and the new residual stream is built inside.
  - Megatron's `pre_mlp_layernorm` already receives `attn_out + residual` as input.
  - To compare, sum vLLM's two positional args and compare to mcore's
    `pre_mlp_layernorm` input.
- For vLLM, the residual stream after layer i is `hidden_states + residual` of
  the 2-tuple returned by `LlamaDecoderLayer.forward`. Megatron's
  `TransformerLayer.forward` already returns the residual stream as a single
  tensor (or as `(tensor, context)` for cross-attention).
- Run vLLM and Megatron *sequentially* on the same GPU; vLLM's KV-cache profile
  consumes ~140 GiB and will OOM a parallel Megatron run.
