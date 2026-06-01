# MXFP8 batch-invariant matmul: design reference

Two parts:

- **Part 1** — side-by-side reference for the existing **BF16** batch-invariant
  (BI) `matmul_kernel_persistent` in vLLM and Megatron-Core. (The BF16 case is
  what `my_docs/llama3_8b_numeric_mismatch.md` "Module 3" already proved
  bit-identical given identical inputs; this section is the kernel-level
  walkthrough that documents why.)
- **Part 2** — concrete design for an MXFP8 BI matmul kernel and the wiring
  needed to install it on both engines (vLLM dispatcher and Megatron's
  `general_gemm` patch).

Cited file paths are absolute relative to the repo root
`/lustre/fsw/coreai_dlalgo_llm/users/guyueh/rl_projects/mxfp8/RL/`.

---

# Part 1 — BF16 batch-invariant matmul kernels

## 1.1 Source-of-truth file paths

- vLLM:
  `3rdparty/vllm/vllm/model_executor/layers/batch_invariant.py`
  - `_compute_pid` — line 41–48
  - `matmul_kernel_persistent` — line 51–139 (Triton `@triton.jit`)
  - `matmul_persistent` — line 142–216 (host wrapper, dtype-config dict)
  - `bmm_kernel` / `bmm_batch_invariant` — line 219–349 / 653–738
  - `mm_batch_invariant` — line 599–600
  - `addmm_batch_invariant` — line 741–742
  - `matmul_batch_invariant` — line 603–650
  - `linear_batch_invariant` — line 903–908
  - `enable_batch_invariant_mode` — line 921–988 (aten op patching)
- Megatron-Core (vendored under Megatron-Bridge):
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/custom_layers/batch_invariant_kernels.py`
  - `_compute_pid` — line 62–69
  - `matmul_kernel_persistent` — line 72–151
  - `matmul_persistent` — line 178–247
  - `mm_batch_invariant` — line 481–483
  - `addmm_batch_invariant` — line 486–488
  - `BatchInvariantTEGemmFn` — line 713–827 (autograd Fn calling `matmul_persistent`)
  - `_te_general_gemm_patched` — line 830–874 (TE `general_gemm` hook)
  - `_te_patch_for_batch_invariant` — line 537–618 (installs the hook)
  - `enable_batch_invariant_mode` — line 965–978

Both files derive from
[`thinking-machines-lab/batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops)
(noted in the Megatron header at line 1–3).

## 1.2 The `matmul_kernel_persistent` Triton kernel

Both files define a kernel with this **exact signature** (modulo whitespace):

```python
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr, B_LARGE: tl.constexpr, C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
```

### K-loop structure (identical on both sides)

The body is identical except for one cosmetic difference (see 1.4):

1. **Persistent grid**: `start_pid = tl.program_id(axis=0)`, with
   `min(NUM_SMS, ceil(M/BM) * ceil(N/BN))` programs launched
   (vLLM `batch_invariant.py:159–166`, Megatron `batch_invariant_kernels.py:195–198`).
2. **Outer tile loop** is `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`
   — each persistent program walks tiles in a fixed, batch-shape-independent
   stride. (vLLM:87, Megatron:109)
3. **Tile ordering** is the GROUP_SIZE_M Z-curve via `_compute_pid` — pid_m and
   pid_n derived from `tile_id`, `num_pid_in_group=GROUP_SIZE_M * num_pid_n`,
   and `GROUP_SIZE_M`. (vLLM:88, Megatron:110)
4. **Inner K loop** is a plain Python `for ki in range(k_tiles):` — emits a
   fixed `tl.dot(a, b, accumulator)` per K-tile in a fixed order; no swizzling
   or split-K. (vLLM:103–121, Megatron:125–135)
5. **Accumulator** is `tl.zeros((BM, BN), dtype=tl.float32)`. Single
   `tl.dot(a, b, accumulator)` per K-tile accumulates into the same fp32
   register tile. Pipeline depth is `num_stages` for the K-loop loads.
6. **Output cast**: `c = accumulator.to(c_ptr.dtype.element_ty)` — single cast
   to BF16/FP16/FP32 at store. (vLLM:138, Megatron:150)
7. **Bias** (when `HAS_BIAS`): `bias = tl.load(...).to(tl.float32);
   accumulator += bias` — bias added **to the fp32 accumulator before the cast**.
   (vLLM:134–137, Megatron:146–149)

### Launch-config dict (per dtype, host side)

For bfloat16 (our path), both files specify **identical** configs (vLLM:169–176,
Megatron:201–208):

```python
torch.bfloat16: {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_stages": 3,
    "num_warps": 8,
},
```

For fp16 there is a small mismatch — see 1.4.

For fp32 both are also identical: `BM=BN=128`, `BK=32`, `num_stages=3`, `num_warps=8`.

### Persistent-grid arithmetic

```text
NUM_TILES   = ceil(M/BM) * ceil(N/BN)
GRID        = min(NUM_SMS, NUM_TILES)
EACH PID    walks tile ids [start_pid, start_pid + NUM_SMS, ...]
TILE ORDER  determined entirely by (NUM_SMS, M, N, GROUP_SIZE_M)
            — does *not* depend on K
            — does *not* depend on batch size (M is the flattened
              batch×seq dim, but the per-output-tile K reduction is
              always the same fixed sequential dot-product, so output[m,n]
              for any (m,n) is computed identically regardless of how
              many other rows are in the batch).
```

That fixed K-order is the source of batch invariance for BF16.

## 1.3 How each framework dispatches

### vLLM (`3rdparty/vllm/vllm/model_executor/layers/batch_invariant.py:921–988`)

When `VLLM_BATCH_INVARIANT=1` and SM ≥ 80 (Ampere) or SM ≥ 100 (Blackwell), the
following ops are registered through `torch.library.Library("aten", "IMPL")`:

| aten op | impl | line |
|---|---|---|
| `aten::mm` | `mm_batch_invariant` | 938 |
| `aten::addmm` | `addmm_batch_invariant` | 939 |
| `aten::matmul` | `matmul_batch_invariant` | 940 |
| `aten::linear` | `linear_batch_invariant` | 941 |
| `aten::_log_softmax` | `_log_softmax_batch_invariant` | 957 |
| `aten::softmax` / `aten::_softmax` | `softmax_batch_invariant` | 959–960 |
| `aten::mean.dim` | `mean_batch_invariant` | 961 |
| `aten::bmm` | `bmm_batch_invariant` | 967 (also `torch.bmm` monkey-patched at 970) |

Plus side effects: TF32 is disabled, BF16/FP16 reduced-precision reductions are
turned off, and `preferred_blas_library("cublaslt")` is set so the fallback
path is deterministic.

`mm_batch_invariant(a, b) -> matmul_persistent(a, b)` (line 599–600).
`addmm_batch_invariant(bias, a, b) -> matmul_persistent(a, b, bias=bias)`
(line 741–742) — bias is passed into the kernel and added to the **fp32
accumulator**.

`linear_batch_invariant(input, weight, bias)` re-implements `F.linear` as
`matmul_batch_invariant(input, weight.t())` followed by an external bias add
(line 903–908) — the **bias is added outside the kernel, on bf16 output**, so
in vLLM `linear` and `addmm` have *different* bias rounding precision. Neither
is used for our Llama-3.1-8B path (all linears are bias=False).

`matmul_batch_invariant` (line 603–650) flattens ND×2D and 2D×ND linear-layer
matmuls to use the 2D persistent kernel; ND×ND batched matmul (e.g. attention's
QK^T) uses `bmm_batch_invariant` → `bmm_kernel` (a separate persistent kernel
for 3D inputs, line 219–349).

### Megatron-Core (`3rdparty/.../batch_invariant_kernels.py:965–978`)

When `enable_batch_invariant_mode()` is called, only **four** aten ops are
patched:

| aten op | impl | line |
|---|---|---|
| `aten::mm` | `mm_batch_invariant` | 973 |
| `aten::addmm` | `addmm_batch_invariant` | 974 |
| `aten::_log_softmax` | `_log_softmax_batch_invariant` | 975 |
| `aten::mean.dim` | `mean_batch_invariant` | 976 |

**No** `aten::matmul`, `aten::linear`, `aten::softmax`, `aten::_softmax`, or
`aten::bmm` patches. Megatron compensates by reaching deeper into TE — also at
line 977 it calls `_te_patch_for_batch_invariant()` (line 537–618), which
monkey-patches:

- `transformer_engine.pytorch.cpp_extensions.general_gemm`
- `transformer_engine.pytorch.module.linear.general_gemm`
- `transformer_engine.pytorch.module.layernorm_linear.general_gemm`
- `megatron.core.extensions.transformer_engine.general_gemm`
- `transformer_engine.pytorch.RMSNorm.forward`
- `transformer_engine.pytorch.module.layernorm.{rmsnorm, rmsnorm_forward, rmsnorm_fwd}`

All four `general_gemm` symbols redirect to `_te_general_gemm_patched`
(line 830–874), which extracts `A, B, out_dtype, layout, out, bias, grad` from
TE's flexible signature and then calls `BatchInvariantTEGemmFn.apply(A, B,
bias if not grad else None, out_dtype, layout)`.

`BatchInvariantTEGemmFn.forward` (line 713–772):

```python
opA = A.T.contiguous() if layout[0]=="T" else A.contiguous()  # [K, O]
opB = B.T.contiguous() if layout[1]=="T" else B.contiguous()  # [..., K]
opB_2d = opB.reshape(-1, K)                                    # flatten lead dims
base_2d = matmul_persistent(opB_2d, opA, bias=None)            # GEMM
out = base_2d.reshape(*leading_shape, base_2d.shape[-1])
if bias is not None:
    out = out + bias                                            # bias added *after*
if out_dtype is not None:
    out = out.to(out_dtype)
```

So **for Llama linear layers**, both engines end up calling the **same** kernel
(`matmul_persistent`) with the same configs, just reached through different
patching layers.

## 1.4 Side-by-side divergence table (BF16 path)

Every difference, including cosmetic ones, found by reading both files
line by line:

| # | Aspect | vLLM | Megatron | Effect |
|---|---|---|---|---|
| 1 | `_compute_pid` signature | takes `(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)` (vLLM:42) | takes `(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)` (Megatron:63) | `NUM_SMS` is unused in Megatron's `_compute_pid` body. **Cosmetic only**; same function body. |
| 2 | NUM_SMS query | `num_compute_units(a.device.index)` (vLLM:151) — vLLM utility | `get_compute_units()` (Megatron:154–175) — match/case on accelerator type (CUDA/XPU/CPU) | Different code paths; both return `multi_processor_count` on CUDA. |
| 3 | `BLOCK_SIZE_N` for **fp16** | `_fp16_block_size_n` — 128 or 256 chosen by `get_max_shared_memory_bytes()` at line 945 (256 if >106496 bytes shared mem) | hard-coded 256 (Megatron:211) | On hardware with smaller SMEM (≤104 KB, i.e. SM75/SM80) Megatron's fp16 BM=128,BN=256 config can exceed shared memory and fail to compile. **vLLM is safer.** Does not affect bf16. |
| 4 | bf16 / fp32 configs | BM=128, BN=128, BK=64 (bf16) / BK=32 (fp32), GROUP_SIZE_M=8, num_stages=3, num_warps=8 | **Identical** | Same kernel parameters. |
| 5 | `aten::matmul` patch | ✓ (`matmul_batch_invariant`, line 940) | ✗ (not patched) | vLLM intercepts `torch.matmul` calls (e.g. inside the model code, attention's `qk^T`, ...). Megatron relies on TE owning all GEMMs in the model, so no matmul aten patch is needed. |
| 6 | `aten::linear` patch | ✓ (line 941) | ✗ | Same reason — Megatron's linears go through TE. |
| 7 | `aten::softmax` / `aten::_softmax` patch | ✓ (line 959–960) | ✗ | vLLM patches softmax to a deterministic 2-pass implementation; Megatron relies on TE's softmax. |
| 8 | `aten::bmm` patch | ✓ (line 967, also monkey-patches `torch.bmm`) plus `bmm_kernel` Triton kernel (line 219–349) | ✗ (no `bmm_kernel` defined) | Affects attention's QK^T and PV (when not done by FlashAttn). |
| 9 | Bias-handling for `addmm` | `matmul_persistent(a, b, bias=bias)` — bias loaded inside kernel, cast to fp32, accumulated, **single bf16 cast at store** | `out = matmul_persistent(opB_2d, opA, bias=None); out = out + bias` — bias added on bf16 output, **two bf16 cast events** | ~1 bf16 ULP per element for biased linears (N/A for Llama-3.1-8B; both have bias=False). Already documented in `llama3_8b_numeric_mismatch.md:382–391`. |
| 10 | Bias-handling for `linear` | bias added on bf16 output (vLLM:903–908) — same as Megatron's TE path | (see #9) | Inside vLLM, `addmm` and `linear` have different bias rounding. Cosmetic for our path. |
| 11 | TF32 / reduced-precision toggling | Explicitly disables `bf16_reduced_precision_reduction`, sets `preferred_blas_library("cublaslt")`, and `init_batch_invariance()` sets `fp32_precision="ieee"` for matmul/cudnn (`batch_invariant.py:972–988, 1018–1021`) | Does not touch these PyTorch settings | Megatron's BI mode assumes the caller has already configured determinism env vars. In our setup, both stacks end up with the same settings, but Megatron's setup is more implicit. |
| 12 | Hopper (SM90) fallback | Disables Triton matmul on SM90 (falls back to cuBLAS workspace tricks at `batch_invariant.py:946–954`) | No SM-family guard — always installs the Triton matmul on CUDA | On Hopper, vLLM uses a different code path entirely (cuBLAS env-var deterministic mode). Megatron unconditionally uses Triton. For Blackwell (SM100, our target) both paths use Triton — no effect. |
| 13 | TE patching | None (vLLM has no TE dependency) | Patches TE `general_gemm`, `RMSNorm`, etc. — see 1.3 | Necessary because all of Megatron's linears go through TE. |
| 14 | Autograd | `mm_batch_invariant` / `addmm_batch_invariant` are plain functions; aten dispatcher handles autograd via PyTorch's built-in mm backward | `BatchInvariantTEGemmFn` defines a custom backward (line 774–827) that uses `grad_out.matmul(opA.T)` and `opB.T.matmul(grad_out)` — also recursively hits the BI kernel through `aten::mm` | Megatron supports training (need BI backward); vLLM is inference-only. |
| 15 | Whitespace / formatting | Slightly different line wrapping (vLLM splits arg lists more aggressively) | — | Cosmetic. |

### What the diff *means* for our BF16 parity claim

Items 1, 11, 12, 13, 14, 15 are not numerical. Items 5–8 affect ops we don't
use on the Llama path (Megatron's attention is TE's `DotProductAttention`, not
torch's `aten::matmul`/`aten::bmm`). Item 3 doesn't affect bf16. Items 9–10
matter only for biased linears (we have none).

So the BF16 BI kernel is **functionally identical** between the two
frameworks for the Llama-3.1-8B forward pass: same Triton kernel, same configs,
same fixed K-order. This matches the empirical bit-identical
`linear_qkv.linear` and `qkv_proj` outputs documented in
`my_docs/llama3_8b_numeric_mismatch.md:341–361, 670`.

## 1.5 The single non-cosmetic kernel-body difference (when both kernels run)

Inside the kernel body the two implementations are **bit-for-bit identical**.
The only "difference" in the kernel body is that Megatron's `_compute_pid`
takes an unused `NUM_SMS` parameter (item 1 in the table). Triton compiles
the call site identically because `NUM_SMS` is a `tl.constexpr` and dead-code
elimination removes it.

Conclusion: under identical (a, b, dtype, strides, device) **the two
`matmul_kernel_persistent` invocations produce bit-identical output**.

---

# Part 2 — MXFP8 batch-invariant matmul design

## 2.1 Existing MXFP8 GEMM paths (non-BI)

### vLLM

vLLM's MXFP8 W8A8 linear lives in
`3rdparty/vllm/vllm/model_executor/layers/quantization/modelopt.py:1571–1673`
(`ModelOptMxFp8LinearMethod`). On Blackwell (SM100+) the chosen kernel is
`FlashInferCutlassMxfp8LinearKernel`
(`3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py`):

- `apply_weights` (line 46–103):
  1. Pad M up to a multiple of 128 (FlashInfer's minimum tile).
  2. **Activations are quantised on the fly** via
     `mxfp8_e4m3_quantize(input_2d, is_sf_swizzled_layout=True)`
     (`mxfp8_utils.py:105–108` → `_mxfp8_e4m3_quantize_impl` → on SM100
     `flashinfer.mxfp8_quantize`). The result is an fp8 E4M3 tensor + a 1D
     swizzled E8M0 scale tensor.
  3. Call `flashinfer.mm_mxfp8(input_mxfp8, weight.t(), input_scale,
     weight_scale, out_dtype=out_dtype, backend="cutlass")`
     (`flashinfer.py:87–94`, wrapped at
     `3rdparty/vllm/vllm/utils/flashinfer.py:618–653`).
  4. Strip padding; add bias on bf16 output.

The FlashInfer `mm_mxfp8(..., backend="cutlass")` lands in a **CUTLASS** kernel
implemented in flashinfer's CUDA code (not in this repo). The kernel uses
Hopper/Blackwell mma instructions with block-scaled fp8 (`tcgen05.mma.kind::mxf8f6f4`
on Blackwell) and **MOST IMPORTANTLY**:

- The kernel selects a tile schedule and split-K factor based on M, N, K.
- The scale tensor must be **swizzled** in CUTLASS's E8M0 "F8_128x4" layout
  (vLLM does this at `mxfp8_utils.py:14–35`).

There are two fallbacks (`kernels/linear/__init__.py:245–258`):
`MarlinMxfp8LinearKernel` (Marlin W8A16 — dequant-on-load to bf16 weight tile),
and `EmulationMxfp8LinearKernel` (dequant whole weight to bf16, call
`torch.nn.functional.linear`). The emulation path is interesting for us — it
shows that **MXFP8 can be made BI by simple dequant + BF16 BI matmul**, at the
cost of throughput.

`init_mxfp8_linear_kernel()` (`kernels/linear/__init__.py:534–566`) picks the
first available kernel from a priority list per platform; on CUDA SM100 it
returns `FlashInferCutlassMxfp8LinearKernel`.

**Batch invariance check**: the FlashInfer CUTLASS MXFP8 kernel is **not**
batch-invariant in general — it picks tile schedule by M, and on Blackwell
uses persistent kernels with stream-K (the same mechanism that broke FA4 BI in
vLLM 0.20.x, see `llama3_8b_numeric_mismatch.md:744–753`). Confirmed by
`my_script/vllm_run_mxfp8_bi_L0.log` only diverging *after* the first MXFP8
GEMM call. We need a deterministic replacement.

### Megatron / TE

Megatron under `MXFP8BlockScaling` recipe
(`3rdparty/.../megatron/core/extensions/transformer_engine.py:245–246`)
delegates all linears to TE. TE's path under the autocast is:

1. `te.pytorch.module.Linear` / `LayerNormLinear` quantises inputs to mxfp8 in
   `_TE_pre_forward` (using the recipe's `Quantizer`).
2. Calls `transformer_engine.pytorch.cpp_extensions.general_gemm(A, B,
   out_dtype=..., layout=..., bias=..., ...)`.
3. `general_gemm` is a thin Python wrapper that calls into TE's C++ ext, which
   dispatches to **cuBLASLt's** block-scaled fp8 GEMM
   (`cublasLtMatmul` with `CUBLASLT_MATMUL_DESC_BLOCK_SCALE` descriptors —
   added in cuBLAS 12.8+ for Blackwell mxfp8). Older TE versions can fall back
   to a CUDA kernel from `transformer_engine_torch`.

cuBLASLt's GEMM with block-scaled fp8 uses **algorithm heuristics** to pick a
tile shape from M/N/K. **It is not batch-invariant** for two reasons:

- Tile shape selection depends on M.
- For tall-skinny M, cuBLASLt may select split-K with non-deterministic
  reduction order.

For BF16, Megatron's BI mode side-steps this by replacing `general_gemm` with
`_te_general_gemm_patched` → `matmul_persistent`. For MXFP8, the same hook
point (`_te_general_gemm_patched`) is the place to inject the new MXFP8 BI
kernel. But the inputs landing at `_te_general_gemm_patched` under MXFP8 are
**TE Float8Tensor / MXFP8Tensor objects**, not raw torch tensors. The patched
function would need to extract `._data` (fp8 e4m3fn) and `._scale_inv` (uint8
E8M0) plus the block size from the quantizer state.

`_is_supported_dtype_for_bik(t)` in `batch_invariant_kernels.py:709–710`
currently rejects anything that isn't `{fp16, bf16, fp32}` — so under MXFP8
recipe the **current Megatron BI patch raises**
`"Unsupported dtype for batch-invariant GEMM"`. That guard must be loosened
once the new kernel exists.

## 2.2 Goals & invariants

Given the format described in `my_script/convert_hf_bf16_ckpt_to_mxfp8.py`
(lines 82–140) and `3rdparty/vllm/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`:

- **Block layout**: 1×32 along K. For a `[N, K]` weight (or `[M, K]`
  activation), the scale tensor has shape `[N, K/32]` (resp. `[M, K/32]`).
  Data dtype is `torch.float8_e4m3fn`; scale dtype is `torch.uint8` holding
  an **E8M0 biased exponent** (`scale_unbiased = scale_uint8 - 127`).
- **Dequantisation**: `x_bf16[m, k] = x_fp8[m, k].to(fp32) * 2^(scale_uint8[m, k//32] - 127)`.
- **Saturation**: the converter clamps the exponent to `[0, 254]` (one
  reserved for "subnormal/zero"; one for NaN/inf).
- **Output semantics**:
  ```
  out_bf16[m, n] = bf16( sum_k ( dequant(a_fp8[m, k], a_scale[m, k//32]) *
                                  dequant(b_fp8[k, n], b_scale[k//32, n]) ) )
  ```
  where the sum runs in fp32 in a **fixed K-order**, then casts to bf16 at
  store. Bias is added to the fp32 accumulator before the cast (matching the
  BF16 BI kernel's choice).

Variants we must handle:

| Inputs | A (activation) | B (weight) | Use case |
|---|---|---|---|
| (a) **W8A16** | bf16 | mxfp8 (fp8 + E8M0 scale) | Inference under `VLLM_BATCH_INVARIANT=1` with offline MXFP8 weights, BF16 activations. Mirrors vLLM's MarlinMxfp8 path. |
| (b) **W8A8 (MXFP8)** | mxfp8 (fp8 + E8M0 scale, computed on-the-fly per call) | mxfp8 | Standard ModelOpt MXFP8 path; the FlashInferCutlassMxfp8 path today. |
| (c) **W8A8 (training)** | mxfp8 (from TE Quantizer in fp8 autocast) | mxfp8 | Megatron MXFP8 training path. |

We need a single kernel that handles all three by reading metadata; the host
wrapper picks the right scale-loading branch.

## 2.3 Triton kernel design

Source file we'd add (both engines):
`batch_invariant.py` / `batch_invariant_kernels.py` — add
`mxfp8_matmul_kernel_persistent` next to `matmul_kernel_persistent`.

### 2.3.1 Tile-shape choice

The BF16 BI kernel uses `BM=128, BN=128, BK=64`. For MXFP8 our K-tile must be a
multiple of 32 (the scale block size). Choices:

- `BK = 32` → 1 scale lookup per K-tile per (m, n) row. Smallest possible.
  Highest scale-load overhead, smallest K-pipeline depth.
- `BK = 64` → 2 scales along K per row per K-tile. Matches BF16 BI's BK.
- `BK = 128` → 4 scales along K per row per K-tile. Largest pipeline; fewer
  total K-iterations.

Recommended: **`BK = 64`**, identical to the BF16 BI kernel, with **2 scales
loaded per K-tile**. This keeps the K-loop iteration count, num_stages, and
shared-memory budget identical between BF16 and MXFP8 paths, which is good for
batch-invariance verification (we can do a side-by-side compile and check tile
order is the same).

Other shapes inherited from BF16 BI:
`BM = 128`, `BN = 128`, `GROUP_SIZE_M = 8`, `num_stages = 3`, `num_warps = 8`.

### 2.3.2 Data + scale loads

For each K-iteration `ki` (covering K indices `[ki*BK, (ki+1)*BK)`):

```text
A tile (fp8 if MXFP8 activations, else bf16):  shape [BM, BK]
B tile (fp8):                                   shape [BK, BN]

A scales: shape [BM, BK/32]  (uint8, E8M0)
B scales: shape [BK/32, BN]  (uint8, E8M0)
```

With `BK=64`, that's 2 scale columns per K-tile.

Per-element multiply expands to: for each (i, j) in (BM, BN), and each (k_block_in_tile,
k_in_block) in (BK/32, 32):

```text
a_lp = a_fp8[i, k_block_in_tile*32 + k_in_block]
b_lp = b_fp8[k_block_in_tile*32 + k_in_block, j]
a_s  = 2^(a_scale[i, k_block_in_tile] - 127)
b_s  = 2^(b_scale[k_block_in_tile, j] - 127)
accumulator[i, j] += (a_lp.to(fp32) * a_s) * (b_lp.to(fp32) * b_s)
```

But we don't need to evaluate `2^x` per element — we can factor:

```text
accumulator[i, j] += (a_s * b_s) *
                     sum_{kb=0..1} sum_{kk=0..31} (a_lp.to(fp32) * b_lp.to(fp32))
```

…**only if `a_s` and `b_s` are constant over the 32-element K block**. They are
by construction: 1 scale per 32 K-elements. So within each 32-element K block
we can do a pure fp8×fp8 dot, then multiply by the product of scales.

### 2.3.3 Per-K-block fp8 dot + scale

The natural Triton primitive here is `tl.dot(a_fp8, b_fp8,
acc_fp32)` where Triton accepts fp8e4m3 / fp8e5m2 operands and an fp32
accumulator (Triton ≥ 2.2). Triton compiles this to Hopper/Blackwell
`mma.sync.aligned.kind::e4m3` instructions which produce fp32 partial
results. The partial-products are then scaled by the block scale.

Two implementation styles:

**Style A — Inline dequant per K-tile (no `tl.dot` on fp8):**

```python
# Per K-tile (BK=64):
a_fp8 = tl.load(a_ptrs)                          # [BM, 64]
b_fp8 = tl.load(b_ptrs)                          # [64, BN]
a_f32 = a_fp8.to(tl.float32)                     # [BM, 64]
b_f32 = b_fp8.to(tl.float32)                     # [64, BN]

# Load scales for this K-tile (2 K-blocks of 32 per BK=64):
a_sc_u8 = tl.load(a_scale_ptrs)                  # [BM, 2]
b_sc_u8 = tl.load(b_scale_ptrs)                  # [2, BN]
# Convert E8M0 -> fp32 multiplier (2^(uint - 127)):
a_mult = exp2_e8m0(a_sc_u8)                      # [BM, 2]
b_mult = exp2_e8m0(b_sc_u8)                      # [2, BN]

# Broadcast scales across the 32 elements within each K block.
# a_f32 has shape [BM, 64]; we tile a_mult [BM, 2] -> [BM, 64]
# by repeating each scalar 32 times along K.
a_scaled = a_f32 * a_mult.repeat_interleave(32, dim=1)
b_scaled = b_f32 * b_mult.repeat_interleave(32, dim=0)

accumulator = tl.dot(a_scaled, b_scaled, accumulator)
```

**Cons**: doubles K-tile shared-memory because we materialise fp32 tiles
(`BM*BK*4 + BK*BN*4 = 128*64*4 + 64*128*4 = 64 KB`, fine on Blackwell's 228 KB
per-SM SMEM). `tl.dot` runs in fp32×fp32 → fp32 — no Tensor Core fp8 mma.
About 4× slower than Style B but maximally portable and unambiguous.

**Style B — Per-K-block dot, scale the partial accumulator:**

```python
# Outer K-tile loop:
for ki in range(k_tiles):                        # k_tiles = K // 64

    # Load fp8 data: 2 K-blocks of 32 each
    a_fp8 = tl.load(a_ptrs)                      # [BM, 64], fp8e4m3
    b_fp8 = tl.load(b_ptrs)                      # [64, BN], fp8e4m3

    # Load 2 scales each per row/col
    a_sc_u8 = tl.load(a_scale_ptrs)              # [BM, 2]
    b_sc_u8 = tl.load(b_scale_ptrs)              # [2, BN]
    a_mult = exp2_e8m0(a_sc_u8)                  # [BM, 2]
    b_mult = exp2_e8m0(b_sc_u8)                  # [2, BN]

    # Sub-block 0 (K indices 0..31 within this tile)
    a0 = a_fp8[:, 0:32]                          # [BM, 32], fp8
    b0 = b_fp8[0:32, :]                          # [32, BN], fp8
    partial0 = tl.dot(a0, b0)                    # [BM, BN], fp32
    accumulator += partial0 * a_mult[:, 0:1] * b_mult[0:1, :]

    # Sub-block 1 (K indices 32..63 within this tile)
    a1 = a_fp8[:, 32:64]
    b1 = b_fp8[32:64, :]
    partial1 = tl.dot(a1, b1)
    accumulator += partial1 * a_mult[:, 1:2] * b_mult[1:2, :]
```

**Pros**: uses Tensor Core fp8 mma (`tl.dot(fp8, fp8, fp32)`), ~10× the
throughput of Style A on Blackwell. Each 32-element fp8 dot is one mma
instruction tile.

**Cons**: introduces a fp32 multiply-and-add between sub-blocks; this is the
exact pattern CUTLASS BlockScaledMma uses, and produces results within ≤1 fp32
ULP of the "scale-first" reference. **The key question is whether the
sub-block ordering matters for batch invariance** — it doesn't, because
sub-block (kb=0, kb=1) ordering inside the K-tile loop is also fixed and
independent of M.

**Recommendation: Style B.** It's faster, matches the reference math up to fp32
ULPs, and the K-order is provably fixed at:

```text
For each (m, n) tile:
  For each K-tile ki ∈ [0, K/BK):     # BK = 64
    For each K-sub-block kb ∈ [0, BK/32):   # 2 sub-blocks per BK=64
      accumulator += (tl.dot(a[:, kb*32:(kb+1)*32], b[kb*32:(kb+1)*32, :]))
                     * a_scale[m, ki*2 + kb] * b_scale[ki*2 + kb, n]
```

This is identical to BF16 BI's `for ki in range(k_tiles): accumulator =
tl.dot(a, b, accumulator)`, just with a per-K-block scale factor multiplied in.
**Batch invariance is preserved because (a) the loop bounds depend only on K,
not M, and (b) `tl.dot` over fixed (BM, 32, BN) tiles has deterministic
reduction order under Triton.**

### 2.3.4 E8M0 → fp32 multiplier

The E8M0 uint8 encodes a biased exponent: `multiplier = 2^(uint - 127)`. Two
fast Triton expressions:

```python
# Option 1: explicit exp2
mult = tl.exp2(scale_u8.to(tl.float32) - 127.0)

# Option 2: bit-pack into fp32 (no exp2 needed; pure bit shift)
# fp32 layout: sign(1) | exponent(8) | mantissa(23)
# scale_u8 is the biased exponent itself, mantissa=0 ⇒ value=2^(scale - 127)
mult_bits = scale_u8.to(tl.uint32) << 23
mult = mult_bits.to(tl.float32, bitcast=True)
```

**Option 2 is faster and lossless** (the encoding *is* 2^(x-127) by
construction), and it's how cuBLAS's block-scaled GEMM does it internally.
Triton supports `bitcast=True` since 2.1. We use Option 2.

Edge case: `scale_u8 == 0` is the "subnormal/zero" encoding — the converter
in `convert_hf_bf16_ckpt_to_mxfp8.py:115–120` maps it to `multiplier=1.0`.
Option 2 produces `2^-127`, which is fp32 denormal — almost certainly wrong
for the math. We must guard:

```python
mult_bits = scale_u8.to(tl.uint32) << 23
mult = tl.where(scale_u8 == 0, 1.0, mult_bits.to(tl.float32, bitcast=True))
```

(The same guard is in the converter.)

Note: `scale_u8 == 255` (E8M0 NaN) we don't need to handle — the converter
clamps to ≤254.

### 2.3.5 Bias and output cast

Identical to BF16 BI kernel:

```python
if HAS_BIAS:
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
    accumulator += bias
c = accumulator.to(c_ptr.dtype.element_ty)
tl.store(c_ptrs, c, mask=c_mask)
```

This matches what `BatchInvariantTEGemmFn` does for the BF16 path (bias added
outside the kernel, after the cast) **only when bias is None**, but matches the
vLLM `addmm` path (bias in fp32 acc). Since our Llama path has no bias, this
isn't load-bearing for parity; but for correctness with biased models we
should mirror **vLLM's `addmm` choice** (bias-in-accumulator, single bf16
cast), which is the lower-rounding-error path.

### 2.3.6 Full kernel pseudocode

```python
@triton.jit(launch_metadata=_matmul_launch_metadata)
def mxfp8_matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_k, stride_b_scale_n,
    BLOCK_SIZE_M: tl.constexpr,            # 128
    BLOCK_SIZE_N: tl.constexpr,            # 128
    BLOCK_SIZE_K: tl.constexpr,            # 64
    SCALE_BLOCK: tl.constexpr,             # 32 (E8M0 block size)
    GROUP_SIZE_M: tl.constexpr,            # 8
    NUM_SMS: tl.constexpr,
    A_IS_FP8: tl.constexpr,                # if False, A is bf16 (W8A16 mode)
    HAS_BIAS: tl.constexpr,
    A_LARGE: tl.constexpr, B_LARGE: tl.constexpr, C_LARGE: tl.constexpr,
):
    SCALES_PER_TILE: tl.constexpr = BLOCK_SIZE_K // SCALE_BLOCK   # 2

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles  = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            # ---- Data loads ----
            a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)

            # ---- Scale loads (2 scales per row/col per K-tile) ----
            offs_sk = ki * SCALES_PER_TILE + tl.arange(0, SCALES_PER_TILE)
            b_sc_ptrs = b_scale_ptr + offs_sk[:, None] * stride_b_scale_k + offs_bn[None, :] * stride_b_scale_n
            b_sc_u8 = tl.load(b_sc_ptrs)                                         # [2, BN], uint8
            b_mult_bits = b_sc_u8.to(tl.uint32) << 23
            b_mult = tl.where(b_sc_u8 == 0, 1.0, b_mult_bits.to(tl.float32, bitcast=True))   # [2, BN]

            if A_IS_FP8:
                a_sc_ptrs = a_scale_ptr + offs_am[:, None] * stride_a_scale_m + offs_sk[None, :] * stride_a_scale_k
                a_sc_u8 = tl.load(a_sc_ptrs)                                     # [BM, 2]
                a_mult_bits = a_sc_u8.to(tl.uint32) << 23
                a_mult = tl.where(a_sc_u8 == 0, 1.0, a_mult_bits.to(tl.float32, bitcast=True))    # [BM, 2]
            else:
                a_mult = tl.full((BLOCK_SIZE_M, SCALES_PER_TILE), 1.0, dtype=tl.float32)

            # ---- Per-K-sub-block fp8 mma + scale ----
            for kb in range(SCALES_PER_TILE):
                k_lo = kb * SCALE_BLOCK
                k_hi = (kb + 1) * SCALE_BLOCK
                a_sub = a[:, k_lo:k_hi]              # [BM, 32]
                b_sub = b[k_lo:k_hi, :]              # [32, BN]
                if A_IS_FP8:
                    partial = tl.dot(a_sub, b_sub)   # fp8 mma -> fp32 [BM, BN]
                else:
                    # W8A16 path: a is bf16, b is fp8 — convert b to bf16 first
                    b_bf16 = b_sub.to(tl.bfloat16)   # bf16 mma path
                    partial = tl.dot(a_sub, b_bf16)
                # broadcast scales
                scale_outer = a_mult[:, kb:kb+1] * b_mult[kb:kb+1, :]            # [BM, BN]
                accumulator += partial * scale_outer

        # ---- Bias + cast + store ----
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)
```

### 2.3.7 Host wrapper

```python
def mxfp8_matmul_persistent(
    a: torch.Tensor,                # bf16 [M, K] OR fp8 [M, K]
    b: torch.Tensor,                # fp8 [K, N]   (already pre-permuted)
    b_scale: torch.Tensor,          # uint8 [K/32, N] E8M0
    a_scale: torch.Tensor | None,   # uint8 [M, K/32] E8M0 (only if A is fp8)
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    assert a.shape[1] == b.shape[0], "K must match"
    M, K = a.shape
    _, N = b.shape
    assert K % 32 == 0, "K must be divisible by 32 (MXFP8 block size)"
    assert b.dtype == torch.float8_e4m3fn
    assert b_scale.dtype == torch.uint8
    a_is_fp8 = a.dtype == torch.float8_e4m3fn
    if a_is_fp8:
        assert a_scale is not None and a_scale.dtype == torch.uint8
    NUM_SMS = num_compute_units(a.device.index)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    grid = lambda META: (min(NUM_SMS,
                              triton.cdiv(M, META["BLOCK_SIZE_M"]) *
                              triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    mxfp8_matmul_kernel_persistent[grid](
        a, b, c, bias,
        a_scale if a_is_fp8 else b_scale,   # placeholder if not used
        b_scale,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        (a_scale.stride(0) if a_is_fp8 else 0),
        (a_scale.stride(1) if a_is_fp8 else 0),
        b_scale.stride(0), b_scale.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, SCALE_BLOCK=32,
        GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
        A_IS_FP8=a_is_fp8, HAS_BIAS=bias is not None,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        num_stages=3, num_warps=8,
    )
    return c
```

The signature is intentionally **a strict superset** of `matmul_persistent`:
all the BF16 fields appear in the same order; scale pointers and `SCALE_BLOCK`
are appended. The dispatcher can therefore reuse the same plumbing.

### 2.3.8 Note on `tl.dot` fp8 support

`tl.dot` supports `(fp8e4m3, fp8e4m3, fp32)` as of Triton 2.2 (the version both
vLLM and Megatron ship). On Blackwell SM100, Triton compiles this to
`tcgen05.mma.kind::e4m3` — the same instruction the FlashInfer CUTLASS MXFP8
kernel uses for the inner mma. So the per-32-K-block fp8 mma is on Tensor
Cores; only the per-block scale multiply happens in CUDA cores.

Triton also has an experimental `tl.dot(..., scale_a=, scale_b=,
out_dtype=tl.float32)` API that does block-scaled mma in a single instruction
(maps to `tcgen05.mma.kind::mxf8f6f4` directly). **Worth investigating** —
that would let us collapse the per-K-block manual scale multiply into the mma
itself. As of Triton 3.2 the API is `tl.dot_scaled` (see
`https://github.com/triton-lang/triton/blob/main/python/triton/language/standard.py`).
But (a) it's experimental, (b) it may not be deterministic in the same way the
manual path is, (c) it may not be available in the Triton version bundled with
vLLM 0.20.x. Conservative recommendation: ship the manual Style-B kernel first,
revisit `tl.dot_scaled` once batch-invariance is confirmed empirically.

## 2.4 Integration plan

### 2.4.1 vLLM dispatcher integration

The cleanest hook is at the **linear-method** level, not at the aten level —
because the aten dispatcher loses MXFP8 metadata (scales aren't on the tensor;
they're attributes of the `Layer` object).

Concretely, modify
`3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/` to add a new kernel
class `BatchInvariantMxfp8LinearKernel`:

```python
# 3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py (NEW)

class BatchInvariantMxfp8LinearKernel(Mxfp8LinearKernel):

    @classmethod
    def is_supported(cls, compute_capability=None):
        # Always supported when VLLM_BATCH_INVARIANT=1
        if not envs.VLLM_BATCH_INVARIANT:
            return False, "only used in batch-invariant mode"
        return True, None

    @classmethod
    def can_implement(cls, c): return True, None

    def process_weights_after_loading(self, layer):
        # No swizzling — keep weight as [N, K] fp8 and scale as [N, K/32] uint8.
        # The Triton kernel reads them in this layout directly.
        weight = layer.weight.data           # [N, K]
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight       = Parameter(weight.contiguous(),       requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.contiguous(), requires_grad=False)

    def apply_weights(self, layer, x, bias=None):
        # x: bf16 [..., K]. We do W8A16 (don't quantise activations).
        N, K = layer.weight.shape
        input_2d = x.view(-1, K)
        out = mxfp8_matmul_persistent(
            a=input_2d,
            b=layer.weight.t(),                  # [K, N] view
            b_scale=layer.weight_scale.t(),      # [K/32, N] view (or transpose)
            a_scale=None,
            bias=bias,
            out_dtype=x.dtype,
        )
        return out.view(*x.shape[:-1], N)
```

Then in `kernels/linear/__init__.py:245–250` move this new kernel to the front
of `_POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]`:

```python
_POSSIBLE_MXFP8_KERNELS = {
    PlatformEnum.CUDA: [
        BatchInvariantMxfp8LinearKernel,            # ← NEW, first
        FlashInferCutlassMxfp8LinearKernel,
        MarlinMxfp8LinearKernel,
        EmulationMxfp8LinearKernel,
    ],
    ...
}
```

The existing `init_mxfp8_linear_kernel` picks the first kernel whose
`is_supported` and `can_implement` both succeed; under `VLLM_BATCH_INVARIANT=1`
ours wins automatically.

**Note**: this is a W8A16 path (bf16 activations, mxfp8 weights). For full
W8A8 BI, an additional `a_scale` argument is needed; the kernel already
supports it via `A_IS_FP8=True`. The activation quantisation would have to
happen in a separate Triton kernel that's also deterministic (no problem —
`mxfp8_e4m3_quantize_torch` in `mxfp8_utils.py:38–84` is a pure elementwise
op + per-block amax, both naturally batch-invariant). For inference, W8A16 is
usually sufficient since the activations don't need to be fp8 for the matmul
to use fp8 Tensor Cores (Blackwell's `mma.kind::e4m3` supports
fp8×fp8→fp32, fp8×bf16→fp32, and bf16×fp8→fp32).

#### Why not patch `aten::mm`?

Because at the `aten::mm` site we receive `(bf16 activation, bf16 weight)` —
the MXFP8 dequant has already happened upstream in `ModelOptMxFp8LinearMethod.apply`
or in the emulation kernel. If we want to *avoid* that dequant, we must
intercept above aten, at the `linear_method.apply()` level. Hence the kernel
class hierarchy.

The current BF16 BI `aten::mm` / `addmm` patches stay in place — they catch
the residual matmuls (e.g. `lm_head` if it's not quantised) and the BI ops
inside `BatchInvariantMxfp8LinearKernel` itself if any.

### 2.4.2 Megatron / TE dispatcher integration

The hook point is `_te_general_gemm_patched` in
`3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/transformer/custom_layers/batch_invariant_kernels.py:830–874`.

Currently this function:

1. Calls `_extract_te_gemm_args(args, kwargs)` to parse the TE signature.
2. Calls `_is_supported_dtype_for_bik(A.dtype)` — line 851–852, which only
   accepts fp16/bf16/fp32 and **raises for fp8**.
3. Calls `BatchInvariantTEGemmFn.apply(A, B, bias, out_dtype, layout)` which
   calls `matmul_persistent` for the BF16 kernel.

Changes required:

```python
def _te_general_gemm_patched(*args, **kwargs):
    A, B, out_dtype, layout, out, bias, grad = _extract_te_gemm_args(args, kwargs)
    # ... existing guardrails ...

    # NEW: detect MXFP8 quantised tensors
    if _is_mxfp8_tensor(A) or _is_mxfp8_tensor(B):
        # Both must be MXFP8 (or one MXFP8, one bf16 for W8A16)
        result = BatchInvariantMxfp8TEGemmFn.apply(A, B, bias if not grad else None, out_dtype, layout)
    elif _is_supported_dtype_for_bik(A.dtype):
        result = BatchInvariantTEGemmFn.apply(A, B, bias if not grad else None, out_dtype, layout)
    else:
        raise RuntimeError(f"Unsupported dtype for batch-invariant GEMM: {A.dtype}, {B.dtype}")

    # ... existing bias-grad and out handling ...
```

`_is_mxfp8_tensor` checks `isinstance(t, transformer_engine.pytorch.tensor.mxfp8_tensor.MXFP8Tensor)`
(TE class) — guarded by `try/except ImportError` for older TE versions.

`BatchInvariantMxfp8TEGemmFn` (new autograd Fn, parallel to
`BatchInvariantTEGemmFn` at line 713–827) extracts:

- `A._data` (fp8 e4m3fn tensor)
- `A._scale_inv` (uint8 E8M0 tensor) — TE's name; equivalent to our
  `a_scale`. Note: TE stores scales in a swizzled layout under MXFP8 recipe;
  we need to un-swizzle to row-major `[M, K/32]` to match our kernel's
  expectation, OR write a swizzled-aware kernel variant. **The un-swizzle
  step is itself batch-invariant** (pure permutation), so it's fine.
- Same for B.

Then dispatch into `mxfp8_matmul_persistent(...)`.

**Critical**: in TE, weight tensors under MXFP8 recipe are quantised
**lazily**, sometimes both as the row-major and column-major layouts (for
forward/backward). Pulling them apart cleanly requires looking at TE's
`Float8Tensor` / `MXFP8Tensor` `.dequantize()` and `.transpose()` methods.

#### Why `general_gemm` and not aten?

Same reason as vLLM: TE doesn't go through `aten::mm` for MXFP8 GEMMs — it
goes through `cublasLtMatmul` directly via `transformer_engine_torch` C++.
The aten patch never sees them. The Python wrapper `general_gemm` is the
highest-level hook before TE's C++ entry.

### 2.4.3 Where the entry points live (exact line numbers to modify)

| Engine | File | Lines | Modification |
|---|---|---|---|
| vLLM | `3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py` | NEW file | Add `BatchInvariantMxfp8LinearKernel`. |
| vLLM | `3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/__init__.py` | 4–12 | Export the new class. |
| vLLM | `3rdparty/vllm/vllm/model_executor/kernels/linear/__init__.py` | 245–250 | Prepend `BatchInvariantMxfp8LinearKernel` to the CUDA priority list. |
| vLLM | `3rdparty/vllm/vllm/model_executor/layers/batch_invariant.py` | 41 (after `_compute_pid`) | Add `mxfp8_matmul_kernel_persistent` and `mxfp8_matmul_persistent`. |
| Megatron | `3rdparty/.../batch_invariant_kernels.py` | 62 (after `_compute_pid`) | Add `mxfp8_matmul_kernel_persistent` and `mxfp8_matmul_persistent`. |
| Megatron | `3rdparty/.../batch_invariant_kernels.py` | 709–710 (`_is_supported_dtype_for_bik`) | Add fp8e4m3fn to the allow list. |
| Megatron | `3rdparty/.../batch_invariant_kernels.py` | 713 (after `BatchInvariantTEGemmFn`) | Add `BatchInvariantMxfp8TEGemmFn` class. |
| Megatron | `3rdparty/.../batch_invariant_kernels.py` | 830 (`_te_general_gemm_patched`) | Insert the MXFP8 branch (see snippet above) before the `BatchInvariantTEGemmFn.apply` call. |

The two `mxfp8_matmul_persistent` host wrappers + kernels should be
**byte-identical** between the two files (same goal as the BF16 kernels), so
that a `diff` between them only shows the cosmetic `_compute_pid` arity
mismatch and nothing else.

### 2.4.4 Test plan (to add later)

- Unit test in `my_script/`: load an MXFP8 weight slice via
  `convert_hf_bf16_ckpt_to_mxfp8.py`, compare `mxfp8_matmul_persistent(x_bf16,
  w_fp8, scale)` against `(x_bf16 @ dequant_mxfp8_to_bf16(w_fp8, scale).t())`
  with both engines' BF16 BI matmul. Expected: equal up to fp32 rounding
  noise across the K reduction (the dequant + bf16 GEMM materialises bf16
  intermediates per-K-block, so it's a slightly different math; for a strict
  reference, run dequant-to-fp32 then GEMM in fp32 and compare).
- Batch-invariance test: same as `test_vllm_batch_invariant.py` but with
  MXFP8 weights. Already exists in `test_vllm_batch_invariant.py`; would
  pass once the new kernel routes.
- Cross-engine bit-identity: re-run the `compare.py` workflow from
  `llama3_8b_numeric_mismatch.md` with MXFP8 weights loaded on both sides.
  Expected end state: identical to BF16 BI behaviour
  (`llama3_8b_numeric_mismatch.md:894–911`, final logits bit-identical).

## 2.5 Risks and open questions

These are substantive, not nits — each one should be empirically resolved
before merging.

1. **Is `tl.dot(fp8, fp8, fp32)` deterministic with the Triton version each
   engine ships?** The mma instruction is — the hardware reduction tree
   inside `mma.sync.aligned.kind::e4m3` is fixed. But Triton's `tl.dot` may
   apply an additional warp-level reduction across the 16×16 result tile to
   build the full BM×BN tile, and that reduction's exact tree could change
   between Triton versions. We need to fix Triton at one version (vLLM 0.20.x
   pins Triton via `vllm-flash-attn`; Megatron pins Triton via TE) and check
   that the same Triton compiles both BI kernels. If versions diverge, the
   "bit-identical" claim across engines breaks. **Action**: pin Triton
   explicitly in both `pyproject.toml` extras; document the version in the
   report below.

2. **Triton's experimental `tl.dot_scaled` (block-scaled mma) — is it
   batch-invariant, and is it available?** This is the "one instruction does
   block-scaled MMA + scale" intrinsic. It maps directly to
   `tcgen05.mma.kind::mxf8f6f4` on Blackwell and is the fastest MXFP8 path
   possible. But:
   - `tl.dot_scaled` only entered Triton main around 3.1–3.2. vLLM 0.20.x
     bundles Triton 2.2.0 → not available.
   - Even when available, the Triton compiler may pick different tile shapes
     for `tl.dot_scaled` based on K (in particular if the inner-K is large,
     it may apply an internal split-K), which would break BI.

   **Action**: stick with the manual Style-B kernel for now. Re-evaluate when
   both engines move to Triton 3.x.

3. **Does cuBLASLt-based MXFP8 GEMM (TE's default under MXFP8 recipe) pick
   an algorithm based on M?** Yes — `cublasLtMatmulAlgoGetHeuristic` looks at
   the problem shape including M. Two M values that differ slightly can land
   on different tile schedules (`CTA_M_128_N_128_K_64` vs
   `CTA_M_256_N_128_K_64`), changing the accumulation order. This is
   *exactly* the same pattern that broke FA4 BI in vLLM 0.20.x. Therefore:
   - **TE MXFP8 GEMM is not batch-invariant out of the box.**
   - The `_te_general_gemm_patched` override is **necessary**, not optional,
     under MXFP8.

4. **TE MXFP8 tensor's scale layout.** TE under `MXFP8BlockScaling` stores
   scales in a layout that's optimised for cuBLASLt's block-scaled mma
   (specifically, a 128×4 tile-major layout — see the `swizzle_mxfp8_scale`
   utility in vLLM's `mxfp8_utils.py:14–35` for the same idea). Our Triton
   kernel expects row-major `[M, K/32]` (or `[K/32, N]`). We need either:
   - An un-swizzle pre-pass inside `BatchInvariantMxfp8TEGemmFn` (extra
     memory bandwidth, but only once per layer load), or
   - A swizzled-aware Triton kernel that reads scales from the swizzled
     layout directly.
   The former is simpler and easier to verify; the latter is faster. Start
   with un-swizzle, optimise later.

5. **Saturating cast behaviour for activation quantisation (W8A8 path).**
   When activations are quantised on the fly per call (vLLM's current MXFP8
   apply path), the quantiser must produce **identical** fp8 output and
   E8M0 scale for every call with the same activation tensor. The
   FlashInfer `mxfp8_quantize` implementation may use shared-memory
   reductions for the per-block amax. Need to verify (a) batch-invariance of
   `flashinfer.mxfp8_quantize`, and (b) bit-identical output between
   FlashInfer's CUDA quantiser and the Triton/Python implementation in
   `_mxfp8_e4m3_quantize_torch` (`mxfp8_utils.py:38–84`). If they differ, the
   two engines' on-the-fly activation quantisation will diverge even with
   identical BI matmul.

6. **Bias semantics across engines.** For Llama-3.1-8B there is no bias, but
   for completeness: under MXFP8 we should choose **bias-in-accumulator,
   single bf16 cast at store** (matching vLLM `addmm_batch_invariant`). This
   is a deviation from Megatron's `BatchInvariantTEGemmFn` which does the
   bias add on the bf16 output. **Action**: when porting this design to the
   BF16 kernels for full parity, fix Megatron's bias path to match vLLM's.

7. **Performance gap vs. FlashInfer CUTLASS.** A hand-written Triton kernel
   with manual per-block scale multiply will be ~2–3× slower than
   FlashInfer's CUTLASS BlockScaledMma (which uses the dedicated
   `tcgen05.mma.kind::mxf8f6f4` instruction). For inference BI mode the
   tradeoff is acceptable (correctness > throughput); for training BI mode
   it's a bigger hit. Long-term direction: see #2 (move to `tl.dot_scaled`).

8. **lm_head quantisation.** Most MXFP8 ModelOpt checkpoints leave `lm_head`
   in bf16 (see `convert_hf_bf16_ckpt_to_mxfp8.py:167–168` which skips
   lm_head). The lm_head GEMM thus continues to go through the BF16 BI
   `aten::mm` patch. No change needed there.

## 2.6 Summary

- **Kernel choice**: Style B (per-K-block fp8 mma + per-block scale multiply
  via `tl.dot(a_fp8, b_fp8, fp32_acc)` with manual scale multiply), tile
  shape (BM=128, BN=128, BK=64, GROUP=8, ns=3, nw=8) — identical to BF16 BI
  for easy diff.
- **Scale encoding**: E8M0 uint8 → fp32 multiplier via bit-cast
  (`uint8 << 23` then `bitcast` to fp32), with `scale==0` ⇒ multiplier=1
  guard.
- **Batch-invariance**: persistent grid + fixed K-loop + fixed sub-block
  order. The exact same trick the BF16 kernel uses, extended only by
  per-K-block scale factors that are themselves loaded in a fixed order.
- **vLLM integration**: new `BatchInvariantMxfp8LinearKernel` class in
  `kernels/linear/mxfp8/`, registered first in `_POSSIBLE_MXFP8_KERNELS` so
  it's chosen automatically under `VLLM_BATCH_INVARIANT=1`.
- **Megatron integration**: extend `_te_general_gemm_patched` to detect
  TE's MXFP8 tensors and route them to `BatchInvariantMxfp8TEGemmFn`,
  which un-swizzles the TE scale layout and calls
  `mxfp8_matmul_persistent`.
- **Biggest risk**: TE's MXFP8 scale layout (swizzled, layout differs by
  TE version) — needs a robust un-swizzle in the Megatron patch. Second-
  biggest risk: cross-engine Triton version skew breaking the
  "bit-identical kernel between engines" property we relied on for BF16
  parity.

Once both kernels are wired, the expected MXFP8 BI parity behaviour is
exactly the same as the BF16 BI parity behaviour documented in
`llama3_8b_numeric_mismatch.md`:

- Same `linear_qkv.linear` post-norm output to fp32 precision.
- Same residual stream across all 32 layers given the other BI fixes (RoPE,
  SwiGLU, SDPA, RMSNorm) are kept.
- Final logits bit-identical.
