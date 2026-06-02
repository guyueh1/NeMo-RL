# vLLM patch state for BF16 and MXFP8 bit-identity

The BF16 bit-identical result documented in
[`llama3_8b_numeric_mismatch.md`](llama3_8b_numeric_mismatch.md) and the
MXFP8 bit-identical result documented in
[`llama3_8b_mxfp8_numeric_mismatch.md`](llama3_8b_mxfp8_numeric_mismatch.md)
both depend on the following state outside the NeMo-RL repo:

1. **vLLM commit**: `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`
   (vLLM 0.20.1.dev0+g88d34c640, checked out under `3rdparty/vllm/`).

2. **Local edit (BF16)** on top of that SHA: `RMSNorm.forward_cuda` in
   `vllm/model_executor/layers/layernorm.py` is patched so that, under
   `VLLM_BATCH_INVARIANT=1` with `residual is not None`, the residual-add
   is done in bf16 in-place and the normalization is dispatched through
   `rms_norm_batch_invariant` (the BI Triton kernel) instead of vLLM's
   `fused_add_rms_norm` C++ kernel (which uses `cub::BlockReduce` +
   `rsqrtf` and diverges from Megatron at large input magnitudes).

3. **New file (MXFP8)**:
   `vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py` — adds a
   `BatchInvariantMxfp8LinearKernel` class that:
   - Dequantises the MXFP8 weight to bf16 via
     `dequant_mxfp8_to_bf16(weight, weight_scale)`.
   - Quant→dequant round-trips the bf16 activation via
     `mxfp8_e4m3_quantize` (FlashInfer on Blackwell) +
     `dequant_mxfp8_to_bf16` (mirrors TE's `fp8_autocast`).
   - Routes the GEMM through `matmul_persistent` (the BF16 batch-invariant
     matmul).
   The kernel is registered first in
   `_POSSIBLE_MXFP8_KERNELS[CUDA]` in
   `vllm/model_executor/kernels/linear/__init__.py`, so under
   `VLLM_BATCH_INVARIANT=1` it wins automatically over FlashInfer
   cutlass / Marlin / emulation.

The exact diff (BF16 layernorm + MXFP8 kernel registration) is preserved
in [`vllm_local_patches.diff`](vllm_local_patches.diff). The full new
kernel source is preserved in
[`vllm_mxfp8_batch_invariant_kernel.py`](vllm_mxfp8_batch_invariant_kernel.py)
— copy it into `3rdparty/vllm/vllm/model_executor/kernels/linear/mxfp8/`
verbatim to reproduce.

## Reproducing this state

```bash
# Inside 3rdparty/vllm/
git checkout 88d34c6409e9fb3c7b8ca0c04756f061d2099eb1
git apply ../../my_docs/vllm_local_patches.diff
cp ../../my_docs/vllm_mxfp8_batch_invariant_kernel.py \
   vllm/model_executor/kernels/linear/mxfp8/batch_invariant.py
```

## What's intentionally NOT in vLLM under this state

- No change to vLLM's BI RMSNorm Triton kernel formula. The kernel uses
  vLLM's original `inv_rms = 1.0 / tl.sqrt(mean_sq + eps)` (two rounded
  fp32 ops). Megatron's `install_vllm_style_rmsnorm` calls vLLM's BI
  RMSNorm Triton kernel directly so the two engines invoke the literally
  identical kernel.

- No FA2 / FA4 changes. The vLLM side still calls
  `vllm.vllm_flash_attn.flash_attn_varlen_func(fa_version=2, num_splits=1)`
  via its standard BI codepath. Megatron's `install_vllm_style_sdpa` calls
  the same function on the same kernel.

- No change to vLLM's MXFP8 quantiser. `mxfp8_e4m3_quantize` continues to
  dispatch to FlashInfer's CUDA kernel on Blackwell;
  `compare_mxfp8_quant.py` confirms it produces byte-identical output to
  TE's `tex.quantize` for the same bf16 input.
