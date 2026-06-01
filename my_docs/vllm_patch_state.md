# vLLM patch state for BF16 bit-identity

The BF16 bit-identical result documented in
[`llama3_8b_numeric_mismatch.md`](llama3_8b_numeric_mismatch.md) depends on
two pieces of state outside the NeMo-RL repo:

1. **vLLM commit**: `88d34c6409e9fb3c7b8ca0c04756f061d2099eb1`
   (vLLM 0.20.1.dev0+g88d34c640, checked out under `3rdparty/vllm/`).
2. **Local edit** on top of that SHA: `RMSNorm.forward_cuda` in
   `vllm/model_executor/layers/layernorm.py` is patched so that, under
   `VLLM_BATCH_INVARIANT=1` with `residual is not None`, the residual-add
   is done in bf16 in-place and the normalization is dispatched through
   `rms_norm_batch_invariant` (the BI Triton kernel) instead of vLLM's
   `fused_add_rms_norm` C++ kernel (which uses `cub::BlockReduce` +
   `rsqrtf` and diverges from Megatron at large input magnitudes).

The exact diff is preserved in [`vllm_local_patches.diff`](vllm_local_patches.diff).

## Reproducing this state

```bash
# Inside 3rdparty/vllm/
git checkout 88d34c6409e9fb3c7b8ca0c04756f061d2099eb1
git apply ../../my_docs/vllm_local_patches.diff
```

The `vllm/model_executor/layers/batch_invariant.py` is also part of the
diff but only contains kernel-tuning entries that match what shipped on
this vLLM SHA; if `git apply` reports a clean apply, the resulting tree
reproduces the BF16-bit-identical configuration.

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
