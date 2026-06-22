---
name: debug-generation-training-mismatch
description: Debug NeMo-RL generation-vs-training numeric mismatches, especially vLLM versus Megatron logprob drift, true-on-policy batch-invariant mode, BF16/MXFP8 parity, tensor dumps, and standalone capture comparisons.
when_to_use: vLLM Megatron mismatch; generation training mismatch; logprob mismatch; true-on-policy debug; batch-invariant debug; BF16 parity; MXFP8 parity; tensor dumps; compare vLLM and Megatron captures.
---

# Debug Generation/Training Mismatch

Use this when a NeMo-RL run shows generation/training logprob drift, or when
you need to localize a vLLM-vs-Megatron numeric mismatch. Work from the earliest
comparable tensor to the first divergent tensor. Do not start by comparing final
logits unless the token positions and padding are known to match.

## Current Controls

Use the policy-level flags, not the old per-engine patch flags:

```bash
++policy.bf16_true_on_policy=true
++policy.mxfp8_matmul_batch_invariant=true
NEMO_RL_MXFP8_MATMUL_BI_BACKEND=native  # or qdq
```

`policy.bf16_true_on_policy` turns on Megatron BI mode, vLLM
`VLLM_BATCH_INVARIANT`, and the BF16 parity patches. vLLM RMSNorm, RoPE, and
SwiGLU are patched to match Megatron; Megatron attention is still patched to
use vLLM FA2. It requires sync vLLM
(`policy.generation.vllm_cfg.async_engine=false`). For vLLM tensor hooks, force
eager mode.

For BF16 true-on-policy rollout speed, prefer vLLM CUDA graph-only mode rather
than compiled vLLM. In the GRPO path this should be auto-configured when
`policy.bf16_true_on_policy=true` and `enforce_eager=true` is not explicitly
set:

```bash
++policy.generation.vllm_kwargs.compilation_config.mode=0
++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
++policy.generation.vllm_kwargs.compilation_config.splitting_ops=[]
```

This disables torch/vLLM compilation while keeping decode CUDA graph capture.
True-on-policy vLLM class patches must be installed before vLLM engine
construction because CUDA graphs are captured during engine init.

`policy.mxfp8_matmul_batch_invariant` requires `bf16_true_on_policy=true`,
vLLM `precision=fp8`, `is_mx=true`, and Megatron `fp8_cfg.enabled=true` with
`fp8_recipe=mxfp8`. Select the MXFP8 backend with
`NEMO_RL_MXFP8_MATMUL_BI_BACKEND`: `native` uses NeMo-RL's native FP8 BI
kernel runtime patch; `qdq` dequants MXFP8 operands and reuses the BF16 BI
matmul.

## Main Workflow

1. Reproduce with the smallest deterministic setup: one node, TP=PP=1 if
   possible, fixed prompts, sync vLLM, eager vLLM, no async rollout, and a small
   number of prompts.
2. First run the NeMo-RL comparison path. Use tensor dumps if aggregate logprob
   metrics show drift:

```bash
uv run nemo_rl/algorithms/run_logprob_comparison.py \
  --config examples/configs/grpo_math_1B.yaml \
  --tensor-dump-dir /tmp/nemo-rl-mismatch \
  --tensor-dump-max-steps 1 \
  --tensor-dump-max-calls-per-module 1 \
  --vllm-prefill-check-first-batch \
  --inspect-vllm-layernorm-impl \
  policy.bf16_true_on_policy=true \
  policy.generation.vllm_cfg.async_engine=false
```

3. Compare saved NeMo-RL tensor dumps:

```bash
uv run python my_script/compare_logprob_tensor_dumps.py \
  --dump-dir /tmp/nemo-rl-mismatch
```

4. If NeMo-RL dumps are too coarse, run standalone vLLM and Megatron captures
   on the same prompt batch, then compare them with `my_script/compare.py`.
5. Localize by module order: embeddings, first RMSNorm output, QKV linear input
   and output, RoPE Q/K, SDPA output, output projection, post-attention
   residual, second RMSNorm, SwiGLU input/output, down projection, layer output.
6. Stop when you find the first tensor whose input is bit-identical and output
   diverges. That kernel or boundary is the next fix; downstream drift is
   secondary.

## Standalone Capture Commands

BF16:

```bash
uv run --extra vllm python my_script/vllm_forward.py \
  --batch-invariant --capture-layers 0,20

uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py \
  --batch-invariant \
  --split-all-fused \
  --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm \
  --capture-layers 0,20

uv run --extra mcore python my_script/compare.py --batch-invariant
```

MXFP8:

```bash
uv run --extra vllm python my_script/convert_hf_bf16_ckpt_to_mxfp8.py \
  --input-hf-path meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /path/to/Llama-3.1-8B-Instruct.mxfp8 \
  --output-dtype mxfp8

uv run --extra vllm python my_script/vllm_forward.py \
  --mxfp8 \
  --model /path/to/Llama-3.1-8B-Instruct.mxfp8 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --batch-invariant --capture-layers 0,20

uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py \
  --mxfp8 --batch-invariant \
  --split-all-fused \
  --vllm-rope --vllm-swiglu --vllm-sdpa --vllm-rmsnorm \
  --capture-layers 0,20

uv run --extra mcore python my_script/compare.py --mxfp8 --batch-invariant
```

Run vLLM and Megatron sequentially on the same GPU when possible. vLLM's KV
cache profile can leave memory fragmented for a following Megatron process.

## Known Mismatch Sources

- Fusion boundary: TE `LayerNormColumnParallelLinear` hooks capture the pre-norm
  residual stream, while vLLM hooks usually capture the post-RMSNorm tensor.
  Use `--split-fused` for layer 0 or `--split-all-fused` for full-depth parity.
- RMSNorm: true-on-policy routes vLLM RMSNorm through Megatron-Core's
  `BatchInvariantRMSNormFn`, including the residual-add path.
- RoPE: vLLM and Megatron differ in cos/sin caching, casting, and kernel
  ordering. True-on-policy patches vLLM base RoPE to Megatron's unfused
  PyTorch formula.
- SwiGLU: vLLM's historical CUDA `SiluAndMul` and Megatron's fused elementwise
  path can differ by one BF16 rounding event. True-on-policy patches vLLM
  `SiluAndMul` to Megatron's fused SwiGLU function.
- SDPA: TE attention and vLLM attention can differ even with identical Q/K/V.
  Use the vLLM-style SDPA patch and make sequence lengths explicit.
- MXFP8 padding: Megatron/TE MXFP8 activations require dimensions divisible by
  32. The standalone Megatron script pads sequence length for MXFP8; do not
  compare final logits at padded positions.
- MXFP8 scale layout: TE may swizzle scales for GEMM. Compact scales are needed
  when dequanting MXFP8 tensors for comparison or QDQ BI mode.
- MXFP8 activation quantization: BF16 weights alone are not enough. For parity,
  both engines must quantize/dequantize weights and activations equivalently.
- Final logits can mislead. For padded MXFP8 runs, compare real token positions
  and per-layer residual streams before trusting last-position logits.
- vLLM compiled mode can change BF16 generation/prompt logprobs relative to
  vLLM eager even when generated token ids match. In the Llama3.1 8B BF16
  investigation, compiled-with-CUDA-graph and compiled-without-CUDA-graph
  drifted identically from eager, and the drift also reproduced without NeMo
  patches. Treat this as a vLLM compiled numeric path difference, not a CUDA
  graph replay issue.
- Policy-rescoring generated tokens with the training policy can make a final
  comparison exact, but it is not a raw vLLM on-policyness fix. Do not use it
  to claim true-on-policy generation logprobs.

## vLLM Compile/CUDA Graph Triage

Use `tools/model_diagnostics/6.vllm_compile_logprob_drift.py` to isolate vLLM
eager-vs-candidate drift on fixed eager-generated token ids. Run both patched
and `--no-install-nemo-patches` variants before blaming NeMo patches.

Interpretation:

- Eager decode-vs-prefill exact, compiled candidates drift, and token ids match:
  compiled vLLM changed the numeric path.
- Compiled candidates drift the same with `cudagraph_mode=NONE`: CUDA graph
  replay is not the direct cause.
- Graph-only candidate (`mode=0`, `cudagraph_mode=FULL_DECODE_ONLY`,
  `splitting_ops=[]`) exact versus eager: use graph-only mode for speed.
- Full NeMo-RL graph-only run drifts but the micro is exact: check whether
  true-on-policy class patches were installed before vLLM engine construction
  and CUDA graph capture.

For final validation, run `run_logprob_comparison.py` with
`--vllm-prefill-check-first-batch`. A good BF16 graph-only run should log
`CompilationMode.NONE`, `CUDAGraphMode.FULL_DECODE_ONLY`, disabled Inductor
compilation, completed graph capture, and `mean_abs=0`, `max_abs=0` for both
the vLLM prefill check and generation-vs-training comparison.

## Retained Scripts

- `my_script/tensor_capture.py`: shared forward-hook capture helpers.
- `my_script/vllm_forward.py`: standalone vLLM prefill capture.
- `my_script/megatron_forward.py`: standalone Megatron capture with BI,
  vLLM-style patches, splitting, and MXFP8 support.
- `my_script/compare.py`: standalone vLLM/Megatron capture comparison.
- `my_script/compare_logprob_tensor_dumps.py`: NeMo-RL policy/generation tensor
  dump comparison.
- `my_script/vllm_generation_prefill_logprob_consistency.py`: vLLM-only check
  that rollout logprobs match prompt-logprob prefill scoring.
- `my_script/compare_random_sdpa_qkv.py`: isolated random Q/K/V attention-path
  check for SDPA layout and paged/direct differences.
- `my_script/compare_mxfp8_quant.py`: vLLM/TE MXFP8 quantizer byte comparison.
- `my_script/compare_megatron_fused_norm_linear.py`: Megatron fused-vs-split
  LayerNormLinear diagnostic.
- `my_script/convert_hf_bf16_ckpt_to_mxfp8.py`: one-time HF BF16 to MXFP8
  checkpoint conversion for vLLM capture.
- `tools/model_diagnostics/6.vllm_compile_logprob_drift.py`: vLLM-only
  eager-vs-compiled/CUDA-graph drift reproducer on fixed eager-generated token
  ids.

Generated `.pt`, `.json`, `.png`, logs, and `__pycache__` files should stay out
of the repo unless the user explicitly asks to preserve a specific artifact.
