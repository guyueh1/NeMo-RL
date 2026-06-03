r"""Run vLLM prefill and save last-token logits.

Run a batch of real prompts through a vLLM Llama-3.1-8B engine in prefill mode
(eager) and save the last-token logits for each prompt.

BF16 (default):
    uv run --extra vllm python my_script/vllm_forward.py

MXFP8 (pass ``--mxfp8`` and point ``--model`` at an MXFP8 checkpoint produced
by ``my_script/convert_hf_bf16_ckpt_to_mxfp8.py``):
    uv run --extra vllm python my_script/vllm_forward.py \\
        --mxfp8 --model /path/to/llama3.1-8b-instruct-mxfp8

By default loads 32 prompts from ``openai/gsm8k`` (``question`` field) and
sends them to vLLM as a single batch. The output ``.pt`` payload contains a
per-prompt list of token ids plus a single ``(N, V)`` tensor of last-token
logits aligned with ``compare.py``'s scatter plot.
"""

import argparse
import os

# Required so apply_model() can ship our hook-installer closure to the worker
# process via pickle (the default msgpack encoder rejects functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Parse --batch-invariant *before* importing vllm so we can flip VLLM_BATCH_INVARIANT
# in the environment that the worker process will inherit.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--batch-invariant", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.batch_invariant:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch
from tensor_capture import (
    install_debug_tensor_hooks,
    save_debug_tensor_capture_from_env,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_SUBSET = "main"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_FIELD = "question"
DEFAULT_NUM_PROMPTS = 32


def unwrap_apply_model_result(result):
    """Return the single-worker result from vLLM's apply_model result."""
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item:
                return item
        return result[0] if result else {}
    return result


def install_rmsnorm_bi_residual_patch(model):
    """Route fused add+RMSNorm through the BI Triton kernel.

    Monkey-patch ``RMSNorm.forward_cuda`` when a residual tensor is provided.

    Mirrors the small upstream edit we are reverting in
    ``3rdparty/vllm/vllm/model_executor/layers/layernorm.py``. Without this,
    when ``VLLM_BATCH_INVARIANT=1`` and ``residual`` is not ``None``,
    ``forward_cuda`` falls through to ``fused_add_rms_norm`` (cub::BlockReduce
    + rsqrtf), which diverges from the BI Triton kernel
    (``tl.sum BLOCK_SIZE=1024`` + ``1.0/sqrt``) at large magnitudes — observed
    as ~1 bf16 ULP drift starting at layer 6 vs Megatron's BI RMSNorm path.

    Runs inside the vLLM worker via ``llm.apply_model``.
    """
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
    from vllm.model_executor.layers.layernorm import RMSNorm

    orig_forward_cuda = RMSNorm.forward_cuda

    def patched_forward_cuda(self, x, residual=None):
        if residual is not None:
            residual.add_(x)
            return (
                rms_norm_batch_invariant(
                    residual, self.weight.data, self.variance_epsilon
                ),
                residual,
            )
        return orig_forward_cuda(self, x, residual)

    RMSNorm.forward_cuda = patched_forward_cuda

    # CustomOp.__init__ binds `_forward_method = self.forward_cuda` once at
    # instance construction, so a class-level patch installed after the model
    # is loaded does not propagate to existing RMSNorm modules. Rebind each
    # instance to the (now-patched) bound method.
    patched_count = 0
    for mod in model.modules():
        if isinstance(mod, RMSNorm):
            mod._forward_method = mod.forward_cuda
            patched_count += 1
    print(
        f"[vllm-patch] monkey-patched RMSNorm.forward_cuda + rebound "
        f"{patched_count} instances"
    )
    return None


def install_mxfp8_bi_emulation_patch(model):
    """Route MXFP8 GEMM through dequant + BF16 batch-invariant matmul.

    Monkey-patch ``vllm.utils.flashinfer.mm_mxfp8``.

    Must run inside the vLLM worker process (via ``llm.apply_model``) since
    the engine core is a subprocess; a module-level patch in the parent
    won't propagate. The only caller of ``vllm_flashinfer.mm_mxfp8`` is
    ``FlashInferCutlassMxfp8LinearKernel.apply_weights`` (line 87 of
    ``vllm/model_executor/kernels/linear/mxfp8/flashinfer.py``), and that
    call uses module attribute lookup at call time, so reassigning the
    module-level binding after model load still takes effect.

    The patch unswizzles the activation- and weight-side E8M0 scales
    (mm_mxfp8 receives them swizzled), dequants both operands to bf16,
    then routes through ``matmul_persistent`` (the BF16 BI matmul that
    matches Megatron's ``BatchInvariantTEGemmFn`` path when Megatron is
    run with ``--mxfp8-bi-dequant``).
    """
    import vllm.utils.flashinfer as vllm_flashinfer
    from vllm.model_executor.layers.batch_invariant import matmul_persistent
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        dequant_mxfp8_to_bf16,
    )

    def _unswizzle_mxfp8_scale(sf_1d, M, K):
        """Inverse of ``swizzle_mxfp8_scale``: flat 1D swizzled → ``[M, K/32]``."""
        factor = MXFP8_BLOCK_SIZE * 4  # 128
        num_m_tiles = (M + 127) // 128
        num_k_tiles = (K + factor - 1) // factor
        scale_cols = K // MXFP8_BLOCK_SIZE
        sf_5d = sf_1d.view(num_m_tiles, num_k_tiles, 32, 4, 4)
        sf_unswizzled = sf_5d.transpose(1, 3).contiguous()
        sf_padded = sf_unswizzled.view(num_m_tiles * 128, num_k_tiles * 4)
        return sf_padded[:M, :scale_cols].contiguous()

    def _bi_mm_mxfp8(A, B, A_scale, B_scale, out_dtype, backend="cutlass"):
        # A: [M, K] fp8 (activation, possibly padded). B: [K, N] fp8
        # (transposed view of original weight [N, K]).
        M, K = A.shape
        N = B.shape[1]

        A_scale_2d = _unswizzle_mxfp8_scale(A_scale, M, K)
        B_scale_2d = _unswizzle_mxfp8_scale(B_scale, N, K)

        # Dequant blocks live along the K axis of the original [N, K] weight,
        # so dequant via the original weight view (B.t()), not B itself.
        A_bf16 = dequant_mxfp8_to_bf16(A, A_scale_2d)  # [M, K]
        W_bf16 = dequant_mxfp8_to_bf16(B.t().contiguous(), B_scale_2d)  # [N, K]

        # BF16 BI matmul: [M, K] @ [K, N] -> [M, N].
        out = matmul_persistent(A_bf16, W_bf16.t())
        return out.to(out_dtype)

    vllm_flashinfer.mm_mxfp8 = _bi_mm_mxfp8
    print(
        "[vllm-patch] monkey-patched vllm.utils.flashinfer.mm_mxfp8 -> "
        "dequant + BF16 BI matmul (matmul_persistent)"
    )
    return None


def default_output(batch_invariant: bool, mxfp8: bool) -> str:
    parts = ["vllm_capture"]
    if mxfp8:
        parts.append("mxfp8")
    if batch_invariant:
        parts.append("bi")
    name = "_".join(parts) + ".pt"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def load_prompts(dataset: str, subset: str, split: str, field: str, n: int, seed: int):
    """Load `n` non-empty prompts from a HuggingFace dataset.

    Uses a deterministic shuffle (seed) before selecting the first `n` rows
    so that two runs of this script see the same prompts in the same order.
    """
    from datasets import load_dataset

    kwargs = {"split": split}
    if subset:
        ds = load_dataset(dataset, subset, **kwargs)
    else:
        ds = load_dataset(dataset, **kwargs)
    ds = ds.shuffle(seed=seed)
    prompts = []
    for row in ds:
        text = row.get(field)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        prompts.append(text)
        if len(prompts) >= n:
            break
    if len(prompts) < n:
        raise RuntimeError(
            f"only found {len(prompts)} non-empty '{field}' rows in "
            f"{dataset}:{subset}:{split}, needed {n}"
        )
    return prompts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model path or HF id. For --mxfp8, pass the MXFP8 "
        "ckpt path produced by convert_hf_bf16_ckpt_to_mxfp8.py.",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer source (HF id or path). Defaults to "
        "--model; useful when the MXFP8 ckpt dir does not "
        "bundle tokenizer files.",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=DEFAULT_NUM_PROMPTS,
        help="Number of prompts to draw from --dataset (default: 32).",
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-subset", default=DEFAULT_DATASET_SUBSET)
    p.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    p.add_argument("--dataset-field", default=DEFAULT_DATASET_FIELD)
    p.add_argument("--dataset-seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.add_argument("--batch-invariant", action="store_true")
    p.add_argument(
        "--mxfp8",
        action="store_true",
        help="Run the model in MXFP8 precision. Requires --model "
        "to point at an MXFP8-quantized ckpt (vLLM detects "
        "the quantization from the ckpt's quantization_config).",
    )
    p.add_argument(
        "--mxfp8-bi-emulation",
        action="store_true",
        help="Patch vllm's mm_mxfp8 (called by "
        "FlashInferCutlassMxfp8LinearKernel) to dequant both "
        "operands to bf16 and route through matmul_persistent "
        "(BF16 batch-invariant matmul). Mirrors Megatron's "
        "--mxfp8-bi-dequant. Only meaningful with --mxfp8 "
        "--batch-invariant.",
    )
    p.add_argument(
        "--capture-debug-tensors",
        action="store_true",
        help="Save layer-entry tensors for every decoder layer and "
        "input/output tensors for every module in decoder layer 0.",
    )
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.output is None:
        args.output = default_output(args.batch_invariant, args.mxfp8)
    return args


def main():
    args = parse_args()
    print(
        f"[vllm] precision={'mxfp8' if args.mxfp8 else 'bf16'} "
        f"batch_invariant={args.batch_invariant} "
        f"(VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', '0')})"
    )
    print(f"[vllm] model:     {args.model}")
    print(f"[vllm] tokenizer: {args.tokenizer}")
    print(
        f"[vllm] dataset:   {args.dataset}:{args.dataset_subset}:"
        f"{args.dataset_split} field={args.dataset_field!r} n={args.num_prompts}"
    )

    debug_capture_path = args.output + ".debug_tensors.pt"
    if args.capture_debug_tensors:
        os.environ["DEBUG_TENSOR_CAPTURE_PATH"] = debug_capture_path

    prompts = load_prompts(
        args.dataset,
        args.dataset_subset,
        args.dataset_split,
        args.dataset_field,
        args.num_prompts,
        args.dataset_seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids_list = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    seq_lens = [len(ids) for ids in token_ids_list]
    # Use the full tokenizer length (incl. added/special tokens), since the LM
    # head's logit dim matches this rather than the base vocab.
    vocab_size = len(tokenizer)
    print(
        f"[vllm] loaded {len(prompts)} prompts; "
        f"seq_len min/mean/max = {min(seq_lens)}/"
        f"{sum(seq_lens) / len(seq_lens):.1f}/{max(seq_lens)}; "
        f"vocab_size={vocab_size}"
    )

    # vLLM auto-detects MXFP8 via the ckpt's `quantization_config`; no extra
    # kwarg is needed. We keep activations in bf16 in both paths.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        enforce_eager=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_logprobs=vocab_size,
        seed=0,
    )

    if args.batch_invariant:
        # Correctness fix: route the fused-add RMSNorm through the BI Triton
        # kernel; otherwise the cub::BlockReduce/rsqrtf path runs and drifts.
        llm.apply_model(install_rmsnorm_bi_residual_patch)

    if args.mxfp8_bi_emulation:
        if not (args.mxfp8 and args.batch_invariant):
            raise SystemExit(
                "--mxfp8-bi-emulation requires --mxfp8 and --batch-invariant"
            )
        llm.apply_model(install_mxfp8_bi_emulation_patch)

    if args.capture_debug_tensors:
        hook_info = unwrap_apply_model_result(
            llm.apply_model(install_debug_tensor_hooks)
        )
        print(f"[vllm] debug tensor hooks installed: {hook_info}")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=None,
        logprobs=vocab_size,
        seed=0,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids_list],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print(
        f"[vllm] generated {sum(len(o.outputs[0].token_ids) for o in outputs)} "
        f"new tokens across {len(outputs)} prompts"
    )

    # Build (N, V) logprob tensor from outputs[i].outputs[0].logprobs[0],
    # which is a {token_id: Logprob(logprob=..., ...)} dict for the single
    # sampled token. With `logprobs=V`, the dict spans the full vocab.
    next_token_logprobs = torch.full(
        (len(outputs), vocab_size), float("nan"), dtype=torch.float32
    )
    for i, out in enumerate(outputs):
        step_logprobs = out.outputs[0].logprobs[0]
        for token_id, lp in step_logprobs.items():
            next_token_logprobs[i, token_id] = float(lp.logprob)
    if torch.isnan(next_token_logprobs).any():
        n_nan = int(torch.isnan(next_token_logprobs).sum())
        raise RuntimeError(
            f"vLLM logprobs incomplete: {n_nan} entries are NaN. "
            f"Check that SamplingParams(logprobs={vocab_size}) covered all tokens."
        )

    payload = {
        "prompts": prompts,
        "token_ids_list": token_ids_list,
        "seq_lens": seq_lens,
        "next_token_logprobs": next_token_logprobs,
    }
    if args.capture_debug_tensors:
        save_info = unwrap_apply_model_result(
            llm.apply_model(save_debug_tensor_capture_from_env)
        )
        print(f"[vllm] worker saved debug tensors: {save_info}")
        debug_capture = torch.load(
            debug_capture_path, map_location="cpu", weights_only=False
        )
        payload.update(debug_capture)
        num_first_layer = len(debug_capture.get("first_layer_inputs", {}))
        num_layers = debug_capture.get("num_layers", "?")
        print(
            f"[vllm] captured {num_first_layer} layer-0 modules; "
            f"num_layers={num_layers}"
        )
    torch.save(payload, args.output)
    print(f"[vllm] saved capture to {args.output}")


if __name__ == "__main__":
    main()
