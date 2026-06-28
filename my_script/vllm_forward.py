# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Run vLLM prefill and save last-token logits.

Run a batch of real prompts through a vLLM engine in prefill mode (eager) and
save the last-token logits for each prompt. Defaults to Llama-3.1-8B; pass
``--model nemotron3-nano`` or the full
``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`` id for Nemotron 3 Nano.

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
import pprint
from functools import wraps

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
    find_decoder_layers,
    inspect_vllm_layernorm_impl,
    install_debug_tensor_hooks,
    save_debug_tensor_capture_from_env,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NEMOTRON3_NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_ALIASES = {
    "llama": DEFAULT_MODEL,
    "llama3.1-8b": DEFAULT_MODEL,
    "llama-3.1-8b": DEFAULT_MODEL,
    "nemotron3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron-3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron3-nano-30b-a3b": NEMOTRON3_NANO_MODEL,
}
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_SUBSET = "main"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_FIELD = "question"
DEFAULT_NUM_PROMPTS = 32


class DebugTensorHookInstaller:
    def __init__(self, capture_layers: list[int]):
        self.capture_layers = capture_layers

    def __call__(self, model):
        return install_debug_tensor_hooks(model, capture_layers=self.capture_layers)


class VllmMambaInternalCaptureInstaller:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls

    def __call__(self, model):
        return install_vllm_mamba_internal_capture(
            model,
            max_calls=self.max_calls,
        )


class VllmMoEInternalCaptureInstaller:
    def __init__(
        self,
        max_kernel_calls: int,
        target_layers: list[int] | None = None,
    ):
        self.max_kernel_calls = max_kernel_calls
        self.target_layers = target_layers

    def __call__(self, model):
        return install_vllm_moe_internal_capture(
            model,
            max_kernel_calls=self.max_kernel_calls,
            target_layers=self.target_layers,
        )


class VllmMambaMegatronGatedRMSNormInstaller:
    def __init__(self, target_layers: list[int] | None):
        self.target_layers = target_layers

    def __call__(self, model):
        return install_vllm_mamba_megatron_gated_rmsnorm_patch(
            model,
            target_layers=self.target_layers,
        )


class VllmMoERouterWeightScalingInstaller:
    def __call__(self, model):
        return install_vllm_moe_router_weight_scaling_patch(model)


class VllmMoEReferenceFc2Installer:
    def __init__(self, target_layers: list[int] | None):
        self.target_layers = target_layers

    def __call__(self, model):
        return install_vllm_moe_reference_fc2_patch(
            model,
            target_layers=self.target_layers,
        )


def resolve_model_ref(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def is_nemotron3_nano_ref(value: str | None) -> bool:
    if value is None:
        return False
    value = resolve_model_ref(value)
    normalised = value.lower().replace("_", "-")
    return (
        value == NEMOTRON3_NANO_MODEL or "nvidia-nemotron-3-nano-30b-a3b" in normalised
    )


def model_output_tag(model: str, tokenizer: str | None) -> str:
    if is_nemotron3_nano_ref(model) or is_nemotron3_nano_ref(tokenizer):
        return "_nemotron3_nano"
    return ""


def parse_layer_indices(value: str) -> list[int]:
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid layer range {item!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise argparse.ArgumentTypeError("--capture-layers parsed to an empty set")
    return sorted(layers)


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


def _snapshot_debug_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot_debug_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_debug_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot_debug_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _call_arg(args, kwargs, index, name, default=None):
    if name in kwargs:
        return kwargs[name]
    if index < len(args):
        return args[index]
    return default


def _snapshot_vllm_mamba_metadata(metadata):
    if metadata is None:
        return None
    fields = (
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "chunk_size",
        "prep_initial_states",
        "has_initial_states_p",
        "seq_idx_p",
        "query_start_loc_p",
        "cu_chunk_seqlen_p",
        "last_chunk_indices_p",
        "state_indices_tensor_p",
        "block_idx_last_computed_token",
        "block_idx_last_scheduled_token",
        "block_idx_first_scheduled_token_p",
        "num_computed_tokens_p",
    )
    return {
        field: _snapshot_debug_value(getattr(metadata, field, None))
        for field in fields
        if hasattr(metadata, field)
    }


def _token_major_2d(value):
    if isinstance(value, torch.Tensor) and value.dim() == 2:
        return value.transpose(0, 1).contiguous()
    return None


def _tensor_debug_meta(value):
    if not isinstance(value, torch.Tensor):
        return _snapshot_debug_value(value)
    return {
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": tuple(value.stride()),
    }


def install_vllm_mamba_internal_capture(model, max_calls: int = 1):
    """Capture vLLM Mamba2 conv and SSD-scan boundaries inside the worker."""
    from vllm.model_executor.layers.mamba import mamba_mixer2

    capture = {
        "max_calls": max_calls,
        "mamba_conv1d_total_calls": 0,
        "mamba_scan_total_calls": 0,
        "mamba_conv1d_calls": [],
        "mamba_scan_calls": [],
    }
    model._debug_mamba_internal_capture = capture

    original_conv = getattr(
        mamba_mixer2,
        "_debug_unpatched_causal_conv1d_fn",
        mamba_mixer2.causal_conv1d_fn,
    )
    mamba_mixer2._debug_unpatched_causal_conv1d_fn = original_conv

    original_scan = getattr(
        mamba_mixer2,
        "_debug_original_mamba_chunk_scan_combined_varlen",
        mamba_mixer2.mamba_chunk_scan_combined_varlen,
    )
    mamba_mixer2._debug_original_mamba_chunk_scan_combined_varlen = original_scan

    def should_capture(key):
        return max_calls < 0 or len(capture[key]) < max_calls

    @wraps(original_conv)
    def wrapped_causal_conv1d_fn(*args, **kwargs):
        call_index = capture["mamba_conv1d_total_calls"]
        capture["mamba_conv1d_total_calls"] += 1
        should_save = should_capture("mamba_conv1d_calls")

        x = _call_arg(args, kwargs, 0, "x")
        weight = _call_arg(args, kwargs, 1, "weight")
        bias = _call_arg(args, kwargs, 2, "bias")
        query_start_loc = _call_arg(args, kwargs, 4, "query_start_loc")
        cache_indices = _call_arg(args, kwargs, 5, "cache_indices")
        has_initial_state = _call_arg(args, kwargs, 6, "has_initial_state")
        metadata = kwargs.get("metadata")

        output = original_conv(*args, **kwargs)

        if should_save:
            capture["mamba_conv1d_calls"].append(
                {
                    "call_index": call_index,
                    "input": _snapshot_debug_value(x),
                    "input_token_major": _snapshot_debug_value(_token_major_2d(x)),
                    "output": _snapshot_debug_value(output),
                    "output_token_major": _snapshot_debug_value(
                        _token_major_2d(output)
                    ),
                    "weight": _snapshot_debug_value(weight),
                    "bias": _snapshot_debug_value(bias),
                    "query_start_loc": _snapshot_debug_value(query_start_loc),
                    "cache_indices": _snapshot_debug_value(cache_indices),
                    "has_initial_state": _snapshot_debug_value(has_initial_state),
                    "metadata": _snapshot_vllm_mamba_metadata(metadata),
                }
            )
        return output

    @wraps(original_scan)
    def wrapped_mamba_chunk_scan_combined_varlen(*args, **kwargs):
        call_index = capture["mamba_scan_total_calls"]
        capture["mamba_scan_total_calls"] += 1
        should_save = should_capture("mamba_scan_calls")

        x = _call_arg(args, kwargs, 0, "x")
        dt = _call_arg(args, kwargs, 1, "dt")
        A = _call_arg(args, kwargs, 2, "A")
        B = _call_arg(args, kwargs, 3, "B")
        C = _call_arg(args, kwargs, 4, "C")
        out = kwargs.get("out")

        result = original_scan(*args, **kwargs)

        if should_save:
            capture["mamba_scan_calls"].append(
                {
                    "call_index": call_index,
                    "x": _snapshot_debug_value(x),
                    "dt": _snapshot_debug_value(dt),
                    "A": _snapshot_debug_value(A),
                    "B": _snapshot_debug_value(B),
                    "C": _snapshot_debug_value(C),
                    "D": _snapshot_debug_value(kwargs.get("D")),
                    "dt_bias": _snapshot_debug_value(kwargs.get("dt_bias")),
                    "seq_idx": _snapshot_debug_value(kwargs.get("seq_idx")),
                    "cu_seqlens": _snapshot_debug_value(kwargs.get("cu_seqlens")),
                    "cu_chunk_seqlens": _snapshot_debug_value(
                        kwargs.get("cu_chunk_seqlens")
                    ),
                    "last_chunk_indices": _snapshot_debug_value(
                        kwargs.get("last_chunk_indices")
                    ),
                    "initial_states": _snapshot_debug_value(
                        kwargs.get("initial_states")
                    ),
                    "state_dtype": str(kwargs.get("state_dtype")),
                    "out_after": _snapshot_debug_value(out),
                    "result": _snapshot_debug_value(result),
                }
            )
        return result

    mamba_mixer2.causal_conv1d_fn = wrapped_causal_conv1d_fn
    mamba_mixer2.mamba_chunk_scan_combined_varlen = (
        wrapped_mamba_chunk_scan_combined_varlen
    )

    return {"max_calls": max_calls, "patched": True}


def install_vllm_moe_internal_capture(
    model,
    max_kernel_calls: int = 2,
    target_layers: list[int] | None = None,
):
    """Capture vLLM Triton MoE expert GEMM and activation boundaries."""
    from vllm.model_executor.layers.fused_moe import fused_moe
    from vllm.model_executor.layers.fused_moe import modular_kernel as moe_mk

    selected_expert_ids = None
    selected_expert_names = {}
    missing_layers = []
    target_layer_set = None
    if target_layers is not None:
        layer_info = _collect_vllm_moe_fused_experts_for_layers(model, target_layers)
        target_layer_set = layer_info["target_layer_set"]
        selected_expert_ids = layer_info["selected_expert_ids"]
        selected_expert_names = layer_info["selected_expert_names"]
        missing_layers = layer_info["missing_layers"]

    capture = {
        "moe_internal_capture_max_kernel_calls": max_kernel_calls,
        "moe_internal_capture_target_layers": None
        if target_layer_set is None
        else sorted(target_layer_set),
        "moe_internal_capture_missing_layers": missing_layers,
        "moe_internal_capture_patched_module_names": [
            selected_expert_names[expert_id]
            for expert_id in sorted(selected_expert_names)
            if selected_expert_ids is not None and expert_id in selected_expert_ids
        ],
        "moe_kernel_total_calls": 0,
        "moe_activation_total_calls": 0,
        "moe_kernel_calls": [],
        "moe_activation_calls": [],
        "moe_modules": [],
        "enabled_stack": [],
    }
    for module_name, module in model.named_modules():
        runner = getattr(module, "runner", None)
        if runner is None:
            continue
        quant_method = getattr(module, "quant_method", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        impl = getattr(moe_kernel, "impl", None)
        fused_experts = getattr(impl, "fused_experts", None)
        prepare_finalize = getattr(impl, "prepare_finalize", None)
        capture["moe_modules"].append(
            {
                "module": module_name,
                "module_class": module.__class__.__name__,
                "runner_class": runner.__class__.__name__,
                "quant_method_class": quant_method.__class__.__name__
                if quant_method is not None
                else None,
                "kernel_impl_class": impl.__class__.__name__
                if impl is not None
                else None,
                "fused_experts_class": fused_experts.__class__.__name__
                if fused_experts is not None
                else None,
                "prepare_finalize_class": prepare_finalize.__class__.__name__
                if prepare_finalize is not None
                else None,
            }
        )
    model._debug_moe_internal_capture = capture

    original_kernel = getattr(
        fused_moe,
        "_debug_original_invoke_fused_moe_triton_kernel",
        fused_moe.invoke_fused_moe_triton_kernel,
    )
    fused_moe._debug_original_invoke_fused_moe_triton_kernel = original_kernel

    original_apply = getattr(
        fused_moe.TritonExperts,
        "_debug_original_apply",
        fused_moe.TritonExperts.apply,
    )
    fused_moe.TritonExperts._debug_original_apply = original_apply

    original_activation = getattr(
        moe_mk.FusedMoEExpertsModular,
        "_debug_original_activation",
        moe_mk.FusedMoEExpertsModular.activation,
    )
    moe_mk.FusedMoEExpertsModular._debug_original_activation = original_activation

    def should_capture(key):
        return max_kernel_calls < 0 or len(capture[key]) < max_kernel_calls

    def is_current_experts_enabled():
        if selected_expert_ids is None:
            return True
        return bool(capture["enabled_stack"] and capture["enabled_stack"][-1])

    @wraps(original_apply)
    def wrapped_triton_experts_apply(self, *args, **kwargs):
        if selected_expert_ids is None:
            return original_apply(self, *args, **kwargs)
        capture["enabled_stack"].append(id(self) in selected_expert_ids)
        try:
            return original_apply(self, *args, **kwargs)
        finally:
            capture["enabled_stack"].pop()

    @wraps(original_kernel)
    def wrapped_invoke_fused_moe_triton_kernel(*args, **kwargs):
        call_index = capture["moe_kernel_total_calls"]
        capture["moe_kernel_total_calls"] += 1
        should_save = is_current_experts_enabled() and should_capture(
            "moe_kernel_calls"
        )

        A = _call_arg(args, kwargs, 0, "A")
        B = _call_arg(args, kwargs, 1, "B")
        C = _call_arg(args, kwargs, 2, "C")
        A_scale = _call_arg(args, kwargs, 3, "A_scale")
        B_scale = _call_arg(args, kwargs, 4, "B_scale")
        topk_weights = _call_arg(args, kwargs, 5, "topk_weights")
        sorted_token_ids = _call_arg(args, kwargs, 6, "sorted_token_ids")
        expert_ids = _call_arg(args, kwargs, 7, "expert_ids")
        num_tokens_post_padded = _call_arg(args, kwargs, 8, "num_tokens_post_padded")
        mul_routed_weight = _call_arg(args, kwargs, 9, "mul_routed_weight")
        top_k = _call_arg(args, kwargs, 10, "top_k")
        config = _call_arg(args, kwargs, 11, "config")

        result = original_kernel(*args, **kwargs)

        if should_save:
            capture["moe_kernel_calls"].append(
                {
                    "call_index": call_index,
                    "A": _snapshot_debug_value(A),
                    "C_after": _snapshot_debug_value(C),
                    "A_scale": _snapshot_debug_value(A_scale),
                    "B_meta": _tensor_debug_meta(B),
                    "B_scale_meta": _tensor_debug_meta(B_scale),
                    "topk_weights": _snapshot_debug_value(topk_weights),
                    "sorted_token_ids": _snapshot_debug_value(sorted_token_ids),
                    "expert_ids": _snapshot_debug_value(expert_ids),
                    "num_tokens_post_padded": _snapshot_debug_value(
                        num_tokens_post_padded
                    ),
                    "mul_routed_weight": _snapshot_debug_value(mul_routed_weight),
                    "top_k": _snapshot_debug_value(top_k),
                    "config": _snapshot_debug_value(config),
                    "compute_type": str(kwargs.get("compute_type")),
                    "use_fp8_w8a8": _snapshot_debug_value(kwargs.get("use_fp8_w8a8")),
                    "use_int8_w8a8": _snapshot_debug_value(kwargs.get("use_int8_w8a8")),
                    "use_int8_w8a16": _snapshot_debug_value(
                        kwargs.get("use_int8_w8a16")
                    ),
                    "use_int4_w4a16": _snapshot_debug_value(
                        kwargs.get("use_int4_w4a16")
                    ),
                    "per_channel_quant": _snapshot_debug_value(
                        kwargs.get("per_channel_quant")
                    ),
                    "block_shape": _snapshot_debug_value(kwargs.get("block_shape")),
                    "B_bias_meta": _tensor_debug_meta(kwargs.get("B_bias")),
                }
            )
        return result

    @wraps(original_activation)
    def wrapped_activation(self, activation, output, input):
        call_index = capture["moe_activation_total_calls"]
        capture["moe_activation_total_calls"] += 1
        should_save = is_current_experts_enabled() and should_capture(
            "moe_activation_calls"
        )

        result = original_activation(self, activation, output, input)

        if should_save:
            capture["moe_activation_calls"].append(
                {
                    "call_index": call_index,
                    "activation": getattr(activation, "value", str(activation)),
                    "input": _snapshot_debug_value(input),
                    "output": _snapshot_debug_value(output),
                }
            )
        return result

    fused_moe.invoke_fused_moe_triton_kernel = wrapped_invoke_fused_moe_triton_kernel
    fused_moe.TritonExperts.apply = wrapped_triton_experts_apply
    moe_mk.FusedMoEExpertsModular.activation = wrapped_activation

    return {
        "max_kernel_calls": max_kernel_calls,
        "patched": True,
        "target_layers": None if target_layer_set is None else sorted(target_layer_set),
        "missing_layers": missing_layers,
        "patched_module_names": capture["moe_internal_capture_patched_module_names"],
        "num_moe_modules": len(capture["moe_modules"]),
        "module_classes": sorted(
            {
                module_info["fused_experts_class"]
                for module_info in capture["moe_modules"]
                if module_info["fused_experts_class"] is not None
            }
        ),
    }


def install_vllm_mamba_reference_conv_patch(model):
    """Route simple vLLM Mamba2 prefill causal-conv through reference math."""
    from vllm.model_executor.layers.mamba import mamba_mixer2

    original_conv = getattr(
        mamba_mixer2,
        "_debug_original_causal_conv1d_fn",
        mamba_mixer2.causal_conv1d_fn,
    )
    mamba_mixer2._debug_original_causal_conv1d_fn = original_conv

    def fallback(args, kwargs):
        return original_conv(*args, **kwargs)

    @wraps(original_conv)
    def reference_causal_conv1d_fn(*args, **kwargs):
        x = _call_arg(args, kwargs, 0, "x")
        weight = _call_arg(args, kwargs, 1, "weight")
        bias = _call_arg(args, kwargs, 2, "bias")
        query_start_loc = _call_arg(args, kwargs, 4, "query_start_loc")
        has_initial_state = _call_arg(args, kwargs, 6, "has_initial_state")
        activation = kwargs.get("activation", "silu")

        unsupported_apc = any(
            kwargs.get(name) is not None
            for name in (
                "block_idx_first_scheduled_token",
                "block_idx_last_scheduled_token",
                "initial_state_idx",
                "num_computed_tokens",
            )
        )
        if (
            not isinstance(x, torch.Tensor)
            or x.dim() != 2
            or not isinstance(weight, torch.Tensor)
            or bias is None
            or query_start_loc is None
            or unsupported_apc
        ):
            return fallback(args, kwargs)
        if isinstance(has_initial_state, torch.Tensor) and bool(
            has_initial_state.any()
        ):
            return fallback(args, kwargs)
        if activation not in (None, False, "silu", "swish", True):
            return fallback(args, kwargs)

        original_dtype = x.dtype
        x_token_major = x.transpose(0, 1).contiguous().float()
        weight_f = weight.float()
        bias_f = bias.float()
        width = weight_f.shape[1]
        out = torch.empty(
            x_token_major.shape,
            dtype=original_dtype,
            device=x_token_major.device,
        )

        for seq_idx in range(query_start_loc.numel() - 1):
            start = int(query_start_loc[seq_idx].item())
            end = int(query_start_loc[seq_idx + 1].item())
            for pos in range(end - start):
                acc = bias_f.clone()
                for tap_idx in range(width):
                    src_pos = pos - (width - 1) + tap_idx
                    if src_pos >= 0:
                        acc = (
                            acc + x_token_major[start + src_pos] * weight_f[:, tap_idx]
                        )
                if activation in ("silu", "swish", True):
                    acc = torch.nn.functional.silu(acc)
                out[start + pos] = acc.to(original_dtype)

        return out.transpose(0, 1).contiguous()

    mamba_mixer2.causal_conv1d_fn = reference_causal_conv1d_fn
    return {"patched": True}


def install_vllm_mamba_megatron_gated_rmsnorm_patch(model, target_layers=None):
    """Use Megatron/mamba-ssm rounding for vLLM Mamba2 gated RMSNorm."""
    from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated

    original_forward_cuda = getattr(
        Mixer2RMSNormGated,
        "_debug_original_forward_cuda",
        Mixer2RMSNormGated.forward_cuda,
    )
    Mixer2RMSNormGated._debug_original_forward_cuda = original_forward_cuda

    def patched_forward_cuda(self, x: torch.Tensor, gate: torch.Tensor):
        if (
            not self.use_rms_norm
            or self.tp_size != 1
            or x.shape[-1] != self.per_rank_hidden_size
            or x.shape[-1] % self.group_size != 0
        ):
            return original_forward_cuda(self, x, gate)

        input_dtype = x.dtype
        gated = x.float() * torch.nn.functional.silu(gate.float())
        *prefix_dims, hidden_dim = gated.shape
        group_count = hidden_dim // self.group_size
        grouped = gated.reshape(*prefix_dims, group_count, self.group_size)
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normed = grouped * torch.rsqrt(variance + self.variance_epsilon)
        normed = normed.reshape(*prefix_dims, hidden_dim)
        return (normed * self.weight.data.float()).to(input_dtype)

    Mixer2RMSNormGated.forward_cuda = patched_forward_cuda

    selected_module_ids = None
    selected_module_names = {}
    missing_layers = []
    if target_layers is not None:
        target_layer_set = set(int(layer_idx) for layer_idx in target_layers)
        layers = find_decoder_layers(model)
        invalid_layers = sorted(
            layer_idx
            for layer_idx in target_layer_set
            if layer_idx < 0 or layer_idx >= len(layers)
        )
        if invalid_layers:
            raise ValueError(
                f"Mamba gated RMSNorm target layers out of range for "
                f"{len(layers)} layers: {invalid_layers}"
            )

        selected_module_ids = set()
        for layer_idx in sorted(target_layer_set):
            found = False
            for module_name, module in layers[layer_idx].named_modules():
                if isinstance(module, Mixer2RMSNormGated):
                    selected_module_ids.add(id(module))
                    selected_module_names[id(module)] = (
                        f"layer{layer_idx}.{module_name or '<layer>'}"
                    )
                    found = True
            if not found:
                missing_layers.append(layer_idx)

    patched_modules = 0
    for module in model.modules():
        if isinstance(module, Mixer2RMSNormGated):
            if (
                selected_module_ids is not None
                and id(module) not in selected_module_ids
            ):
                continue
            patched_method = patched_forward_cuda.__get__(module, module.__class__)
            module.forward_cuda = patched_method
            module._forward_method = patched_method
            patched_modules += 1

    return {
        "patched": True,
        "patched_modules": patched_modules,
        "target_layers": None if target_layers is None else sorted(target_layer_set),
        "missing_layers": missing_layers,
        "patched_module_names": [
            selected_module_names[module_id]
            for module_id in sorted(selected_module_names)
            if selected_module_ids is not None and module_id in selected_module_ids
        ],
    }


def install_vllm_moe_router_weight_scaling_patch(model):
    """Move NemotronH routed MoE scale from output-side to router weights."""
    patched = []
    for module_name, module in model.named_modules():
        runner = getattr(module, "runner", None)
        router = getattr(module, "router", None)
        scale = getattr(runner, "routed_scaling_factor", 1.0)
        if runner is None or router is None or scale == 1.0:
            continue
        if not hasattr(router, "routed_scaling_factor"):
            continue

        router.routed_scaling_factor = scale
        runner.routed_scaling_factor = 1.0
        if hasattr(module, "routed_scaling_factor"):
            module.routed_scaling_factor = scale
        patched.append(
            {
                "module": module_name,
                "scale": float(scale),
                "router_class": router.__class__.__name__,
            }
        )

    return {"patched": bool(patched), "patched_modules": patched}


def _vllm_moe_flat_output_view(C):
    if C.dim() == 2:
        return C
    if C.dim() != 3:
        return None
    if C.stride(0) != C.shape[1] * C.stride(1):
        return None
    return C.as_strided(
        (C.shape[0] * C.shape[1], C.shape[2]),
        (C.stride(1), C.stride(2)),
    )


def _get_vllm_moe_fused_experts(module):
    runner = getattr(module, "runner", None)
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None and runner is not None:
        quant_method = getattr(runner, "quant_method", None)
    moe_kernel = getattr(quant_method, "moe_kernel", None)
    if moe_kernel is not None:
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        if fused_experts is not None:
            return fused_experts
    return getattr(runner, "fused_experts", None)


def _collect_vllm_moe_fused_experts_for_layers(model, target_layers):
    target_layer_set = set(int(layer_idx) for layer_idx in target_layers)
    layers = find_decoder_layers(model)
    invalid_layers = sorted(
        layer_idx
        for layer_idx in target_layer_set
        if layer_idx < 0 or layer_idx >= len(layers)
    )
    if invalid_layers:
        raise ValueError(
            f"MoE reference fc2 target layers out of range for "
            f"{len(layers)} layers: {invalid_layers}"
        )

    selected_expert_ids = set()
    selected_expert_names = {}
    missing_layers = []
    for layer_idx in sorted(target_layer_set):
        found = False
        for module_name, module in layers[layer_idx].named_modules():
            fused_experts = _get_vllm_moe_fused_experts(module)
            if fused_experts is None:
                continue
            found = True
            selected_expert_ids.add(id(fused_experts))
            selected_expert_names[id(fused_experts)] = (
                f"layer{layer_idx}.{module_name or '<layer>'}"
            )
        if not found:
            missing_layers.append(layer_idx)

    return {
        "target_layer_set": target_layer_set,
        "selected_expert_ids": selected_expert_ids,
        "selected_expert_names": selected_expert_names,
        "missing_layers": missing_layers,
    }


def install_vllm_moe_reference_fc2_patch(model, target_layers=None):
    """Use reference BF16 matmul for unquantized routed MoE fc2 calls."""
    from vllm.model_executor.layers.fused_moe import fused_moe

    previous_kernel = fused_moe.invoke_fused_moe_triton_kernel
    previous_apply = fused_moe.TritonExperts.apply
    selected_expert_ids = None
    selected_expert_names = {}
    missing_layers = []
    target_layer_set = None
    if target_layers is not None:
        layer_info = _collect_vllm_moe_fused_experts_for_layers(model, target_layers)
        target_layer_set = layer_info["target_layer_set"]
        selected_expert_ids = layer_info["selected_expert_ids"]
        selected_expert_names = layer_info["selected_expert_names"]
        missing_layers = layer_info["missing_layers"]

    patch_state = {
        "patched_calls": 0,
        "fallback_calls": 0,
        "disabled_calls": 0,
        "enabled_stack": [],
        "target_layers": None if target_layers is None else sorted(target_layer_set),
        "missing_layers": missing_layers,
        "patched_module_names": [
            selected_expert_names[expert_id]
            for expert_id in sorted(selected_expert_names)
            if selected_expert_ids is not None and expert_id in selected_expert_ids
        ],
    }
    model._debug_moe_reference_fc2_patch = patch_state

    def _current_experts_enabled():
        if selected_expert_ids is None:
            return True
        stack = patch_state["enabled_stack"]
        return bool(stack and stack[-1])

    @wraps(previous_apply)
    def wrapped_triton_experts_apply(self, *args, **kwargs):
        if selected_expert_ids is None:
            return previous_apply(self, *args, **kwargs)
        patch_state["enabled_stack"].append(id(self) in selected_expert_ids)
        try:
            return previous_apply(self, *args, **kwargs)
        finally:
            patch_state["enabled_stack"].pop()

    def _is_supported_target(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        mul_routed_weight,
        top_k,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        block_shape,
        B_bias,
    ):
        return (
            mul_routed_weight
            and int(top_k) == 1
            and A_scale is None
            and B_scale is None
            and topk_weights is not None
            and sorted_token_ids is not None
            and expert_ids is not None
            and block_shape is None
            and B_bias is None
            and not use_fp8_w8a8
            and not use_int8_w8a8
            and not use_int8_w8a16
            and not use_int4_w4a16
            and isinstance(A, torch.Tensor)
            and isinstance(B, torch.Tensor)
            and isinstance(C, torch.Tensor)
            and A.dtype == torch.bfloat16
            and B.dtype == torch.bfloat16
            and C.dtype == torch.bfloat16
            and A.dim() == 2
            and B.dim() == 3
            and B.shape[2] == A.shape[1]
        )

    @wraps(previous_kernel)
    def wrapped_invoke_fused_moe_triton_kernel(*args, **kwargs):
        A = _call_arg(args, kwargs, 0, "A")
        B = _call_arg(args, kwargs, 1, "B")
        C = _call_arg(args, kwargs, 2, "C")
        A_scale = _call_arg(args, kwargs, 3, "A_scale")
        B_scale = _call_arg(args, kwargs, 4, "B_scale")
        topk_weights = _call_arg(args, kwargs, 5, "topk_weights")
        sorted_token_ids = _call_arg(args, kwargs, 6, "sorted_token_ids")
        expert_ids = _call_arg(args, kwargs, 7, "expert_ids")
        mul_routed_weight = _call_arg(args, kwargs, 9, "mul_routed_weight")
        top_k = _call_arg(args, kwargs, 10, "top_k")
        config = _call_arg(args, kwargs, 11, "config")
        use_fp8_w8a8 = kwargs.get("use_fp8_w8a8", False)
        use_int8_w8a8 = kwargs.get("use_int8_w8a8", False)
        use_int8_w8a16 = kwargs.get("use_int8_w8a16", False)
        use_int4_w4a16 = kwargs.get("use_int4_w4a16", False)
        block_shape = kwargs.get("block_shape")
        B_bias = kwargs.get("B_bias")

        if not _current_experts_enabled():
            patch_state["disabled_calls"] += 1
            return previous_kernel(*args, **kwargs)

        if not _is_supported_target(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            mul_routed_weight,
            top_k,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
            B_bias,
        ):
            patch_state["fallback_calls"] += 1
            return previous_kernel(*args, **kwargs)

        c_flat = _vllm_moe_flat_output_view(C)
        if c_flat is None:
            patch_state["fallback_calls"] += 1
            return previous_kernel(*args, **kwargs)

        block_size = int(config["BLOCK_SIZE_M"])
        sorted_token_ids_1d = sorted_token_ids.to(torch.long).flatten()
        expert_ids_1d = expert_ids.to(torch.long).flatten()
        route_weights = topk_weights.reshape(-1)
        c_flat.zero_()
        for block_idx, expert_id in enumerate(expert_ids_1d.tolist()):
            if expert_id < 0:
                continue
            start = block_idx * block_size
            end = min(start + block_size, sorted_token_ids_1d.numel())
            route_indices = sorted_token_ids_1d[start:end]
            route_indices = route_indices[
                (route_indices >= 0) & (route_indices < A.shape[0])
            ]
            if route_indices.numel() == 0:
                continue

            weighted_input = (
                A[route_indices].float()
                * route_weights[route_indices].float().unsqueeze(-1)
            ).to(torch.bfloat16)
            output = torch.matmul(weighted_input, B[expert_id].transpose(0, 1))
            c_flat[route_indices] = output.to(C.dtype)

        patch_state["patched_calls"] += 1
        return None

    fused_moe.invoke_fused_moe_triton_kernel = wrapped_invoke_fused_moe_triton_kernel
    fused_moe.TritonExperts.apply = wrapped_triton_experts_apply
    return {
        "patched": True,
        "mode": "bf16_input_weighted_fc2",
        "target_layers": patch_state["target_layers"],
        "missing_layers": missing_layers,
        "patched_module_names": patch_state["patched_module_names"],
    }


def inspect_vllm_moe_reference_fc2_patch_state(model):
    return getattr(model, "_debug_moe_reference_fc2_patch", {})


def save_vllm_mamba_internal_capture_from_env(model):
    path = os.environ["DEBUG_MAMBA_INTERNAL_CAPTURE_PATH"]
    capture = getattr(model, "_debug_mamba_internal_capture", {})
    torch.save(capture, path)
    return {
        "path": path,
        "num_conv_calls": len(capture.get("mamba_conv1d_calls", [])),
        "total_conv_calls": capture.get("mamba_conv1d_total_calls", 0),
        "num_scan_calls": len(capture.get("mamba_scan_calls", [])),
        "total_scan_calls": capture.get("mamba_scan_total_calls", 0),
    }


def save_vllm_moe_internal_capture_from_env(model):
    path = os.environ["DEBUG_MOE_INTERNAL_CAPTURE_PATH"]
    capture = getattr(model, "_debug_moe_internal_capture", {})
    torch.save(capture, path)
    return {
        "path": path,
        "num_kernel_calls": len(capture.get("moe_kernel_calls", [])),
        "total_kernel_calls": capture.get("moe_kernel_total_calls", 0),
        "num_activation_calls": len(capture.get("moe_activation_calls", [])),
        "total_activation_calls": capture.get("moe_activation_total_calls", 0),
        "num_moe_modules": len(capture.get("moe_modules", [])),
    }


def default_output(batch_invariant: bool, mxfp8: bool, model_tag: str = "") -> str:
    parts = ["vllm_capture"]
    if model_tag:
        parts.append(model_tag.lstrip("_"))
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


def _normalise_token_ids_list(value, lengths=None):
    if isinstance(value, torch.Tensor):
        token_ids_rows = value.detach().cpu().to(torch.long)
        if token_ids_rows.dim() == 1:
            token_ids_rows = token_ids_rows.unsqueeze(0)
        if token_ids_rows.dim() != 2:
            raise ValueError(
                f"expected 1D or 2D token id tensor, got {tuple(token_ids_rows.shape)}"
            )
        if lengths is None:
            return [
                [int(token_id) for token_id in row.tolist()] for row in token_ids_rows
            ]
        length_values = [int(length) for length in lengths]
        return [
            [int(token_id) for token_id in token_ids_rows[i, :length].tolist()]
            for i, length in enumerate(length_values)
        ]

    token_ids_list = []
    for row in value:
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().to(torch.long).tolist()
        token_ids_list.append([int(token_id) for token_id in row])
    return token_ids_list


def load_token_ids_from_file(path: str, token_ids_key: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if token_ids_key not in payload:
            raise KeyError(f"{path} does not contain token id key {token_ids_key!r}")
        lengths = payload.get("seq_lens")
        if lengths is None:
            lengths = payload.get("input_lengths")
        token_ids_list = _normalise_token_ids_list(payload[token_ids_key], lengths)
        prompts = payload.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != len(token_ids_list):
            prompts = [f"{token_ids_key}[{i}]" for i in range(len(token_ids_list))]
        else:
            prompts = [str(prompt) for prompt in prompts]
        return prompts, token_ids_list, payload

    token_ids_list = _normalise_token_ids_list(payload)
    prompts = [f"{token_ids_key}[{i}]" for i in range(len(token_ids_list))]
    return prompts, token_ids_list, {"token_ids_list": token_ids_list}


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
    p.add_argument(
        "--token-ids-file",
        default=None,
        help=(
            "Torch .pt file containing token id rows to replay instead of "
            "loading prompts from --dataset."
        ),
    )
    p.add_argument(
        "--token-ids-key",
        default="token_ids_list",
        help="Payload key to read from --token-ids-file (default: token_ids_list).",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--batch-invariant", action="store_true")
    p.add_argument(
        "--dump-engine-args",
        action="store_true",
        help="Print the exact vLLM LLM kwargs and sampling params used.",
    )
    p.add_argument(
        "--engine-args-only",
        action="store_true",
        help="With --dump-engine-args, print args and exit before constructing LLM.",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max_model_len override.",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional vLLM gpu_memory_utilization override.",
    )
    p.add_argument(
        "--load-format",
        default=None,
        help="Optional vLLM load_format override.",
    )
    p.add_argument(
        "--served-model-name",
        default=None,
        help="Optional vLLM served_model_name override.",
    )
    p.add_argument(
        "--skip-tokenizer-init",
        action="store_true",
        help="Pass skip_tokenizer_init=True to vLLM. Only use with prompt_token_ids.",
    )
    p.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="Pass enable_prefix_caching=True to vLLM.",
    )
    p.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Pass enable_chunked_prefill=True to vLLM.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size (default: 1).",
    )
    p.add_argument(
        "--no-enforce-eager",
        action="store_true",
        help="Run vLLM without enforce_eager=True, matching NeMo-RL default.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )
    p.add_argument(
        "--logprobs-mode",
        default=None,
        help="Optional vLLM logprobs_mode override.",
    )
    p.add_argument(
        "--attention-backend",
        default=None,
        help="Optional vLLM attention_backend override.",
    )
    p.add_argument(
        "--moe-backend",
        default=None,
        help="Optional vLLM moe_backend override, e.g. triton.",
    )
    p.add_argument(
        "--enable-log-stats",
        action="store_true",
        help="Pass disable_log_stats=False to vLLM.",
    )
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
        "input/output tensors for every module in --capture-layers.",
    )
    p.add_argument(
        "--capture-layers",
        type=parse_layer_indices,
        default=[0],
        help="Comma-separated layer indices/ranges to fully hook when "
        "--capture-debug-tensors is set, e.g. 0,1,5 or 0-2. Default: 0.",
    )
    p.add_argument(
        "--capture-mamba-internals",
        action="store_true",
        help="Capture vLLM Mamba2 causal-conv and SSD-scan boundary tensors. "
        "Use with NemotronH/Nemotron 3 Nano localization.",
    )
    p.add_argument(
        "--capture-moe-internals",
        action="store_true",
        help="Capture vLLM Triton MoE expert GEMM and activation boundary tensors. "
        "Use with NemotronH/Nemotron 3 Nano localization.",
    )
    p.add_argument(
        "--mamba-reference-conv",
        action="store_true",
        help="Patch simple vLLM Mamba2 prefill causal-conv calls to a PyTorch "
        "reference implementation for parity debugging.",
    )
    p.add_argument(
        "--mamba-megatron-gated-rmsnorm",
        action="store_true",
        help="Patch vLLM Mamba2 grouped gated RMSNorm to use the "
        "Megatron/mamba-ssm FP32 weight-multiply rounding point.",
    )
    p.add_argument(
        "--mamba-megatron-gated-rmsnorm-layers",
        type=parse_layer_indices,
        default=None,
        help="Optional comma-separated layer indices/ranges limiting "
        "--mamba-megatron-gated-rmsnorm to selected Mamba layers. "
        "Providing this option implies --mamba-megatron-gated-rmsnorm.",
    )
    p.add_argument(
        "--moe-router-weight-scaling",
        action="store_true",
        help="Patch vLLM NemotronH MoE modules to apply routed_scaling_factor "
        "inside router top-k weights instead of after the fused routed output. "
        "This matches Megatron's routed-probability scaling point for BF16 "
        "diagnostics.",
    )
    p.add_argument(
        "--moe-reference-fc2",
        action="store_true",
        help="Patch unquantized vLLM Triton MoE second-GEMM routed fc2 calls "
        "to a reference BF16 per-expert matmul with route weights applied to "
        "the input, matching the Megatron/PyTorch rounding point diagnosed for "
        "NemotronH BF16.",
    )
    p.add_argument(
        "--moe-reference-fc2-layers",
        type=parse_layer_indices,
        default=None,
        help="Optional comma-separated layer indices/ranges limiting "
        "--moe-reference-fc2 to selected MoE layers. Providing this option "
        "implies --moe-reference-fc2.",
    )
    p.add_argument(
        "--mamba-internal-capture-max-calls",
        type=int,
        default=1,
        help="Number of Mamba conv/scan calls to save; negative means all "
        "calls. Default: 1.",
    )
    p.add_argument(
        "--moe-internal-capture-max-kernel-calls",
        type=int,
        default=2,
        help="Number of vLLM MoE Triton kernel calls and activation calls to "
        "save; negative means all calls. Default: 2, enough for layer-1 fc1 "
        "and fc2 in a NemotronH prefill after engine initialization.",
    )
    p.add_argument(
        "--moe-internal-capture-layers",
        type=parse_layer_indices,
        default=None,
        help="Optional comma-separated decoder layer indices/ranges limiting "
        "--capture-moe-internals to selected MoE layers.",
    )
    p.add_argument(
        "--dump-layernorm-impl",
        action="store_true",
        help="Print the runtime vLLM layernorm implementation after model setup.",
    )
    args = p.parse_args()
    args.model = resolve_model_ref(args.model)
    if args.tokenizer is None:
        args.tokenizer = args.model
    else:
        args.tokenizer = resolve_model_ref(args.tokenizer)
    args.trust_remote_code = (
        args.trust_remote_code
        or is_nemotron3_nano_ref(args.model)
        or is_nemotron3_nano_ref(args.tokenizer)
    )
    if args.output is None:
        args.output = default_output(
            args.batch_invariant,
            args.mxfp8,
            model_output_tag(args.model, args.tokenizer),
        )
    if args.mamba_megatron_gated_rmsnorm_layers is not None:
        args.mamba_megatron_gated_rmsnorm = True
    if args.moe_reference_fc2_layers is not None:
        args.moe_reference_fc2 = True
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
    if args.token_ids_file:
        print(f"[vllm] token ids: {args.token_ids_file} key={args.token_ids_key!r}")
    else:
        print(
            f"[vllm] dataset:   {args.dataset}:{args.dataset_subset}:"
            f"{args.dataset_split} field={args.dataset_field!r} n={args.num_prompts}"
        )

    debug_capture_path = args.output + ".debug_tensors.pt"
    mamba_internal_capture_path = args.output + ".mamba_internals.pt"
    moe_internal_capture_path = args.output + ".moe_internals.pt"
    if args.capture_debug_tensors:
        os.environ["DEBUG_TENSOR_CAPTURE_PATH"] = debug_capture_path
    if args.capture_mamba_internals:
        os.environ["DEBUG_MAMBA_INTERNAL_CAPTURE_PATH"] = mamba_internal_capture_path
    if args.capture_moe_internals:
        os.environ["DEBUG_MOE_INTERNAL_CAPTURE_PATH"] = moe_internal_capture_path

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    token_ids_payload = None
    if args.token_ids_file:
        prompts, token_ids_list, token_ids_payload = load_token_ids_from_file(
            args.token_ids_file,
            args.token_ids_key,
        )
    else:
        prompts = load_prompts(
            args.dataset,
            args.dataset_subset,
            args.dataset_split,
            args.dataset_field,
            args.num_prompts,
            args.dataset_seed,
        )
        token_ids_list = [
            tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts
        ]
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
    llm_kwargs = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "enforce_eager": not args.no_enforce_eager,
        "dtype": "bfloat16",
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_logprobs": vocab_size,
        "seed": 0,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.load_format is not None:
        llm_kwargs["load_format"] = args.load_format
    if args.served_model_name is not None:
        llm_kwargs["served_model_name"] = args.served_model_name
    if args.skip_tokenizer_init:
        llm_kwargs["skip_tokenizer_init"] = True
    if args.trust_remote_code:
        llm_kwargs["trust_remote_code"] = True
    if args.logprobs_mode is not None:
        llm_kwargs["logprobs_mode"] = args.logprobs_mode
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend
    if args.moe_backend is not None:
        llm_kwargs["moe_backend"] = args.moe_backend
    if args.enable_log_stats:
        llm_kwargs["disable_log_stats"] = False

    if args.dump_engine_args:
        print(
            "[vllm-engine-args] standalone_llm\n"
            f"{pprint.pformat(llm_kwargs, sort_dicts=True)}",
            flush=True,
        )

    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "prompt_logprobs": None,
        "logprobs": vocab_size,
        "seed": 0,
    }
    if args.dump_engine_args:
        print(
            "[vllm-sampling-args] standalone_generate\n"
            f"{pprint.pformat(sampling_kwargs, sort_dicts=True)}",
            flush=True,
        )
    if args.engine_args_only:
        return

    llm = LLM(**llm_kwargs)

    if args.dump_layernorm_impl:
        before_info = unwrap_apply_model_result(
            llm.apply_model(inspect_vllm_layernorm_impl)
        )
        print(
            "[vllm-layernorm-impl] standalone_before_patch\n"
            f"{pprint.pformat(before_info, sort_dicts=True)}",
            flush=True,
        )

    if args.batch_invariant:
        # Correctness fix: route the fused-add RMSNorm through the BI Triton
        # kernel; otherwise the cub::BlockReduce/rsqrtf path runs and drifts.
        llm.apply_model(install_rmsnorm_bi_residual_patch)

    if args.dump_layernorm_impl:
        after_info = unwrap_apply_model_result(
            llm.apply_model(inspect_vllm_layernorm_impl)
        )
        print(
            "[vllm-layernorm-impl] standalone_after_patch\n"
            f"{pprint.pformat(after_info, sort_dicts=True)}",
            flush=True,
        )

    if args.mxfp8_bi_emulation:
        if not (args.mxfp8 and args.batch_invariant):
            raise SystemExit(
                "--mxfp8-bi-emulation requires --mxfp8 and --batch-invariant"
            )
        llm.apply_model(install_mxfp8_bi_emulation_patch)

    if args.mamba_reference_conv:
        reference_conv_info = unwrap_apply_model_result(
            llm.apply_model(install_vllm_mamba_reference_conv_patch)
        )
        print(f"[vllm] Mamba reference conv installed: {reference_conv_info}")

    if args.mamba_megatron_gated_rmsnorm:
        gated_norm_info = unwrap_apply_model_result(
            llm.apply_model(
                VllmMambaMegatronGatedRMSNormInstaller(
                    args.mamba_megatron_gated_rmsnorm_layers
                )
            )
        )
        print(f"[vllm] Mamba Megatron gated RMSNorm installed: {gated_norm_info}")

    if args.moe_router_weight_scaling:
        moe_scale_info = unwrap_apply_model_result(
            llm.apply_model(VllmMoERouterWeightScalingInstaller())
        )
        print(f"[vllm] MoE router weight scaling installed: {moe_scale_info}")

    if args.moe_reference_fc2:
        moe_reference_fc2_info = unwrap_apply_model_result(
            llm.apply_model(VllmMoEReferenceFc2Installer(args.moe_reference_fc2_layers))
        )
        print(f"[vllm] MoE reference fc2 installed: {moe_reference_fc2_info}")

    if args.capture_mamba_internals:
        mamba_capture_info = unwrap_apply_model_result(
            llm.apply_model(
                VllmMambaInternalCaptureInstaller(args.mamba_internal_capture_max_calls)
            )
        )
        print(f"[vllm] Mamba internal capture installed: {mamba_capture_info}")

    if args.capture_moe_internals:
        moe_capture_info = unwrap_apply_model_result(
            llm.apply_model(
                VllmMoEInternalCaptureInstaller(
                    args.moe_internal_capture_max_kernel_calls,
                    args.moe_internal_capture_layers,
                )
            )
        )
        print(f"[vllm] MoE internal capture installed: {moe_capture_info}")

    if args.capture_debug_tensors:
        hook_info = unwrap_apply_model_result(
            llm.apply_model(DebugTensorHookInstaller(args.capture_layers))
        )
        print(f"[vllm] debug tensor hooks installed: {hook_info}")

    sampling_params = SamplingParams(**sampling_kwargs)
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids_list],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print(
        f"[vllm] generated {sum(len(o.outputs[0].token_ids) for o in outputs)} "
        f"new tokens across {len(outputs)} prompts"
    )
    if args.moe_reference_fc2:
        moe_reference_fc2_state = unwrap_apply_model_result(
            llm.apply_model(inspect_vllm_moe_reference_fc2_patch_state)
        )
        print(f"[vllm] MoE reference fc2 state: {moe_reference_fc2_state}")

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
        "model": args.model,
        "tokenizer": args.tokenizer,
        "model_family": "nemotron3_nano"
        if is_nemotron3_nano_ref(args.model) or is_nemotron3_nano_ref(args.tokenizer)
        else "llama",
        "prompts": prompts,
        "token_ids_list": token_ids_list,
        "seq_lens": seq_lens,
        "next_token_logprobs": next_token_logprobs,
    }
    if token_ids_payload is not None:
        payload["token_ids_source_file"] = args.token_ids_file
        payload["token_ids_source_key"] = args.token_ids_key
        for key in (
            "offline_target_token_ids",
            "offline_generation_logprobs",
            "offline_metadata",
            "sample_rows",
            "sample_prompt_token_ids_list",
            "sample_response_token_ids_list",
            "sample_full_token_ids_list",
        ):
            if key in token_ids_payload:
                payload[key] = token_ids_payload[key]
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
            f"module_layers={debug_capture.get('captured_module_layers')} "
            f"num_layers={num_layers}"
        )
    if args.capture_mamba_internals:
        mamba_save_info = unwrap_apply_model_result(
            llm.apply_model(save_vllm_mamba_internal_capture_from_env)
        )
        print(f"[vllm] worker saved Mamba internals: {mamba_save_info}")
        mamba_capture = torch.load(
            mamba_internal_capture_path, map_location="cpu", weights_only=False
        )
        payload.update(mamba_capture)
    if args.capture_moe_internals:
        moe_save_info = unwrap_apply_model_result(
            llm.apply_model(save_vllm_moe_internal_capture_from_env)
        )
        print(f"[vllm] worker saved MoE internals: {moe_save_info}")
        moe_capture = torch.load(
            moe_internal_capture_path, map_location="cpu", weights_only=False
        )
        payload.update(moe_capture)
    torch.save(payload, args.output)
    print(f"[vllm] saved capture to {args.output}")


if __name__ == "__main__":
    main()
