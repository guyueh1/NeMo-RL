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

"""vLLM true-on-policy runtime patches used by generation workers."""

from __future__ import annotations

import math
import os
from typing import Any

import torch

from nemo_rl.models.true_on_policy import (
    get_mxfp8_matmul_bi_backend,
    install_te_cublas_workspace_limit_from_env,
)

G_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_rmsnorm_patch"
G_ORIGINAL_FORWARD_ATTR = "_nemo_rl_original_forward_cuda"
G_MEGATRON_ROPE_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_rope_patch"
G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_swiglu_patch"
G_MEGATRON_ROPE_CACHE_ATTR = "_nemo_rl_megatron_style_cos_sin_cache"
G_MXFP8_QDQ_PATCH_MARKER_ATTR = "_nemo_rl_mxfp8_bi_qdq_patch"
G_MXFP8_NATIVE_PATCH_MARKER_ATTR = "_nemo_rl_mxfp8_bi_native_patch"
G_MXFP8_CUBLAS_PATCH_MARKER_ATTR = "_nemo_rl_mxfp8_bi_cublas_patch"
G_ORIGINAL_MXFP8_MM_ATTR = "_nemo_rl_original_mm_mxfp8"
G_TRUE_ON_POLICY_COMPONENTS_ENV = "NEMO_RL_VLLM_TRUE_ON_POLICY_PATCH_COMPONENTS"
G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS = ("rmsnorm", "rope", "swiglu")
G_TRUE_ON_POLICY_BF16_PATCH_COMPONENT_SET = frozenset(
    G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS
)


def _rebind_custom_op_forward_methods(
    model: torch.nn.Module,
    target_cls: type,
) -> int:
    rebound_count = 0
    for module in model.modules():
        if isinstance(module, target_cls):
            # CustomOp binds _forward_method at construction, so a class-level
            # patch needs to be rebound onto existing module instances.
            module._forward_method = module.forward_cuda
            rebound_count += 1
    return rebound_count


def install_megatron_style_rmsnorm_patch(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Route vLLM RMSNorm through Megatron's BI RMSNorm implementation.

    Replaces vLLM ``RMSNorm.forward_cuda``. The no-residual path normally calls
    vLLM ``rms_norm`` and the residual path normally calls
    ``fused_add_rms_norm``. Under true-on-policy mode both are routed through
    Megatron-Core's ``BatchInvariantRMSNormFn`` after the same residual add that
    vLLM performs.
    """
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        BatchInvariantRMSNormFn,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm

    current_forward = RMSNorm.forward_cuda
    original_forward = getattr(
        current_forward,
        G_ORIGINAL_FORWARD_ATTR,
        current_forward,
    )
    already_installed = bool(getattr(current_forward, G_PATCH_MARKER_ATTR, False))

    if not already_installed:

        def patched_forward_cuda(self, x, residual=None):
            if self.variance_size_override is not None or not getattr(
                self, "has_weight", True
            ):
                return original_forward(self, x, residual)

            if residual is not None:
                residual_out = x + residual
                return (
                    BatchInvariantRMSNormFn.apply(
                        residual_out,
                        self.weight.data,
                        self.variance_epsilon,
                        False,
                    ),
                    residual_out,
                )

            return BatchInvariantRMSNormFn.apply(
                x,
                self.weight.data,
                self.variance_epsilon,
                False,
            )

        setattr(patched_forward_cuda, G_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        RMSNorm.forward_cuda = patched_forward_cuda

    rebound_count = _rebind_custom_op_forward_methods(model, RMSNorm)

    return {
        "already_installed": already_installed,
        "rebound_count": rebound_count,
    }


def install_batch_invariant_rmsnorm_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Backward-compatible wrapper for the Megatron-style RMSNorm patch."""
    return install_megatron_style_rmsnorm_patch(model)


def _megatron_style_rotate_half(
    x: torch.Tensor,
    *,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.flatten(-2)


def _megatron_style_duplicate_freqs(
    values: torch.Tensor,
    *,
    rotary_interleaved: bool,
) -> torch.Tensor:
    if not rotary_interleaved:
        return torch.cat((values, values), dim=-1)
    return torch.stack((values, values), dim=-1).flatten(-2)


def _get_megatron_style_cos_sin_cache(module, device: torch.device) -> torch.Tensor:
    cache = getattr(module, G_MEGATRON_ROPE_CACHE_ATTR, None)
    if cache is None or cache.device != device:
        cache = module._compute_cos_sin_cache().to(device=device)
        setattr(module, G_MEGATRON_ROPE_CACHE_ATTR, cache)
    return cache


def _get_megatron_style_inv_freq_from_module_attrs(
    module,
    device: torch.device,
) -> torch.Tensor | None:
    base = getattr(module, "base", None)
    rotary_dim = getattr(module, "rotary_dim", None)
    if base is None or rotary_dim is None:
        return None

    inv_freq = 1.0 / (
        float(base)
        ** (
            torch.arange(0, int(rotary_dim), 2, dtype=torch.float32, device=device)
            / int(rotary_dim)
        )
    )

    class_name = type(module).__name__
    is_llama3_rope = class_name == "Llama3RotaryEmbedding" or all(
        hasattr(module, attr)
        for attr in (
            "scaling_factor",
            "low_freq_factor",
            "high_freq_factor",
            "old_context_len",
        )
    )
    if not is_llama3_rope:
        return inv_freq

    factor = float(getattr(module, "scaling_factor", 8.0))
    low_freq_factor = float(getattr(module, "low_freq_factor", 1.0))
    high_freq_factor = float(getattr(module, "high_freq_factor", 4.0))
    old_context_len = float(getattr(module, "old_context_len", 8192.0))

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled_inv_freq = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * scaled_inv_freq / factor + smooth_factor * scaled_inv_freq
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, scaled_inv_freq)


def _get_megatron_style_inv_freq(module, device: torch.device) -> torch.Tensor | None:
    inv_freq = _get_megatron_style_inv_freq_from_module_attrs(module, device)
    if inv_freq is not None:
        return inv_freq

    compute_inv_freq = getattr(module, "_compute_inv_freq", None)
    base = getattr(module, "base", None)
    inv_freq = None
    if callable(compute_inv_freq) and base is not None:
        try:
            inv_freq = compute_inv_freq(base)
        except TypeError:
            inv_freq = compute_inv_freq()

    if inv_freq is None:
        inv_freq = getattr(module, "inv_freq", None)
    if inv_freq is None:
        return None
    return inv_freq.to(device=device)


def _get_megatron_style_freqs_half(
    *,
    module,
    positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    inv_freq = _get_megatron_style_inv_freq(module, device)
    if inv_freq is None:
        return None

    positions = positions.to(device=device, dtype=inv_freq.dtype)
    class_name = type(module).__name__
    if class_name == "LinearScalingRotaryEmbedding" and hasattr(
        module, "scaling_factor"
    ):
        positions = positions / module.scaling_factor

    return torch.outer(positions, inv_freq)


def _apply_megatron_style_rope(
    *,
    module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    original_shape = tensor.shape
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    tensor = tensor.view(num_tokens, -1, module.head_size)
    tensor_rot = tensor[..., : module.rotary_dim]
    tensor_pass = tensor[..., module.rotary_dim :]

    rotary_interleaved = not module.is_neox_style
    freqs_half = _get_megatron_style_freqs_half(
        module=module,
        positions=positions,
        device=tensor.device,
    )
    if freqs_half is None:
        cos_sin_cache = _get_megatron_style_cos_sin_cache(module, tensor.device)
        cos_sin = cos_sin_cache.index_select(0, positions.to(device=tensor.device))
        cos_half, sin_half = cos_sin.chunk(2, dim=-1)
        cos = _megatron_style_duplicate_freqs(
            cos_half,
            rotary_interleaved=rotary_interleaved,
        ).to(tensor_rot.dtype)
        sin = _megatron_style_duplicate_freqs(
            sin_half,
            rotary_interleaved=rotary_interleaved,
        ).to(tensor_rot.dtype)
    else:
        freqs = _megatron_style_duplicate_freqs(
            freqs_half,
            rotary_interleaved=rotary_interleaved,
        )
        cos = torch.cos(freqs).to(tensor_rot.dtype)
        sin = torch.sin(freqs).to(tensor_rot.dtype)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    tensor_rot = (tensor_rot * cos) + (
        _megatron_style_rotate_half(
            tensor_rot,
            rotary_interleaved=rotary_interleaved,
        )
        * sin
    )
    return torch.cat((tensor_rot, tensor_pass), dim=-1).reshape(original_shape)


def install_megatron_style_rope_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM base RoPE through Megatron's unfused RoPE formula.

    Replaces vLLM ``RotaryEmbedding.forward_cuda``, which normally dispatches
    to vLLM's in-place rotary custom op using ``cos_sin_cache``. The patch uses
    Megatron's explicit formula: duplicate half-dimension frequencies, cast
    cos/sin to the activation dtype, rotate half the activation, and combine
    with PyTorch elementwise ops.
    """
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    current_forward = RotaryEmbedding.forward_cuda
    original_forward = getattr(
        current_forward,
        G_ORIGINAL_FORWARD_ATTR,
        current_forward,
    )
    already_installed = bool(
        getattr(current_forward, G_MEGATRON_ROPE_PATCH_MARKER_ATTR, False)
    )

    if not already_installed:

        def patched_forward_cuda(self, positions, query, key=None):
            if self.use_flashinfer:
                return original_forward(self, positions, query, key)

            query = _apply_megatron_style_rope(
                module=self,
                positions=positions,
                tensor=query,
            )
            if key is not None:
                key = _apply_megatron_style_rope(
                    module=self,
                    positions=positions,
                    tensor=key,
                )
            return query, key

        setattr(patched_forward_cuda, G_MEGATRON_ROPE_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        RotaryEmbedding.forward_cuda = patched_forward_cuda

    rebound_count = _rebind_custom_op_forward_methods(model, RotaryEmbedding)

    return {
        "already_installed": already_installed,
        "rebound_count": rebound_count,
    }


def install_megatron_style_swiglu_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM ``SiluAndMul`` through Megatron's unfused SwiGLU path."""
    import torch.nn.functional as F
    from vllm.model_executor.layers.activation import SiluAndMul

    current_forward = SiluAndMul.forward_cuda
    original_forward = getattr(
        current_forward,
        G_ORIGINAL_FORWARD_ATTR,
        current_forward,
    )
    already_installed = bool(
        getattr(current_forward, G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR, False)
    )

    if not already_installed:

        def patched_forward_cuda(self, x):
            x_glu, x_linear = torch.chunk(x, 2, dim=-1)
            return F.silu(x_glu) * x_linear

        setattr(patched_forward_cuda, G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        SiluAndMul.forward_cuda = patched_forward_cuda

    rebound_count = _rebind_custom_op_forward_methods(model, SiluAndMul)

    return {
        "already_installed": already_installed,
        "rebound_count": rebound_count,
    }


def install_mxfp8_bi_emulation_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM MXFP8 GEMM through dequant + BF16 BI matmul.

    vLLM's MXFP8 dense linear path normally calls FlashInfer/CUTLASS
    ``mm_mxfp8``. Megatron's matching path dequants TE MXFP8 operands to BF16
    and then calls the BF16 batch-invariant persistent matmul. This patch makes
    vLLM generation use the same operation when
    ``policy.mxfp8_matmul_batch_invariant=true`` and
    ``NEMO_RL_MXFP8_MATMUL_BI_BACKEND=qdq``.
    """
    del model  # The patch is module-level inside the vLLM worker process.

    import vllm.utils.flashinfer as vllm_flashinfer
    from vllm.model_executor.layers.batch_invariant import matmul_persistent
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        dequant_mxfp8_to_bf16,
    )

    current_mm_mxfp8 = vllm_flashinfer.mm_mxfp8
    already_installed = bool(
        getattr(current_mm_mxfp8, G_MXFP8_QDQ_PATCH_MARKER_ATTR, False)
    )
    original_mm_mxfp8 = getattr(
        current_mm_mxfp8,
        G_ORIGINAL_MXFP8_MM_ATTR,
        current_mm_mxfp8,
    )

    if not already_installed:

        def _unswizzle_mxfp8_scale(
            scale_1d: torch.Tensor,
            *,
            m: int,
            k: int,
        ) -> torch.Tensor:
            factor = MXFP8_BLOCK_SIZE * 4
            num_m_tiles = (m + 127) // 128
            num_k_tiles = (k + factor - 1) // factor
            scale_cols = k // MXFP8_BLOCK_SIZE
            scale_5d = scale_1d.view(num_m_tiles, num_k_tiles, 32, 4, 4)
            scale_unswizzled = scale_5d.transpose(1, 3).contiguous()
            scale_padded = scale_unswizzled.view(num_m_tiles * 128, num_k_tiles * 4)
            return scale_padded[:m, :scale_cols].contiguous()

        def _bi_mm_mxfp8(
            a: torch.Tensor,
            b: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
            out_dtype: torch.dtype,
            backend: str = "cutlass",  # noqa: ARG001 - preserves vLLM API.
        ) -> torch.Tensor:
            # a: [M, K] activation. b: [K, N] transposed [N, K] weight.
            m, k = a.shape
            n = b.shape[1]

            a_scale_2d = _unswizzle_mxfp8_scale(a_scale, m=m, k=k)
            b_scale_2d = _unswizzle_mxfp8_scale(b_scale, m=n, k=k)

            a_bf16 = dequant_mxfp8_to_bf16(a, a_scale_2d)
            weight_bf16 = dequant_mxfp8_to_bf16(b.t().contiguous(), b_scale_2d)
            return matmul_persistent(a_bf16, weight_bf16.t()).to(out_dtype)

        setattr(_bi_mm_mxfp8, G_MXFP8_QDQ_PATCH_MARKER_ATTR, True)
        setattr(_bi_mm_mxfp8, G_ORIGINAL_MXFP8_MM_ATTR, original_mm_mxfp8)
        vllm_flashinfer.mm_mxfp8 = _bi_mm_mxfp8

    return {
        "already_installed": already_installed,
        "patched": True,
    }


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _swizzle_mxfp8_scale(scale_2d: torch.Tensor, *, m: int, k: int) -> torch.Tensor:
    """Convert compact [M, K / 32] E8M0 scales to the cuBLAS swizzled layout."""
    block_size = 32
    scale_cols = k // block_size
    num_m_tiles = _ceil_div(m, 128)
    num_k_tiles = _ceil_div(scale_cols, 4)
    padded = torch.zeros(
        (num_m_tiles * 128, num_k_tiles * 4),
        dtype=scale_2d.dtype,
        device=scale_2d.device,
    )
    padded[:m, :scale_cols] = scale_2d[:m, :scale_cols]
    return (
        padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def _as_swizzled_mxfp8_scale(scale: torch.Tensor, *, m: int, k: int) -> torch.Tensor:
    scale_cols = k // 32
    swizzled_numel = _ceil_div(m, 128) * _ceil_div(scale_cols, 4) * 512
    if scale.dim() == 1 and scale.numel() == swizzled_numel:
        return scale.contiguous()
    if scale.dim() == 2:
        return _swizzle_mxfp8_scale(scale, m=m, k=k)
    if scale.dim() == 1 and scale.numel() == m * scale_cols:
        return _swizzle_mxfp8_scale(scale.view(m, scale_cols), m=m, k=k)
    raise RuntimeError(
        "Unsupported MXFP8 scale layout for native BI matmul: "
        f"shape={tuple(scale.shape)}, expected swizzled numel={swizzled_numel} "
        f"or compact shape=({m}, {scale_cols})."
    )


def _as_te_mxfp8_data(data: torch.Tensor) -> torch.Tensor:
    if data.dtype == torch.uint8:
        return data.contiguous()
    if data.dtype == torch.float8_e4m3fn:
        return data.contiguous().view(torch.uint8)
    raise RuntimeError(
        f"cuBLAS MXFP8 backend expected uint8 or float8_e4m3fn data, got {data.dtype}."
    )


def _as_te_mxfp8_scale(scale: torch.Tensor, *, m: int, k: int) -> torch.Tensor:
    scale_cols = k // 32
    padded_m = _ceil_div(m, 128) * 128
    padded_scale_cols = _ceil_div(scale_cols, 4) * 4
    expected_numel = padded_m * padded_scale_cols
    if scale.numel() < expected_numel:
        raise RuntimeError(
            "cuBLAS MXFP8 backend expected swizzled scales with at least "
            f"{expected_numel} elements for shape ({m}, {k}); got "
            f"{scale.numel()}."
        )
    return (
        scale.flatten()[:expected_numel]
        .contiguous()
        .view(
            padded_m,
            padded_scale_cols,
        )
    )


def _make_te_mxfp8_tensor(
    data: torch.Tensor,
    scale: torch.Tensor,
    *,
    fake_dtype: torch.dtype,
) -> Any:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import (
        MXFP8Quantizer,
        MXFP8Tensor,
    )

    m, k = data.shape
    fp8_dtype = tex.DType.kFloat8E4M3
    quantizer = MXFP8Quantizer(
        fp8_dtype=fp8_dtype,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True
    return MXFP8Tensor(
        shape=data.shape,
        dtype=fake_dtype,
        rowwise_data=_as_te_mxfp8_data(data),
        rowwise_scale_inv=_as_te_mxfp8_scale(scale, m=m, k=k),
        columnwise_data=None,
        columnwise_scale_inv=None,
        fp8_dtype=fp8_dtype,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=True,
    )


def install_mxfp8_bi_cublas_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM MXFP8 GEMM through Transformer Engine cuBLASLt."""
    del model  # The patch is module-level inside the vLLM worker process.

    import vllm.utils.flashinfer as vllm_flashinfer

    workspace_limit = install_te_cublas_workspace_limit_from_env()
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    current_mm_mxfp8 = vllm_flashinfer.mm_mxfp8
    already_installed = bool(
        getattr(current_mm_mxfp8, G_MXFP8_CUBLAS_PATCH_MARKER_ATTR, False)
    )
    original_mm_mxfp8 = getattr(
        current_mm_mxfp8,
        G_ORIGINAL_MXFP8_MM_ATTR,
        current_mm_mxfp8,
    )

    if not already_installed:

        def _cublas_mm_mxfp8(
            a: torch.Tensor,
            b: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
            out_dtype: torch.dtype,
            backend: str = "cutlass",  # noqa: ARG001 - preserves vLLM API.
        ) -> torch.Tensor:
            # vLLM passes a: [M, K] activation and b: [K, N] transposed weight.
            # TE's MXFP8 scales are attached along each operand's original K
            # dimension, so call TE like Megatron linears do: weight [N, K],
            # activation [M, K], layout="TN".
            _, k = a.shape
            n = b.shape[1]
            activation = _make_te_mxfp8_tensor(
                a,
                a_scale,
                fake_dtype=out_dtype,
            )
            weight = _make_te_mxfp8_tensor(
                b.t().contiguous(),
                b_scale,
                fake_dtype=out_dtype,
            )
            output_t, *_ = general_gemm(
                weight,
                activation,
                out_dtype=out_dtype,
                layout="TN",
            )
            if output_t.shape == (a.shape[0], n):
                return output_t.contiguous()
            if output_t.shape == (n, a.shape[0]):
                return output_t.t().contiguous()
            else:
                raise RuntimeError(
                    "Unexpected cuBLAS MXFP8 output shape from TE general_gemm: "
                    f"got {tuple(output_t.shape)}, expected {(a.shape[0], n)} "
                    f"or {(n, a.shape[0])} for K={k}."
                )

        setattr(_cublas_mm_mxfp8, G_MXFP8_CUBLAS_PATCH_MARKER_ATTR, True)
        setattr(_cublas_mm_mxfp8, G_ORIGINAL_MXFP8_MM_ATTR, original_mm_mxfp8)
        vllm_flashinfer.mm_mxfp8 = _cublas_mm_mxfp8

    return {
        "already_installed": already_installed,
        "patched": True,
        "te_cublas_workspace_limit": workspace_limit,
    }


def install_mxfp8_bi_matmul_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM MXFP8 GEMM through native block-scaled BI matmul."""
    del model  # The patch is module-level inside the vLLM worker process.

    import vllm.utils.flashinfer as vllm_flashinfer

    from nemo_rl.models.mxfp8_bi_matmul import mxfp8_matmul_persistent

    current_mm_mxfp8 = vllm_flashinfer.mm_mxfp8
    already_installed = bool(
        getattr(current_mm_mxfp8, G_MXFP8_NATIVE_PATCH_MARKER_ATTR, False)
    )
    original_mm_mxfp8 = getattr(
        current_mm_mxfp8,
        G_ORIGINAL_MXFP8_MM_ATTR,
        current_mm_mxfp8,
    )

    if not already_installed:

        def _bi_mm_mxfp8(
            a: torch.Tensor,
            b: torch.Tensor,
            a_scale: torch.Tensor,
            b_scale: torch.Tensor,
            out_dtype: torch.dtype,
            backend: str = "cutlass",  # noqa: ARG001 - preserves vLLM API.
        ) -> torch.Tensor:
            # a: [M, K] activation. b: [K, N] transposed [N, K] weight.
            m, k = a.shape
            n = b.shape[1]
            a_scale_swizzled = _as_swizzled_mxfp8_scale(a_scale, m=m, k=k)
            b_scale_swizzled = _as_swizzled_mxfp8_scale(b_scale, m=n, k=k)
            return mxfp8_matmul_persistent(
                (a, a_scale_swizzled),
                (b.t().contiguous(), b_scale_swizzled),
                output_dtype=out_dtype,
            )

        setattr(_bi_mm_mxfp8, G_MXFP8_NATIVE_PATCH_MARKER_ATTR, True)
        setattr(_bi_mm_mxfp8, G_ORIGINAL_MXFP8_MM_ATTR, original_mm_mxfp8)
        vllm_flashinfer.mm_mxfp8 = _bi_mm_mxfp8

    return {
        "already_installed": already_installed,
        "patched": True,
    }


def install_true_on_policy_patch_components(
    model: torch.nn.Module,
    components: tuple[str, ...],
) -> dict[str, Any]:
    """Install selected BF16 true-on-policy vLLM patches for diagnostics."""
    requested_components = set(components)
    unknown_components = (
        requested_components - G_TRUE_ON_POLICY_BF16_PATCH_COMPONENT_SET
    )
    if unknown_components:
        raise ValueError(
            "Unknown vLLM true-on-policy patch components: "
            f"{sorted(unknown_components)}. Expected subset of "
            f"{list(G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS)}."
        )

    results: dict[str, Any] = {}
    if "rmsnorm" in requested_components:
        results["megatron_style_rmsnorm"] = install_megatron_style_rmsnorm_patch(model)
    if "rope" in requested_components:
        results["megatron_style_rope"] = install_megatron_style_rope_patch(model)
    if "swiglu" in requested_components:
        results["megatron_style_swiglu"] = install_megatron_style_swiglu_patch(model)
    return results


def _get_requested_true_on_policy_components() -> tuple[str, ...]:
    raw_components = os.environ.get(G_TRUE_ON_POLICY_COMPONENTS_ENV)
    if raw_components is None:
        return G_TRUE_ON_POLICY_BF16_PATCH_COMPONENTS
    if raw_components.strip() == "":
        return ()
    return tuple(
        component.strip().lower()
        for component in raw_components.split(",")
        if component.strip()
    )


def install_true_on_policy_patches(
    model: torch.nn.Module,
    *,
    bf16_true_on_policy: bool,
    mxfp8_matmul_batch_invariant: bool,
) -> dict[str, Any]:
    """Install vLLM true-on-policy patches controlled by policy-level flags."""
    results: dict[str, Any] = {}

    if mxfp8_matmul_batch_invariant and not bf16_true_on_policy:
        raise ValueError(
            "policy.mxfp8_matmul_batch_invariant=True requires "
            "policy.bf16_true_on_policy=True because that flag enables "
            "VLLM_BATCH_INVARIANT and the BF16 vLLM patches."
        )

    if bf16_true_on_policy:
        components = _get_requested_true_on_policy_components()
        results["bf16_components"] = components
        results.update(
            install_true_on_policy_patch_components(
                model,
                components,
            )
        )

    if mxfp8_matmul_batch_invariant:
        backend = get_mxfp8_matmul_bi_backend()
        results["mxfp8_matmul_backend"] = backend
        if backend == "qdq":
            results["mxfp8_matmul"] = install_mxfp8_bi_emulation_patch(model)
        elif backend == "native":
            results["mxfp8_matmul"] = install_mxfp8_bi_matmul_patch(model)
        elif backend == "cublas":
            results["mxfp8_matmul"] = install_mxfp8_bi_cublas_patch(model)
        else:
            raise AssertionError(f"Unhandled MXFP8 BI matmul backend: {backend}")

    return results
