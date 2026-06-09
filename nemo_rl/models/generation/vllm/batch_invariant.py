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

from typing import Any

import torch

from nemo_rl.models.true_on_policy import get_mxfp8_matmul_bi_backend

G_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_rmsnorm_patch"
G_ORIGINAL_FORWARD_ATTR = "_nemo_rl_original_forward_cuda"
G_MEGATRON_ROPE_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_rope_patch"
G_MEGATRON_SWIGLU_PATCH_MARKER_ATTR = "_nemo_rl_megatron_style_swiglu_patch"
G_MEGATRON_ROPE_CACHE_ATTR = "_nemo_rl_megatron_style_cos_sin_cache"
G_MXFP8_QDQ_PATCH_MARKER_ATTR = "_nemo_rl_mxfp8_bi_qdq_patch"
G_MXFP8_NATIVE_PATCH_MARKER_ATTR = "_nemo_rl_mxfp8_bi_native_patch"
G_ORIGINAL_MXFP8_MM_ATTR = "_nemo_rl_original_mm_mxfp8"


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

    cos_sin_cache = _get_megatron_style_cos_sin_cache(module, tensor.device)
    cos_sin = cos_sin_cache.index_select(0, positions.to(device=tensor.device))
    cos_half, sin_half = cos_sin.chunk(2, dim=-1)

    rotary_interleaved = not module.is_neox_style
    cos = _megatron_style_duplicate_freqs(
        cos_half,
        rotary_interleaved=rotary_interleaved,
    ).to(tensor_rot.dtype)
    sin = _megatron_style_duplicate_freqs(
        sin_half,
        rotary_interleaved=rotary_interleaved,
    ).to(tensor_rot.dtype)
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
    """Route vLLM ``SiluAndMul`` through Megatron's fused SwiGLU function."""
    from megatron.core.fusions.fused_bias_swiglu import swiglu
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
            return swiglu(x)

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


def install_mxfp8_bi_matmul_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM MXFP8 GEMM through native block-scaled BI matmul."""
    del model  # The patch is module-level inside the vLLM worker process.

    import vllm.utils.flashinfer as vllm_flashinfer
    from vllm.model_executor.layers.batch_invariant import mxfp8_matmul_persistent

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
        results["megatron_style_rmsnorm"] = install_megatron_style_rmsnorm_patch(model)
        results["megatron_style_rope"] = install_megatron_style_rope_patch(model)
        results["megatron_style_swiglu"] = install_megatron_style_swiglu_patch(model)

    if mxfp8_matmul_batch_invariant:
        backend = get_mxfp8_matmul_bi_backend()
        results["mxfp8_matmul_backend"] = backend
        if backend == "qdq":
            results["mxfp8_matmul"] = install_mxfp8_bi_emulation_patch(model)
        else:
            results["mxfp8_matmul"] = install_mxfp8_bi_matmul_patch(model)

    return results
