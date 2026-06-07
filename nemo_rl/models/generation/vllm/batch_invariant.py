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

"""vLLM batch-invariant runtime patches used by generation workers."""

from __future__ import annotations

from typing import Any

import torch

G_PATCH_MARKER_ATTR = "_nemo_rl_batch_invariant_residual_rmsnorm_patch"
G_ORIGINAL_FORWARD_ATTR = "_nemo_rl_original_forward_cuda"
G_MXFP8_PATCH_MARKER_ATTR = "_nemo_rl_mxfp8_bi_emulation_patch"
G_ORIGINAL_MXFP8_MM_ATTR = "_nemo_rl_original_mm_mxfp8"


def install_batch_invariant_rmsnorm_patch(model: torch.nn.Module) -> dict[str, Any]:
    """Route vLLM residual RMSNorm through its batch-invariant Triton kernel.

    vLLM's upstream CUDA path uses ``rms_norm_batch_invariant`` only when the
    RMSNorm has no residual tensor. Decoder post-attention and post-MLP RMSNorms
    use the residual branch, which otherwise calls the fused custom op and
    diverges from Megatron's batch-invariant RMSNorm path. This patch preserves
    vLLM's residual mutation semantics while changing only the normalized output
    computation.
    """
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
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
            if (
                residual is not None
                and self.variance_size_override is None
                and getattr(self, "has_weight", True)
            ):
                residual.add_(x)
                return (
                    rms_norm_batch_invariant(
                        residual,
                        self.weight.data,
                        self.variance_epsilon,
                    ),
                    residual,
                )
            return original_forward(self, x, residual)

        setattr(patched_forward_cuda, G_PATCH_MARKER_ATTR, True)
        setattr(patched_forward_cuda, G_ORIGINAL_FORWARD_ATTR, original_forward)
        RMSNorm.forward_cuda = patched_forward_cuda

    rebound_count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm):
            # CustomOp binds _forward_method at construction, so a class-level
            # patch needs to be rebound onto existing module instances.
            module._forward_method = module.forward_cuda
            rebound_count += 1

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
    ``policy.generation.vllm_cfg.match_vllm_mxfp8_matmul=true``.
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
        getattr(current_mm_mxfp8, G_MXFP8_PATCH_MARKER_ATTR, False)
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

        setattr(_bi_mm_mxfp8, G_MXFP8_PATCH_MARKER_ATTR, True)
        setattr(_bi_mm_mxfp8, G_ORIGINAL_MXFP8_MM_ATTR, original_mm_mxfp8)
        vllm_flashinfer.mm_mxfp8 = _bi_mm_mxfp8

    return {
        "already_installed": already_installed,
        "patched": True,
    }
