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
