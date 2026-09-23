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

from functools import wraps
from typing import Any

import torch

from nemo_rl.models.quantization.mxfp4 import quantize_dequantize_mxfp4


def patch_mcore_language_loss_for_detached_mtp_logits() -> None:
    """Clone detached MTP logits before MCore's in-place fused cross entropy.

    MTP detached heads use ``LinearWithFrozenWeight``, whose custom autograd
    function returns a view. MCore's fused vocabulary cross entropy normalizes
    logits in place, which is forbidden on that view during backward. This is
    unrelated to quantizing the output projection; install the compatibility
    patch before model construction so MTP captures the wrapped loss callback.
    """
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    original = LanguageModule.compute_language_model_loss
    if getattr(original, "_nrl_clones_detached_mtp_logits", False):
        return

    @wraps(original)
    def compute_language_model_loss_with_cloned_logits(
        self: Any,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return original(self, labels, logits.clone())

    compute_language_model_loss_with_cloned_logits._nrl_clones_detached_mtp_logits = (  # type: ignore[attr-defined]
        True
    )
    LanguageModule.compute_language_model_loss = (  # type: ignore[method-assign]
        compute_language_model_loss_with_cloned_logits
    )
    print(
        "Patched MCore language-model loss to clone detached MTP logits before "
        "in-place fused cross entropy."
    )


def _quantize_dequantize_mxfp4_weight_(weight: torch.Tensor) -> None:
    """Round one BF16 or Transformer Engine MXFP8 weight in place."""
    # Keep optional Transformer Engine/Megatron imports off non-Megatron paths.
    from megatron.core.fp8_utils import (
        dequantize_fp8_tensor,
        get_grouped_quantized_members,
        is_float8tensor,
        is_grouped_tensor_with_quantized_storage,
    )

    if is_grouped_tensor_with_quantized_storage(weight):
        members = get_grouped_quantized_members(weight, create_if_missing=True)
    else:
        members = [weight]

    for member in members:
        logical = dequantize_fp8_tensor(member) if is_float8tensor(member) else member
        member.copy_(quantize_dequantize_mxfp4(logical))


class MXFP4MoEWeightHook:
    """One-shot model pre-hook re-armed only at safe phase boundaries."""

    def __init__(
        self,
        model: torch.nn.Module,
        matched_weights: list[tuple[str, torch.nn.Module, tuple[str, ...]]],
    ) -> None:
        self._matched_weights = matched_weights
        self._armed = True
        self._application_count = 0
        self._handle = model.register_forward_pre_hook(self)

    def arm(self) -> None:
        self._armed = True

    def remove(self) -> None:
        self._handle.remove()

    def __call__(
        self,
        _hooked_model: torch.nn.Module,
        _inputs: tuple[Any, ...],
    ) -> None:
        if not self._armed:
            return
        self._armed = False
        with torch.no_grad():
            for module_name, module, weight_names in self._matched_weights:
                for weight_name in weight_names:
                    weight = getattr(module, weight_name, None)
                    if weight is None:
                        raise RuntimeError(
                            "MXFP4 MoE runtime hook lost weight "
                            f"{module_name}.{weight_name}."
                        )
                    _quantize_dequantize_mxfp4_weight_(weight)
        self._application_count += 1
        print(
            "Applied one-shot MXFP4 quantize/dequantize to "
            f"{sum(len(names) for _, _, names in self._matched_weights)} "
            f"routed-expert weights (application {self._application_count})."
        )


def register_mxfp4_moe_weight_hooks(model: torch.nn.Module) -> MXFP4MoEWeightHook:
    """Quantize/dequantize non-MTP routed-expert weights before model forward.

    Megatron parameters can be views into one flat parameter buffer. Quantizing
    individual expert weights immediately before each expert executes therefore
    changes the shared buffer's version after earlier autograd functions have
    saved views from it. Quantize every selected weight from one model-level
    pre-forward hook instead, before autograd can save any parameter views.
    """
    matched_weights: list[tuple[str, torch.nn.Module, tuple[str, ...]]] = []
    for module_name, module in model.named_modules():
        module_path = module_name.split(".")
        if "mtp" in module_path or "experts" not in module_path:
            continue
        if not module_name.endswith((".linear_fc1", ".linear_fc2")):
            continue
        weight_names = tuple(
            name
            for name, _parameter in module.named_parameters(recurse=False)
            if name == "weight" or name.startswith("weight")
        )
        if not weight_names:
            continue
        matched_weights.append((module_name, module, weight_names))

    if not matched_weights:
        raise ValueError(
            "MXFP4 MoE runtime patch did not find routed-expert "
            "linear_fc1/linear_fc2 modules."
        )

    hook = MXFP4MoEWeightHook(model, matched_weights)
    matched_weight_count = sum(
        len(weight_names) for _module_name, _module, weight_names in matched_weights
    )
    print(
        "Registered model-level MXFP4 quantize/dequantize hook on "
        f"{matched_weight_count} routed-expert weights across "
        f"{len(matched_weights)} modules."
    )
    return hook
