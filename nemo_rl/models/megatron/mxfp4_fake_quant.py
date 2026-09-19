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

from typing import Any

import torch

from nemo_rl.models.quantization.mxfp4 import fake_quantize_mxfp4


def _fake_quantize_mxfp4_weight_(weight: torch.Tensor) -> None:
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
        member.copy_(fake_quantize_mxfp4(logical))


def register_mxfp4_moe_weight_hooks(model: torch.nn.Module) -> list[Any]:
    """Register pre-forward fake-quant hooks on routed-expert FC1/FC2 modules."""
    handles = []
    matched_weights = 0
    for module_name, module in model.named_modules():
        if "experts" not in module_name.split("."):
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

        def fake_quant_hook(
            hooked_module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            *,
            hooked_weight_names: tuple[str, ...] = weight_names,
            hooked_module_name: str = module_name,
        ) -> None:
            with torch.no_grad():
                for weight_name in hooked_weight_names:
                    weight = getattr(hooked_module, weight_name, None)
                    if weight is None:
                        raise RuntimeError(
                            "MXFP4 MoE fake-quant hook lost weight "
                            f"{hooked_module_name}.{weight_name}."
                        )
                    _fake_quantize_mxfp4_weight_(weight)

        # Megatron DDP's parameter-gather hook was registered while wrapping the
        # model. Appending ours makes it run after the current MXFP8 parameter is
        # present and immediately before this expert projection executes.
        handles.append(module.register_forward_pre_hook(fake_quant_hook))
        matched_weights += len(weight_names)

    if not handles:
        raise ValueError(
            "mxfp4_moe_weight_fake_quant did not find routed-expert "
            "linear_fc1/linear_fc2 modules."
        )
    print(
        "Registered MXFP4->MXFP8 fake-quant hooks on "
        f"{matched_weights} routed-expert weights across {len(handles)} modules."
    )
    return handles
