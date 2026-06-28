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

"""Forward-hook tensor capture helpers for cross-engine numeric debugging."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

import torch

ROUTER_CONFIG_FIELDS = (
    "moe_router_pre_softmax",
    "moe_router_num_groups",
    "moe_router_group_topk",
    "moe_router_topk_scaling_factor",
    "moe_router_fusion",
    "moe_expert_capacity_factor",
    "moe_token_drop_policy",
    "moe_pad_expert_input_to_capacity",
    "moe_router_force_load_balancing",
    "moe_router_force_biased",
)


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot_value(v) for v in value)
    if isinstance(value, list):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _snapshot_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _snapshot_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": tuple(_snapshot_value(v) for v in args),
        "kwargs": {k: _snapshot_value(v) for k, v in kwargs.items()},
    }


def _get_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the decoder layer ModuleList for vLLM or Megatron language models."""
    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "decoder.layers",
        "language_model.decoder.layers",
    )
    for path in candidate_paths:
        layers = _get_attr_path(model, path)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            first = module[0]
            class_name = first.__class__.__name__.lower()
            if "layer" in class_name or "block" in class_name:
                return module

    raise RuntimeError("Could not find decoder layers on model.")


def _callable_info(fn: Any) -> dict[str, Any]:
    func = getattr(fn, "__func__", fn)
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = None
    return {
        "repr": repr(fn),
        "module": getattr(func, "__module__", None),
        "qualname": getattr(func, "__qualname__", None),
        "source_head": source[:800] if isinstance(source, str) else None,
    }


def _snapshot_router_config(module: torch.nn.Module) -> dict[str, Any]:
    config = getattr(module, "config", None)
    data = {
        "module_class": module.__class__.__module__
        + "."
        + module.__class__.__qualname__,
        "topk": _snapshot_value(getattr(module, "topk", None)),
        "score_function": _snapshot_value(getattr(module, "score_function", None)),
        "routing_type": _snapshot_value(getattr(module, "routing_type", None)),
        "layer_number": _snapshot_value(getattr(module, "layer_number", None)),
        "enable_expert_bias": _snapshot_value(
            getattr(module, "enable_expert_bias", None)
        ),
        "expert_bias": _snapshot_value(getattr(module, "expert_bias", None)),
    }
    if config is not None:
        for field in ROUTER_CONFIG_FIELDS:
            data[field] = _snapshot_value(getattr(config, field, None))
    return data


def inspect_vllm_layernorm_impl(model: torch.nn.Module) -> dict[str, Any]:
    """Inspect the runtime RMSNorm implementation used by vLLM layer 0."""
    from vllm.model_executor.layers import layernorm as vllm_layernorm
    from vllm.model_executor.layers.layernorm import RMSNorm

    layers = find_decoder_layers(model)
    layer0 = layers[0]
    module = getattr(layer0, "post_attention_layernorm", None)
    if module is None:
        module = next(
            (
                candidate
                for candidate in layer0.modules()
                if isinstance(candidate, RMSNorm)
            ),
            None,
        )
    if module is None:
        raise RuntimeError("layer 0 does not expose a vLLM RMSNorm module")

    return {
        "module_class": module.__class__.__module__
        + "."
        + module.__class__.__qualname__,
        "is_rmsnorm": isinstance(module, RMSNorm),
        "variance_epsilon": getattr(module, "variance_epsilon", None),
        "hidden_size": getattr(module, "hidden_size", None),
        "weight_dtype": str(getattr(getattr(module, "weight", None), "dtype", None)),
        "forward_method": _callable_info(getattr(module, "_forward_method", None)),
        "forward_cuda": _callable_info(getattr(module, "forward_cuda", None)),
        "class_forward_cuda": _callable_info(RMSNorm.forward_cuda),
        "fused_add_rms_norm": _callable_info(vllm_layernorm.fused_add_rms_norm),
        "rms_norm_batch_invariant": _callable_info(
            getattr(vllm_layernorm, "rms_norm_batch_invariant", None)
        ),
    }


def _normalise_capture_layers(
    capture_layers: Iterable[int] | None,
    num_layers: int,
) -> set[int]:
    if capture_layers is None:
        return {0}
    layers = {int(layer_idx) for layer_idx in capture_layers}
    invalid = sorted(
        layer_idx for layer_idx in layers if layer_idx < 0 or layer_idx >= num_layers
    )
    if invalid:
        raise ValueError(
            f"capture layer indices out of range for {num_layers} layers: {invalid}"
        )
    return layers


def install_debug_tensor_hooks(
    model: torch.nn.Module,
    capture_layers: Iterable[int] | None = None,
    capture_router_internals: bool = False,
) -> dict[str, Any]:
    """Capture layer inputs and first-layer module inputs/outputs.

    Selected capture layers get pre/post hooks on every submodule. Other layers
    get only a pre-hook on the layer entrypoint. Call-list fields capture every
    forward call; first-call fields are preserved for quick inspection.
    """
    layers = find_decoder_layers(model)
    capture_layer_set = _normalise_capture_layers(capture_layers, len(layers))
    capture: dict[str, Any] = {
        "module_inputs_by_layer": {},
        "module_outputs_by_layer": {},
        "module_input_calls_by_layer": {},
        "module_output_calls_by_layer": {},
        "first_layer_inputs": {},
        "first_layer_input_kwargs": {},
        "first_layer_outputs": {},
        "router_config_by_layer": {},
        "num_layers": len(layers),
        "captured_module_layers": sorted(capture_layer_set),
        "capture_router_internals": bool(capture_router_internals),
    }
    handles = []

    def save_input(
        layer_idx: int, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        layer_input_calls = capture["module_input_calls_by_layer"].setdefault(
            layer_idx, {}
        )
        layer_inputs = capture["module_inputs_by_layer"].setdefault(layer_idx, {})
        entry = _snapshot_call(args, kwargs)
        layer_input_calls.setdefault(name, []).append(entry)
        if name in layer_inputs:
            return
        layer_inputs[name] = entry
        if layer_idx == 0:
            capture["first_layer_inputs"][name] = entry["args"]
            capture["first_layer_input_kwargs"][name] = entry["kwargs"]

    def save_output(layer_idx: int, name: str, output: Any) -> None:
        layer_output_calls = capture["module_output_calls_by_layer"].setdefault(
            layer_idx, {}
        )
        layer_outputs = capture["module_outputs_by_layer"].setdefault(layer_idx, {})
        output_entry = _snapshot_value(output)
        layer_output_calls.setdefault(name, []).append(output_entry)
        if name in layer_outputs:
            return
        layer_outputs[name] = output_entry
        if layer_idx == 0:
            capture["first_layer_outputs"][name] = layer_outputs[name]

    def make_pre_hook(layer_idx: int, name: str):
        def hook(module, args, kwargs):  # noqa: ARG001
            save_input(layer_idx, name, args, kwargs)

        return hook

    def make_post_hook(layer_idx: int, name: str):
        def hook(module, args, kwargs, output):  # noqa: ARG001
            save_output(layer_idx, name, output)

        return hook

    def maybe_wrap_gating(layer_idx: int, name: str, module: torch.nn.Module) -> None:
        gating = getattr(module, "gating", None)
        if not callable(gating) or getattr(
            module, "_debug_capture_gating_wrapped", False
        ):
            return

        capture_name = f"{name}.gating"

        @wraps(gating)
        def wrapped_gating(*args, **kwargs):
            save_input(layer_idx, capture_name, args, kwargs)
            output = gating(*args, **kwargs)
            save_output(layer_idx, capture_name, output)
            return output

        setattr(module, "gating", wrapped_gating)
        setattr(module, "_debug_capture_gating_wrapped", True)

    def maybe_wrap_routing(layer_idx: int, name: str, module: torch.nn.Module) -> None:
        routing = getattr(module, "routing", None)
        if (
            not capture_router_internals
            or not callable(routing)
            or getattr(module, "_debug_capture_routing_wrapped", False)
        ):
            return

        capture_name = f"{name}.routing"
        router_config = capture["router_config_by_layer"].setdefault(layer_idx, {})
        router_config[capture_name] = _snapshot_router_config(module)

        @wraps(routing)
        def wrapped_routing(*args, **kwargs):
            save_input(layer_idx, capture_name, args, kwargs)
            output = routing(*args, **kwargs)
            save_output(layer_idx, capture_name, output)
            return output

        setattr(module, "routing", wrapped_routing)
        setattr(module, "_debug_capture_routing_wrapped", True)

    def maybe_wrap_vllm_select_experts(
        layer_idx: int, name: str, module: torch.nn.Module
    ) -> None:
        router = getattr(module, "router", None)
        select_experts = getattr(router, "select_experts", None)
        if not callable(select_experts) or getattr(
            router, "_debug_capture_select_experts_wrapped", False
        ):
            return

        capture_name = f"{name}.router.select_experts"

        @wraps(select_experts)
        def wrapped_select_experts(*args, **kwargs):
            save_input(layer_idx, capture_name, args, kwargs)
            output = select_experts(*args, **kwargs)
            save_output(layer_idx, capture_name, output)
            return output

        setattr(router, "select_experts", wrapped_select_experts)
        setattr(router, "_debug_capture_select_experts_wrapped", True)

    for layer_idx, layer in enumerate(layers):
        if layer_idx in capture_layer_set:
            for module_name, module in layer.named_modules():
                capture_name = module_name or "<layer>"
                maybe_wrap_gating(layer_idx, capture_name, module)
                maybe_wrap_routing(layer_idx, capture_name, module)
                maybe_wrap_vllm_select_experts(layer_idx, capture_name, module)
                handles.append(
                    module.register_forward_pre_hook(
                        make_pre_hook(layer_idx, capture_name), with_kwargs=True
                    )
                )
                handles.append(
                    module.register_forward_hook(
                        make_post_hook(layer_idx, capture_name), with_kwargs=True
                    )
                )
            continue

        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(layer_idx, "<layer>"), with_kwargs=True
            )
        )

    model._debug_tensor_capture = capture
    model._debug_tensor_capture_handles = handles
    return {
        "num_layers": len(layers),
        "num_hooks": len(handles),
        "captured_module_layers": sorted(capture_layer_set),
        "capture_router_internals": bool(capture_router_internals),
    }


def get_debug_tensor_capture(model: torch.nn.Module) -> dict[str, Any]:
    capture = getattr(model, "_debug_tensor_capture", None)
    if capture is None:
        return {}
    return capture


def save_debug_tensor_capture_from_env(model: torch.nn.Module) -> dict[str, Any]:
    path = os.environ["DEBUG_TENSOR_CAPTURE_PATH"]
    capture = get_debug_tensor_capture(model)
    torch.save(capture, path)
    layer_input_calls = capture.get("module_input_calls_by_layer", {})
    layer0_calls = layer_input_calls.get(0, {})
    return {
        "path": path,
        "num_layers": capture.get("num_layers"),
        "captured_module_layers": capture.get("captured_module_layers"),
        "num_first_layer_modules": len(capture.get("first_layer_inputs", {})),
        "num_layer0_input_calls": {
            name: len(calls) for name, calls in layer0_calls.items()
        },
    }
