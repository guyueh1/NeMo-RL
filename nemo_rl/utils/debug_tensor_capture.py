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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

G_CAPTURE_ATTR = "_nemo_rl_debug_tensor_capture"
G_HANDLES_ATTR = "_nemo_rl_debug_tensor_capture_handles"


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


def _iter_model_roots(model: torch.nn.Module):
    seen: set[int] = set()
    stack = [model]
    while stack:
        cur = stack.pop(0)
        if not isinstance(cur, torch.nn.Module) or id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        for attr in ("module", "model", "language_model", "decoder"):
            child = getattr(cur, attr, None)
            if isinstance(child, torch.nn.Module):
                stack.append(child)


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the decoder layer ModuleList for vLLM or Megatron-style models."""
    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "decoder.layers",
        "language_model.decoder.layers",
        "module.decoder.layers",
        "module.module.decoder.layers",
    )
    for root in _iter_model_roots(model):
        for path in candidate_paths:
            layers = _get_attr_path(root, path)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return layers

        for _, module in root.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
                first = module[0]
                class_name = first.__class__.__name__.lower()
                if "layer" in class_name or "block" in class_name:
                    return module

    raise RuntimeError("Could not find decoder layers on model.")


def _callable_info(fn: Any) -> dict[str, Any]:
    """Return stable identifying information for a bound method/function."""
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


def inspect_vllm_layernorm_impl(model: torch.nn.Module) -> dict[str, Any]:
    """Inspect the runtime RMSNorm implementation used by vLLM layer 0.

    This is intentionally vLLM-specific and used only for numeric debugging.
    """
    layers = find_decoder_layers(model)
    layer0 = layers[0]
    module = getattr(layer0, "post_attention_layernorm", None)
    if module is None:
        raise RuntimeError("layer 0 does not expose post_attention_layernorm")

    from vllm.model_executor.layers import layernorm as vllm_layernorm
    from vllm.model_executor.layers.layernorm import RMSNorm

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


def _new_capture(num_layers: int, max_calls_per_module: int) -> dict[str, Any]:
    return {
        "module_inputs_by_layer": {},
        "module_outputs_by_layer": {},
        "module_input_calls_by_layer": {},
        "module_output_calls_by_layer": {},
        "module_input_total_calls_by_layer": {},
        "module_output_total_calls_by_layer": {},
        "first_layer_inputs": {},
        "first_layer_input_kwargs": {},
        "first_layer_outputs": {},
        "num_layers": num_layers,
        "max_calls_per_module": max_calls_per_module,
    }


def remove_debug_tensor_hooks(model: torch.nn.Module) -> dict[str, Any]:
    """Remove any tensor capture hooks previously installed on ``model``."""
    handles = getattr(model, G_HANDLES_ATTR, [])
    for handle in handles:
        handle.remove()
    setattr(model, G_HANDLES_ATTR, [])
    setattr(model, G_CAPTURE_ATTR, None)
    return {"num_removed_hooks": len(handles)}


def clear_debug_tensor_capture(model: torch.nn.Module) -> dict[str, Any]:
    """Clear captured tensors while keeping installed hook handles alive."""
    capture = getattr(model, G_CAPTURE_ATTR, None)
    if capture is None:
        return {"installed": False}

    num_layers = int(capture["num_layers"])
    max_calls_per_module = int(capture["max_calls_per_module"])
    capture.clear()
    capture.update(_new_capture(num_layers, max_calls_per_module))
    return {
        "installed": True,
        "num_layers": num_layers,
        "max_calls_per_module": max_calls_per_module,
    }


def install_debug_tensor_hooks(
    model: torch.nn.Module, *, max_calls_per_module: int = 1
) -> dict[str, Any]:
    """Capture layer inputs and first-layer module inputs/outputs.

    Layer 0 gets pre/post hooks on every submodule. Layers 1..N get only a
    pre-hook on the layer entrypoint. Call-list fields keep up to
    ``max_calls_per_module`` snapshots per module; total-call counters still
    track every observed forward.
    """
    if max_calls_per_module < 1:
        raise ValueError("max_calls_per_module must be >= 1")

    remove_debug_tensor_hooks(model)
    layers = find_decoder_layers(model)
    capture = _new_capture(len(layers), max_calls_per_module)
    handles = []

    def increment_total(kind: str, layer_idx: int, name: str) -> int:
        total_by_layer = capture[kind].setdefault(layer_idx, {})
        total_by_layer[name] = total_by_layer.get(name, 0) + 1
        return int(total_by_layer[name])

    def save_input(
        layer_idx: int, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        call_number = increment_total(
            "module_input_total_calls_by_layer", layer_idx, name
        )
        layer_input_calls = capture["module_input_calls_by_layer"].setdefault(
            layer_idx, {}
        )
        layer_inputs = capture["module_inputs_by_layer"].setdefault(layer_idx, {})
        if call_number <= max_calls_per_module:
            entry = _snapshot_call(args, kwargs)
            layer_input_calls.setdefault(name, []).append(entry)
            if name not in layer_inputs:
                layer_inputs[name] = entry
                if layer_idx == 0:
                    capture["first_layer_inputs"][name] = entry["args"]
                    capture["first_layer_input_kwargs"][name] = entry["kwargs"]

    def save_output(layer_idx: int, name: str, output: Any) -> None:
        call_number = increment_total(
            "module_output_total_calls_by_layer", layer_idx, name
        )
        layer_output_calls = capture["module_output_calls_by_layer"].setdefault(
            layer_idx, {}
        )
        layer_outputs = capture["module_outputs_by_layer"].setdefault(layer_idx, {})
        if call_number <= max_calls_per_module:
            output_entry = _snapshot_value(output)
            layer_output_calls.setdefault(name, []).append(output_entry)
            if name not in layer_outputs:
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

    first_layer = layers[0]
    for module_name, module in first_layer.named_modules():
        capture_name = module_name or "<layer>"
        handles.append(
            module.register_forward_pre_hook(
                make_pre_hook(0, capture_name), with_kwargs=True
            )
        )
        handles.append(
            module.register_forward_hook(
                make_post_hook(0, capture_name), with_kwargs=True
            )
        )

    for layer_idx, layer in enumerate(layers[1:], start=1):
        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(layer_idx, "<layer>"), with_kwargs=True
            )
        )

    setattr(model, G_CAPTURE_ATTR, capture)
    setattr(model, G_HANDLES_ATTR, handles)
    return {
        "num_layers": len(layers),
        "num_hooks": len(handles),
        "max_calls_per_module": max_calls_per_module,
    }


def get_debug_tensor_capture(model: torch.nn.Module) -> dict[str, Any]:
    capture = getattr(model, G_CAPTURE_ATTR, None)
    if capture is None:
        return {}
    return capture


def _call_counts(capture: dict[str, Any], key: str) -> dict[int, dict[str, int]]:
    return {
        int(layer_idx): {name: int(count) for name, count in layer_counts.items()}
        for layer_idx, layer_counts in capture.get(key, {}).items()
    }


def save_debug_tensor_capture(
    model: torch.nn.Module,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    capture = get_debug_tensor_capture(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"capture": capture, "metadata": metadata or {}}
    torch.save(payload, path)
    layer0_input_calls = capture.get("module_input_calls_by_layer", {}).get(0, {})
    return {
        "path": str(path),
        "num_layers": capture.get("num_layers"),
        "num_first_layer_modules": len(capture.get("first_layer_inputs", {})),
        "num_layer0_input_calls": {
            name: len(calls) for name, calls in layer0_input_calls.items()
        },
        "total_input_calls": _call_counts(capture, "module_input_total_calls_by_layer"),
        "total_output_calls": _call_counts(
            capture, "module_output_total_calls_by_layer"
        ),
    }
