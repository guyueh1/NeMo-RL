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

r"""Compare Megatron fused LayerNormLinear against split norm + linear.

This script builds two Megatron models from the same HF checkpoint:

* fused model: normal TE ``TELayerNormColumnParallelLinear`` for
  ``self_attention.linear_qkv`` and ``mlp.linear_fc1``.
* split model: the same model after replacing those modules with
  ``TENorm + TEColumnParallelLinear`` via Megatron's split helper.

It compares both full-model next-token logits and isolated TE module outputs on
identical captured fused-model inputs. The isolated comparison is the important
one when studying whether TE's fused path keeps the post-RMSNorm activation in
higher precision before GEMM.

Example:
    uv run --extra mcore torchrun --nproc_per_node=1 \
      my_script/compare_megatron_fused_norm_linear.py \
      --batch-invariant --num-prompts 4 --layers 0 \
      --output my_script/fused_vs_split_norm_linear.pt
"""

from __future__ import annotations

import argparse
import inspect
import os
import types
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, print_rank_0
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import Float16Module
from transformers import AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET = "openai/gsm8k"
DEFAULT_DATASET_SUBSET = "main"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_FIELD = "question"
DEFAULT_NUM_PROMPTS = 4
DEFAULT_TARGETS = ("linear_qkv", "linear_fc1")

TARGET_PATHS = {
    "linear_qkv": ("self_attention", "linear_qkv"),
    "linear_fc1": ("mlp", "linear_fc1"),
}


def load_prompts(dataset, subset, split, field, n, seed):
    """Load non-empty text prompts from a Hugging Face dataset."""
    from datasets import load_dataset

    kwargs = {"split": split}
    ds = (
        load_dataset(dataset, subset, **kwargs)
        if subset
        else load_dataset(dataset, **kwargs)
    )
    ds = ds.shuffle(seed=seed)
    prompts = []
    for row in ds:
        text = row.get(field)
        if isinstance(text, str) and text.strip():
            prompts.append(text.strip())
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
        return prompts, token_ids_list

    token_ids_list = _normalise_token_ids_list(payload)
    prompts = [f"{token_ids_key}[{i}]" for i in range(len(token_ids_list))]
    return prompts, token_ids_list


def parse_layers(value: str) -> list[int]:
    """Parse comma/range layer specs such as ``0,2,4-6``."""
    layers: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid layer range {item!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise ValueError("--layers parsed to an empty set")
    return sorted(layers)


def parse_targets(value: str) -> list[str]:
    targets = []
    for item in value.split(","):
        target = item.strip()
        if not target:
            continue
        if target not in TARGET_PATHS:
            raise ValueError(
                f"unknown target {target!r}; expected one of {sorted(TARGET_PATHS)}"
            )
        targets.append(target)
    if not targets:
        raise ValueError("--targets parsed to an empty set")
    return targets


class SingleBatchIterator:
    """Single-use iterator expected by Megatron forward schedules."""

    def __init__(self, input_ids, position_ids):
        self.batch = {"tokens": input_ids, "position_ids": position_ids}
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def forward_step(data_iterator, model, **kwargs):  # noqa: ARG001
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **_):
        return x

    return model(**forward_args), loss_func


def unwrap_model(m):
    if isinstance(m, Float16Module):
        m = m.module
    if hasattr(m, "language_model"):
        m = m.language_model
    return m


def get_target_module(inner, layer_idx: int, target: str):
    owner_name, attr_name = TARGET_PATHS[target]
    layer = inner.decoder.layers[layer_idx]
    owner = getattr(layer, owner_name)
    return getattr(owner, attr_name)


def set_target_module(inner, layer_idx: int, target: str, module) -> None:
    owner_name, attr_name = TARGET_PATHS[target]
    layer = inner.decoder.layers[layer_idx]
    owner = getattr(layer, owner_name)
    setattr(owner, attr_name, module)


def maybe_tensor_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(maybe_tensor_cpu(item) for item in value)
    if isinstance(value, list):
        return [maybe_tensor_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: maybe_tensor_cpu(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def tensor_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            yield from tensor_leaves(item, child)
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from tensor_leaves(item, child)


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_a": tuple(a.shape),
            "shape_b": tuple(b.shape),
            "shape_match": False,
        }

    a32 = a.detach().to(device="cpu", dtype=torch.float32)
    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    diff = a32 - b32
    abs_diff = diff.abs()
    denom = b32.abs().clamp_min(1.0e-12)
    rel_diff = abs_diff / denom
    return {
        "shape": tuple(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "shape_match": True,
        "exact": bool(torch.equal(a32, b32)),
        "mean_abs": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
        "mean_rel": float(rel_diff.mean().item()),
        "max_rel": float(rel_diff.max().item()),
        "mean_signed": float(diff.mean().item()),
        "a_norm": float(a32.norm().item()),
        "b_norm": float(b32.norm().item()),
    }


def compare_values(a: Any, b: Any) -> dict[str, Any]:
    a_leaves = dict(tensor_leaves(a))
    b_leaves = dict(tensor_leaves(b))
    names = sorted(set(a_leaves) | set(b_leaves))
    return {
        name: compare_tensors(a_leaves[name], b_leaves[name])
        if name in a_leaves and name in b_leaves
        else {"present_a": name in a_leaves, "present_b": name in b_leaves}
        for name in names
    }


def direct_te_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation_dtype: torch.dtype,
    sm_margin: int,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """Call the same TE RMSNorm extension that LayerNormLinear uses."""
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine_torch import rmsnorm_fwd

    hidden_size = weight.numel()
    x_2d = x.view((-1, hidden_size))
    x_2d = x_2d.to(dtype=activation_dtype) if x_2d.dtype != activation_dtype else x_2d
    w = weight.view((hidden_size,))
    w = w.to(dtype=activation_dtype) if w.dtype != activation_dtype else w
    y, _, _ = rmsnorm_fwd(
        x_2d,
        w,
        eps,
        None,
        None,
        TE_DType[activation_dtype],
        sm_margin,
        zero_centered_gamma,
    )
    return y.view_as(x)


def layernorm_linear_parts(value: Any) -> dict[str, Any]:
    """Extract linear output, returned bias, and optional LN output."""
    result = {"linear": None, "bias": None, "layernorm": None, "raw": value}
    if isinstance(value, tuple) and len(value) == 2:
        first, second = value
        result["bias"] = second
        if isinstance(first, tuple) and len(first) == 2:
            result["linear"] = first[0]
            result["layernorm"] = first[1]
        else:
            result["linear"] = first
    else:
        result["linear"] = value
    return result


def callable_info(fn: Any) -> dict[str, Any]:
    func = getattr(fn, "__func__", fn)
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = None
    return {
        "repr": repr(fn),
        "module": getattr(func, "__module__", None),
        "qualname": getattr(func, "__qualname__", None),
        "source_file": inspect.getsourcefile(func),
        "source_head": source[:1200] if isinstance(source, str) else None,
    }


def implementation_info(module: torch.nn.Module) -> dict[str, Any]:
    info = {
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "forward": callable_info(getattr(module, "forward", None)),
        "class_forward": callable_info(getattr(module.__class__, "forward", None)),
    }
    for attr in (
        "normalization",
        "eps",
        "zero_centered_gamma",
        "return_layernorm_output",
        "return_layernorm_output_gathered",
        "sequence_parallel",
        "parallel_mode",
        "tp_size",
        "te_return_bias",
    ):
        if hasattr(module, attr):
            value = getattr(module, attr)
            if isinstance(value, torch.Tensor):
                value = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            info[attr] = value
    return info


def reset_first_microbatch(module: torch.nn.Module) -> None:
    if hasattr(module, "is_first_microbatch"):
        module.is_first_microbatch = True
    linear = getattr(module, "_linear", None)
    if linear is not None and hasattr(linear, "is_first_microbatch"):
        linear.is_first_microbatch = True


def run_model(model_list, input_ids, position_ids):
    with torch.no_grad():
        fwd_bwd = get_forward_backward_func()
        iterator = SingleBatchIterator(input_ids, position_ids)
        output = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=iterator,
            model=model_list,
            num_microbatches=1,
            forward_only=True,
            seq_length=input_ids.size(1),
            micro_batch_size=input_ids.size(0),
            collect_non_loss_data=True,
        )
    if isinstance(output, list) and len(output) > 0:
        output = output[0]
    return output


def gather_last_token_logits(output, seq_lens: list[int]):
    if not isinstance(output, torch.Tensor):
        return None
    logits = output.detach().to(torch.float32).cpu()
    if logits.dim() != 3:
        raise RuntimeError(f"expected logits [B, S, V], got {tuple(logits.shape)}")
    idx = torch.tensor([seq_len - 1 for seq_len in seq_lens], dtype=torch.long)
    return logits[torch.arange(len(seq_lens)), idx]


def install_input_capture_hooks(inner, layers: list[int], targets: list[str]):
    captures: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(key: str):
        def hook(module, args, kwargs):  # noqa: ARG001
            if key in captures:
                return
            captures[key] = {
                "args": maybe_tensor_cpu(args),
                "kwargs": maybe_tensor_cpu(kwargs),
                "module_class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            }

        return hook

    for layer_idx in layers:
        for target in targets:
            module = get_target_module(inner, layer_idx, target)
            key = f"layer{layer_idx}.{target}"
            handles.append(
                module.register_forward_pre_hook(make_hook(key), with_kwargs=True)
            )
    return captures, handles


def split_all_targets(inner) -> int:
    from nemo_rl.models.policy.megatron.vllm_kernel_patches import (
        split_all_layers_fused_layernorm_linear,
    )

    return split_all_layers_fused_layernorm_linear(inner)


def patch_fused_targets_to_unfused_forward(
    inner, layers: list[int], targets: list[str]
) -> int:
    """Patch selected fused modules to execute an internally split forward."""
    from megatron.core.extensions.transformer_engine import (
        split_te_layernorm_column_parallel_linear,
    )

    patched = 0
    for layer_idx in layers:
        for target in targets:
            fused = get_target_module(inner, layer_idx, target)
            norm, linear = split_te_layernorm_column_parallel_linear(
                fused, inner.config
            )
            ref_weight = fused.weight
            norm = norm.to(device=ref_weight.device, dtype=ref_weight.dtype).eval()
            linear = linear.to(device=ref_weight.device, dtype=ref_weight.dtype).eval()
            object.__setattr__(fused, "_debug_unfused_norm", norm)
            object.__setattr__(fused, "_debug_unfused_linear", linear)

            def _forward(self, x):
                out = self._debug_unfused_linear(self._debug_unfused_norm(x))
                self.is_first_microbatch = False
                return out

            fused.forward = types.MethodType(_forward, fused)
            patched += 1
    return patched


def provide_model_list(model_provider):
    model_list = model_provider.provide_distributed_model(wrap_with_ddp=False)
    return [model.cuda() for model in model_list]


def run_direct_module_compare(
    fused_inner,
    split_inner,
    captures: dict[str, dict[str, Any]],
    layers: list[int],
    targets: list[str],
    *,
    probe_return_layernorm_output: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for layer_idx in layers:
        for target in targets:
            key = f"layer{layer_idx}.{target}"
            entry = captures.get(key)
            if entry is None:
                results[key] = {"error": "missing captured fused-model input"}
                continue
            input_tensor = first_tensor(entry["args"])
            if input_tensor is None:
                results[key] = {"error": "captured input did not contain a tensor"}
                continue
            x = input_tensor.cuda()
            fused_module = get_target_module(fused_inner, layer_idx, target)
            split_module = get_target_module(split_inner, layer_idx, target)
            reset_first_microbatch(fused_module)
            reset_first_microbatch(split_module)

            with torch.no_grad():
                fused_out = fused_module(x)
                split_out = split_module(x)

            fused_parts = layernorm_linear_parts(fused_out)
            split_parts = layernorm_linear_parts(split_out)
            target_result: dict[str, Any] = {
                "input_shape": tuple(x.shape),
                "fused_impl": implementation_info(fused_module),
                "split_impl": implementation_info(split_module),
                "raw_output": compare_values(fused_out, split_out),
            }
            if isinstance(fused_parts["linear"], torch.Tensor) and isinstance(
                split_parts["linear"], torch.Tensor
            ):
                target_result["linear_output"] = compare_tensors(
                    fused_parts["linear"],
                    split_parts["linear"],
                )

            split_norm = getattr(split_module, "_norm", None)
            if split_norm is not None:
                with torch.no_grad():
                    split_norm_out = split_norm(x)
                target_result["split_norm_output"] = {
                    "shape": tuple(split_norm_out.shape),
                    "dtype": str(split_norm_out.dtype),
                    "impl": implementation_info(split_norm),
                }
            else:
                split_norm_out = None

            if hasattr(fused_module, "layer_norm_weight"):
                activation_dtype = getattr(
                    fused_module,
                    "activation_dtype",
                    fused_module.layer_norm_weight.dtype,
                )
                fwd_ln_sm_margin = getattr(fused_module, "inf_ln_sm_margin", 0)
                with torch.no_grad():
                    direct_norm_out = direct_te_rmsnorm(
                        x,
                        fused_module.layer_norm_weight,
                        fused_module.eps,
                        activation_dtype,
                        fwd_ln_sm_margin,
                        fused_module.zero_centered_gamma,
                    )
                direct_norm_result = {
                    "shape": tuple(direct_norm_out.shape),
                    "dtype": str(direct_norm_out.dtype),
                    "activation_dtype": str(activation_dtype),
                    "sm_margin": fwd_ln_sm_margin,
                }
                if isinstance(split_norm_out, torch.Tensor):
                    direct_norm_result["vs_split_norm"] = compare_tensors(
                        direct_norm_out,
                        split_norm_out,
                    )
                target_result["direct_te_rmsnorm"] = direct_norm_result
            else:
                direct_norm_out = None

            if probe_return_layernorm_output and hasattr(
                fused_module, "return_layernorm_output"
            ):
                old_return_ln = fused_module.return_layernorm_output
                reset_first_microbatch(fused_module)
                try:
                    fused_module.return_layernorm_output = True
                    with torch.no_grad():
                        fused_probe_out = fused_module(x)
                finally:
                    fused_module.return_layernorm_output = old_return_ln
                fused_probe_parts = layernorm_linear_parts(fused_probe_out)
                probe_result: dict[str, Any] = {
                    "raw_output": compare_values(fused_probe_out, split_out),
                }
                if isinstance(fused_probe_parts["linear"], torch.Tensor) and isinstance(
                    split_parts["linear"], torch.Tensor
                ):
                    probe_result["linear_output"] = compare_tensors(
                        fused_probe_parts["linear"],
                        split_parts["linear"],
                    )
                if isinstance(fused_probe_parts["layernorm"], torch.Tensor):
                    probe_result["returned_layernorm"] = {
                        "shape": tuple(fused_probe_parts["layernorm"].shape),
                        "dtype": str(fused_probe_parts["layernorm"].dtype),
                    }
                    if isinstance(direct_norm_out, torch.Tensor):
                        probe_result["returned_layernorm_vs_direct_te_rmsnorm"] = (
                            compare_tensors(
                                fused_probe_parts["layernorm"],
                                direct_norm_out,
                            )
                        )
                    if isinstance(split_norm_out, torch.Tensor):
                        probe_result["returned_layernorm_vs_split_norm"] = (
                            compare_tensors(
                                fused_probe_parts["layernorm"],
                                split_norm_out,
                            )
                        )
                target_result["fused_return_layernorm_output_probe"] = probe_result

            results[key] = target_result
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-subset", default=DEFAULT_DATASET_SUBSET)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--dataset-field", default=DEFAULT_DATASET_FIELD)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument(
        "--token-ids-file",
        default=None,
        help="Torch .pt file containing token id rows instead of loading a dataset.",
    )
    parser.add_argument("--token-ids-key", default="token_ids_list")
    parser.add_argument(
        "--layers",
        default="0",
        help="Comma/range layer selector, e.g. 0,2,4-6. Default: 0.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated targets: linear_qkv,linear_fc1.",
    )
    parser.add_argument("--batch-invariant", action="store_true")
    parser.add_argument(
        "--vllm-rmsnorm",
        action="store_true",
        help=(
            "After enabling batch-invariant mode, route both split TE RMSNorm "
            "and fused TE LayerNormLinear RMSNorm through vLLM's BI RMSNorm."
        ),
    )
    parser.add_argument(
        "--te-bi-rmsnorm",
        action="store_true",
        help=(
            "After enabling batch-invariant mode, route fused TE "
            "LayerNormLinear RMSNorm through Megatron's current BI RMSNorm "
            "function. This does not require vLLM to be importable."
        ),
    )
    parser.add_argument(
        "--mxfp8",
        action="store_true",
        help="Enable Megatron MXFP8 fp8_autocast for the comparison.",
    )
    parser.add_argument(
        "--fp8-format",
        default="e4m3",
        choices=["e4m3", "hybrid"],
    )
    parser.add_argument(
        "--mxfp8-bi-dequant",
        action="store_true",
        help="Under --mxfp8 --batch-invariant, dequant MXFP8 operands into BF16 BI GEMM.",
    )
    parser.add_argument(
        "--patch-fused-forward-to-unfused",
        action="store_true",
        help=(
            "Patch selected fused-model target modules so their forward executes "
            "a split TENorm + TEColumnParallelLinear path. This is a control "
            "for whether the mismatch comes from TE's fused forward semantics."
        ),
    )
    parser.add_argument(
        "--split-model-construction",
        default="second-provider-call",
        choices=["second-provider-call"],
        help=(
            "How to construct the split comparison model. The default calls "
            "provide_distributed_model() a second time from the same provider; "
            "deepcopy is intentionally not used because Megatron modules hold "
            "non-pickleable distributed process groups."
        ),
    )
    parser.add_argument(
        "--no-probe-return-layernorm-output",
        action="store_true",
        help=(
            "Do not run the direct-call probe that temporarily sets "
            "return_layernorm_output=True on fused modules."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "megatron_fused_vs_split_norm_linear.pt",
        ),
    )
    args = parser.parse_args()
    args.layers = parse_layers(args.layers)
    args.targets = parse_targets(args.targets)
    return args


def configure_batch_invariant(args) -> None:
    if not args.batch_invariant:
        if args.vllm_rmsnorm or args.te_bi_rmsnorm:
            raise ValueError(
                "--vllm-rmsnorm and --te-bi-rmsnorm require --batch-invariant"
            )
        return
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        enable_batch_invariant_mode,
    )

    enable_batch_invariant_mode()
    print_rank_0("[compare] batch_invariant_mode ENABLED")
    if args.vllm_rmsnorm:
        from nemo_rl.models.policy.megatron.vllm_kernel_patches import (
            install_vllm_style_rmsnom_to_te,
            install_vllm_style_rmsnorm,
        )

        install_vllm_style_rmsnorm()
        install_vllm_style_rmsnom_to_te()
        print_rank_0("[compare] vLLM-style RMSNorm patches installed")
    elif args.te_bi_rmsnorm:
        from nemo_rl.models.policy.megatron.vllm_kernel_patches import (
            install_vllm_style_rmsnom_to_te,
        )

        install_vllm_style_rmsnom_to_te()
        print_rank_0("[compare] fused TE RMSNorm -> Megatron BI RMSNorm installed")
    if args.mxfp8:
        from megatron_forward import (
            install_mxfp8_compact_scales,
            install_mxfp8_dequant_for_bi_gemm,
            install_mxfp8_passthrough_for_bi_gemm,
        )

        if args.mxfp8_bi_dequant:
            install_mxfp8_compact_scales()
            install_mxfp8_dequant_for_bi_gemm()
            print_rank_0("[compare] MXFP8 dequant -> BF16 BI GEMM patch installed")
        else:
            install_mxfp8_passthrough_for_bi_gemm()
            print_rank_0("[compare] MXFP8 passthrough BI GEMM wrapper installed")


def main():
    args = parse_args()
    configure_batch_invariant(args)

    print_rank_0(f"[compare] loading bridge for {args.model}")
    bridge = AutoBridge.from_hf_pretrained(
        args.model,
        trust_remote_code=is_safe_repo(trust_remote_code=False, hf_path=args.model),
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.expert_model_parallel_size = 1
    model_provider.expert_tensor_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.gradient_accumulation_fusion = False
    if args.mxfp8:
        model_provider.fp8 = args.fp8_format
        model_provider.fp8_recipe = "mxfp8"
        print_rank_0(
            f"[compare] MXFP8 enabled: fp8={args.fp8_format}, fp8_recipe=mxfp8"
        )
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    fused_model_list = provide_model_list(model_provider)
    split_model_list = provide_model_list(model_provider)

    for model in [*fused_model_list, *split_model_list]:
        model.eval()
        disable_mtp_for_inference(model)

    assert len(fused_model_list) == 1, "expected pp=1 and a single model chunk"
    assert len(split_model_list) == 1, "expected pp=1 and a single model chunk"
    fused_inner = unwrap_model(fused_model_list[0])
    split_inner = unwrap_model(split_model_list[0])

    num_split_layers = split_all_targets(split_model_list)
    print_rank_0(
        "[compare] split model converted "
        f"{num_split_layers} decoder layers to TENorm + TEColumnParallelLinear"
    )

    if args.patch_fused_forward_to_unfused:
        num_patched = patch_fused_targets_to_unfused_forward(
            fused_inner,
            args.layers,
            args.targets,
        )
        print_rank_0(
            "[compare] patched fused model target forwards to internally "
            f"split execution for {num_patched} modules"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.token_ids_file:
        prompts, token_ids_list = load_token_ids_from_file(
            args.token_ids_file,
            args.token_ids_key,
        )
        print_rank_0(
            f"[compare] token ids: {args.token_ids_file} "
            f"key={args.token_ids_key!r} n={len(prompts)}"
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
        print_rank_0(
            f"[compare] dataset: {args.dataset}:{args.dataset_subset}:"
            f"{args.dataset_split} field={args.dataset_field!r} n={len(prompts)}"
        )

    seq_lens = [len(ids) for ids in token_ids_list]
    print_rank_0(
        f"[compare] seq_len min/mean/max = {min(seq_lens)}/"
        f"{sum(seq_lens) / len(seq_lens):.1f}/{max(seq_lens)}"
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    padded_seq_len = max(seq_lens)
    if args.mxfp8 and padded_seq_len % 32 != 0:
        padded_seq_len = ((padded_seq_len + 31) // 32) * 32
        print_rank_0(
            f"[compare] padded seq_len {max(seq_lens)} -> {padded_seq_len} for MXFP8"
        )
    padded_ids = [
        ids + [pad_id] * (padded_seq_len - len(ids)) for ids in token_ids_list
    ]
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device="cuda")
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids)
    )

    captures, handles = install_input_capture_hooks(
        fused_inner,
        args.layers,
        args.targets,
    )
    fused_output = run_model(fused_model_list, input_ids, position_ids)
    for handle in handles:
        handle.remove()
    split_output = run_model(split_model_list, input_ids, position_ids)

    fused_logits = gather_last_token_logits(fused_output, seq_lens)
    split_logits = gather_last_token_logits(split_output, seq_lens)
    logits_compare = None
    if fused_logits is not None and split_logits is not None:
        logits_compare = compare_tensors(fused_logits, split_logits)
        print_rank_0(
            "[compare] last-token logits fused-vs-split "
            f"mean_abs={logits_compare['mean_abs']:.9e} "
            f"max_abs={logits_compare['max_abs']:.9e} "
            f"exact={logits_compare['exact']}"
        )

    direct_results = run_direct_module_compare(
        fused_inner,
        split_inner,
        captures,
        args.layers,
        args.targets,
        probe_return_layernorm_output=not args.no_probe_return_layernorm_output
        and not args.patch_fused_forward_to_unfused,
    )
    for key, value in direct_results.items():
        linear = value.get("linear_output", {})
        if linear:
            print_rank_0(
                f"[compare] {key} direct fused-vs-split "
                f"mean_abs={linear['mean_abs']:.9e} "
                f"max_abs={linear['max_abs']:.9e} "
                f"exact={linear['exact']}"
            )
        probe = value.get("fused_return_layernorm_output_probe", {})
        probe_linear = probe.get("linear_output", {})
        if probe_linear:
            print_rank_0(
                f"[compare] {key} return_layernorm_output probe "
                f"mean_abs={probe_linear['mean_abs']:.9e} "
                f"max_abs={probe_linear['max_abs']:.9e} "
                f"exact={probe_linear['exact']}"
            )
        probe_norm = probe.get("returned_layernorm_vs_split_norm", {})
        if probe_norm:
            print_rank_0(
                f"[compare] {key} returned LN vs split norm "
                f"mean_abs={probe_norm['mean_abs']:.9e} "
                f"max_abs={probe_norm['max_abs']:.9e} "
                f"exact={probe_norm['exact']}"
            )
        direct_norm = value.get("direct_te_rmsnorm", {}).get("vs_split_norm", {})
        if direct_norm:
            print_rank_0(
                f"[compare] {key} direct TE RMSNorm vs split norm "
                f"mean_abs={direct_norm['mean_abs']:.9e} "
                f"max_abs={direct_norm['max_abs']:.9e} "
                f"exact={direct_norm['exact']}"
            )
        probe_direct_norm = probe.get("returned_layernorm_vs_direct_te_rmsnorm", {})
        if probe_direct_norm:
            print_rank_0(
                f"[compare] {key} returned LN vs direct TE RMSNorm "
                f"mean_abs={probe_direct_norm['mean_abs']:.9e} "
                f"max_abs={probe_direct_norm['max_abs']:.9e} "
                f"exact={probe_direct_norm['exact']}"
            )

    if dist.is_initialized() and dist.get_rank() == 0:
        payload = {
            "args": vars(args),
            "prompts": prompts,
            "token_ids_list": token_ids_list,
            "seq_lens": seq_lens,
            "last_token_logits": {
                "fused": fused_logits,
                "split": split_logits,
                "compare": logits_compare,
            },
            "captured_inputs": captures,
            "direct_module_compare": direct_results,
        }
        torch.save(payload, args.output)
        print(f"[compare] saved comparison payload to {args.output}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
