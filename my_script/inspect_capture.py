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

"""Inspect selected module entries in standalone forward captures."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Capture label and torch .pt path. May be passed multiple times.",
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Module names to inspect within the selected layer.",
    )
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print min/max/mean for floating-point tensors.",
    )
    return parser.parse_args()


def layer_map(source: Mapping[Any, Any], layer_idx: int) -> Mapping[str, Any]:
    layer = source.get(layer_idx)
    if layer is None:
        layer = source.get(str(layer_idx), {})
    if isinstance(layer, Mapping):
        return layer
    return {}


def tensor_summary(tensor: torch.Tensor, *, include_stats: bool) -> str:
    summary = f"Tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}"
    if include_stats and tensor.numel() > 0 and tensor.is_floating_point():
        values = tensor.float()
        summary += (
            f" min={float(values.min()):.6g}"
            f" max={float(values.max()):.6g}"
            f" mean={float(values.mean()):.6g}"
        )
    return summary


def describe_value(
    value: Any,
    *,
    indent: int,
    max_depth: int,
    max_items: int,
    include_stats: bool,
) -> None:
    prefix = "  " * indent
    if isinstance(value, torch.Tensor):
        print(prefix + tensor_summary(value, include_stats=include_stats))
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        print(prefix + repr(value))
        return
    if indent >= max_depth:
        print(prefix + f"{type(value).__name__} ...")
        return
    if isinstance(value, Mapping):
        keys = list(value.keys())
        print(
            prefix + f"{type(value).__name__} len={len(keys)} keys={keys[:max_items]!r}"
        )
        for key in keys[:max_items]:
            print(prefix + f"- {key!r}:")
            describe_value(
                value[key],
                indent=indent + 1,
                max_depth=max_depth,
                max_items=max_items,
                include_stats=include_stats,
            )
        if len(keys) > max_items:
            print(prefix + f"... {len(keys) - max_items} more keys")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        print(prefix + f"{type(value).__name__} len={len(value)}")
        for idx, item in enumerate(value[:max_items]):
            print(prefix + f"- [{idx}]:")
            describe_value(
                item,
                indent=indent + 1,
                max_depth=max_depth,
                max_items=max_items,
                include_stats=include_stats,
            )
        if len(value) > max_items:
            print(prefix + f"... {len(value) - max_items} more items")
        return
    print(prefix + repr(value))


def print_entry(
    payload: Mapping[str, Any],
    *,
    layer_idx: int,
    name: str,
    direction: str,
    max_depth: int,
    max_items: int,
    max_calls: int,
    include_stats: bool,
) -> None:
    first_key = f"module_{direction}s_by_layer"
    call_key = f"module_{direction}_calls_by_layer"
    first_entry = layer_map(payload.get(first_key, {}), layer_idx).get(name)
    calls = layer_map(payload.get(call_key, {}), layer_idx).get(name)

    if first_entry is None and not calls:
        return

    print(f"\n[{direction}] layer={layer_idx} name={name}")
    if first_entry is not None:
        print("first:")
        describe_value(
            first_entry,
            indent=1,
            max_depth=max_depth,
            max_items=max_items,
            include_stats=include_stats,
        )
    if calls:
        print(f"calls len={len(calls)}")
        for idx, call in enumerate(calls[:max_calls]):
            print(f"call[{idx}]:")
            describe_value(
                call,
                indent=1,
                max_depth=max_depth,
                max_items=max_items,
                include_stats=include_stats,
            )
        if len(calls) > max_calls:
            print(f"... {len(calls) - max_calls} more calls")


def main() -> None:
    args = parse_args()
    for label, path_text in args.capture:
        path = Path(path_text)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        print("=" * 100)
        print(f"capture={label} path={path}")
        print(f"model_family={payload.get('model_family', '?')}")
        print(f"captured_module_layers={payload.get('captured_module_layers', '?')}")
        for name in args.names:
            print_entry(
                payload,
                layer_idx=args.layer,
                name=name,
                direction="input",
                max_depth=args.max_depth,
                max_items=args.max_items,
                max_calls=args.max_calls,
                include_stats=args.stats,
            )
            print_entry(
                payload,
                layer_idx=args.layer,
                name=name,
                direction="output",
                max_depth=args.max_depth,
                max_items=args.max_items,
                max_calls=args.max_calls,
                include_stats=args.stats,
            )


if __name__ == "__main__":
    main()
