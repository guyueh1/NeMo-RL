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

"""Compare two vLLM captures and optional Megatron targets at selected layers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from compare import (
    diff_stats,
    get_layer_entry,
    layer_entry_stream,
    normalize_router_layout,
    select_tensor,
    sparse_routing_from_topk,
)


@dataclass(frozen=True)
class TensorSpec:
    label: str
    base_name: str
    target_name: str | None
    base_outputs: bool
    target_outputs: bool
    base_selectors: tuple[str, ...]
    target_selectors: tuple[str, ...]


G_TENSOR_SPECS = (
    TensorSpec(
        label="mixer/mlp output",
        base_name="mixer",
        target_name="mlp",
        base_outputs=True,
        target_outputs=True,
        base_selectors=("output", "item0", "first"),
        target_selectors=("item0", "output", "first"),
    ),
    TensorSpec(
        label="raw router logits",
        base_name="mixer.gate",
        target_name="mlp.router.gating",
        base_outputs=True,
        target_outputs=True,
        base_selectors=("output", "item0", "first"),
        target_selectors=("output", "item0", "first"),
    ),
    TensorSpec(
        label="vLLM experts output",
        base_name="mixer.experts",
        target_name=None,
        base_outputs=True,
        target_outputs=False,
        base_selectors=("output", "item0", "first"),
        target_selectors=(),
    ),
    TensorSpec(
        label="shared fc2/down output",
        base_name="mixer.shared_experts.down_proj",
        target_name="mlp.shared_experts.linear_fc2",
        base_outputs=True,
        target_outputs=True,
        base_selectors=("output", "item0", "first"),
        target_selectors=("output", "item0", "first"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline vLLM capture.")
    parser.add_argument("--variant", required=True, help="Variant vLLM capture.")
    parser.add_argument(
        "--megatron",
        default=None,
        help="Optional Megatron capture used to report error deltas.",
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--variant-label", default="variant")
    parser.add_argument(
        "--layers",
        default="17",
        help="Layer list/ranges to inspect, for example '15-18,21'.",
    )
    parser.add_argument(
        "--top-rows",
        type=int,
        default=10,
        help="Rows to print for route/output outlier reports.",
    )
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_payload(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dict payload")
    return payload


def trim_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != b.dim():
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
    shape = tuple(min(left, right) for left, right in zip(a.shape, b.shape))
    index = tuple(slice(0, size) for size in shape)
    return a[index], b[index]


def trim_triple(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = trim_pair(a, b)
    a, c = trim_pair(a, c)
    b, c = trim_pair(b, c)
    shape = tuple(
        min(a_s, b_s, c_s) for a_s, b_s, c_s in zip(a.shape, b.shape, c.shape)
    )
    index = tuple(slice(0, size) for size in shape)
    return a[index], b[index], c[index]


def tensor_from_entry(
    payload: dict[str, Any],
    *,
    layer_idx: int,
    name: str,
    outputs: bool,
    selectors: tuple[str, ...],
    seq_lens: list[int],
) -> tuple[torch.Tensor | None, str | None]:
    entry = get_layer_entry(payload, layer_idx, name, outputs=outputs)
    tensor, selector = select_tensor(entry, selectors)
    if tensor is None:
        return None, selector
    return normalize_router_layout(tensor, seq_lens).float(), selector


def print_stats_line(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = trim_pair(a, b)
    stats = diff_stats(a, b)
    print(
        f"  {label:<34s} max={stats['max_abs_diff']:.6f} "
        f"mean={stats['mean_abs_diff']:.4e} cos={stats['cos_sim']:.6f} "
        f"shape={tuple(a.shape)}"
    )


def row_mean_abs(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.float().abs()
    return tensor.float().abs().reshape(tensor.shape[0], -1).mean(dim=1)


def row_max_abs(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.float().abs()
    return tensor.float().abs().reshape(tensor.shape[0], -1).max(dim=1).values


def ids_for_row(route_map: torch.Tensor, row_idx: int) -> str:
    ids = route_map[row_idx].nonzero(as_tuple=False).flatten().tolist()
    return ",".join(str(int(idx)) for idx in ids)


def routing_from_capture(
    payload: dict[str, Any], layer_idx: int, seq_lens: list[int]
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    select_entry = get_layer_entry(
        payload,
        layer_idx,
        "mixer.experts.router.select_experts",
        outputs=True,
    )
    gate_entry = get_layer_entry(payload, layer_idx, "mixer.gate", outputs=True)
    topk_weights, _ = select_tensor(select_entry, ("item0", "output", "first"))
    topk_ids, _ = select_tensor(select_entry, ("item1",))
    logits, _ = select_tensor(gate_entry, ("output", "item0", "first"))
    if topk_weights is None or topk_ids is None:
        return None, None, None
    topk_weights = normalize_router_layout(topk_weights, seq_lens)
    topk_ids = normalize_router_layout(topk_ids, seq_lens)
    if logits is not None:
        logits = normalize_router_layout(logits, seq_lens).float()
        num_experts = logits.shape[-1]
    else:
        num_experts = int(topk_ids.max().item()) + 1
    probs, route_map = sparse_routing_from_topk(topk_weights, topk_ids, num_experts)
    return probs.float(), route_map.bool(), logits


def print_router_report(
    *,
    layer_idx: int,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    seq_lens: list[int],
    top_rows: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    base_probs, base_map, base_logits = routing_from_capture(
        baseline, layer_idx, seq_lens
    )
    variant_probs, variant_map, variant_logits = routing_from_capture(
        variant, layer_idx, seq_lens
    )
    if (
        base_probs is None
        or base_map is None
        or variant_probs is None
        or variant_map is None
    ):
        print("  route selections: MISSING")
        return None, None

    base_probs, variant_probs = trim_pair(base_probs, variant_probs)
    base_map, variant_map = trim_pair(base_map, variant_map)
    route_changed = (base_map != variant_map).any(dim=1)
    entry_match = (base_map == variant_map).float().mean()
    row_match = (~route_changed).float().mean()
    prob_stats = diff_stats(base_probs, variant_probs)
    print(
        f"  route selections: rows={base_map.shape[0]} "
        f"changed={int(route_changed.sum().item())} "
        f"row_match={float(row_match):.6f} entry_match={float(entry_match):.6f} "
        f"prob_max={prob_stats['max_abs_diff']:.6f} "
        f"prob_mean={prob_stats['mean_abs_diff']:.4e}"
    )
    if base_logits is not None and variant_logits is not None:
        print_stats_line("router logits variant-base", variant_logits, base_logits)

    prob_l1 = (variant_probs - base_probs).abs().sum(dim=1)
    values, indices = torch.topk(prob_l1, k=min(top_rows, prob_l1.numel()))
    print("  top route/prob changed rows:")
    print("    row changed prob_l1 baseline_ids variant_ids")
    for value, row_idx in zip(values.tolist(), indices.tolist()):
        print(
            f"    {row_idx:>3d} {str(bool(route_changed[row_idx])):<7s} "
            f"{value:>8.6f} {ids_for_row(base_map, row_idx):<24s} "
            f"{ids_for_row(variant_map, row_idx):<24s}"
        )
    return base_map, variant_map


def print_tensor_report(
    *,
    spec: TensorSpec,
    layer_idx: int,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    megatron: dict[str, Any] | None,
    seq_lens: list[int],
    route_changed: torch.Tensor | None,
    top_rows: int,
) -> None:
    base_tensor, base_selector = tensor_from_entry(
        baseline,
        layer_idx=layer_idx,
        name=spec.base_name,
        outputs=spec.base_outputs,
        selectors=spec.base_selectors,
        seq_lens=seq_lens,
    )
    variant_tensor, variant_selector = tensor_from_entry(
        variant,
        layer_idx=layer_idx,
        name=spec.base_name,
        outputs=spec.base_outputs,
        selectors=spec.base_selectors,
        seq_lens=seq_lens,
    )
    if base_tensor is None or variant_tensor is None:
        print(
            f"  {spec.label}: MISSING base={base_selector} variant={variant_selector}"
        )
        return
    print_stats_line(f"{spec.label} variant-base", variant_tensor, base_tensor)
    if megatron is None or spec.target_name is None:
        return

    target_tensor, target_selector = tensor_from_entry(
        megatron,
        layer_idx=layer_idx,
        name=spec.target_name,
        outputs=spec.target_outputs,
        selectors=spec.target_selectors,
        seq_lens=seq_lens,
    )
    if target_tensor is None:
        print(f"    target missing selector={target_selector}")
        return
    base_tensor, variant_tensor, target_tensor = trim_triple(
        base_tensor, variant_tensor, target_tensor
    )
    base_error = (base_tensor.float() - target_tensor.float()).abs()
    variant_error = (variant_tensor.float() - target_tensor.float()).abs()
    delta = variant_error - base_error
    print_stats_line(f"{spec.label} baseline-target", base_tensor, target_tensor)
    print_stats_line(f"{spec.label} variant-target", variant_tensor, target_tensor)
    flat_delta = delta.reshape(-1)
    print(
        f"    error delta variant-baseline: mean={float(flat_delta.mean()):.4e} "
        f"max_worse={float(flat_delta.max()):.6f} "
        f"max_better={float((-flat_delta).max()):.6f} "
        f"worse_frac={float((flat_delta > 0).float().mean()):.6f}"
    )

    row_delta = row_mean_abs(delta)
    row_base = row_mean_abs(base_error)
    row_variant = row_mean_abs(variant_error)
    row_diff = row_mean_abs(variant_tensor - base_tensor)
    row_max_diff = row_max_abs(variant_tensor - base_tensor)
    values, indices = torch.topk(row_delta, k=min(top_rows, row_delta.numel()))
    print(f"    top rows where variant worsens {spec.label}:")
    print("      row changed delta_mean base_mean variant_mean vv_mean vv_max")
    for value, row_idx in zip(values.tolist(), indices.tolist()):
        changed = (
            "?"
            if route_changed is None or row_idx >= route_changed.numel()
            else str(bool(route_changed[row_idx]))
        )
        print(
            f"      {row_idx:>3d} {changed:<7s} {value:>10.4e} "
            f"{float(row_base[row_idx]):>10.4e} "
            f"{float(row_variant[row_idx]):>10.4e} "
            f"{float(row_diff[row_idx]):>10.4e} "
            f"{float(row_max_diff[row_idx]):>8.6f}"
        )


def main() -> None:
    args = parse_args()
    baseline = load_payload(args.baseline)
    variant = load_payload(args.variant)
    megatron = load_payload(args.megatron) if args.megatron else None
    seq_lens = baseline.get("seq_lens") or variant.get("seq_lens") or []
    layers = parse_layers(args.layers)

    print("=" * 100)
    print(f"baseline      : {args.baseline_label} {args.baseline}")
    print(f"variant       : {args.variant_label} {args.variant}")
    print(f"megatron      : {args.megatron or 'None'}")
    print(f"seq_lens      : {seq_lens}")
    print(f"layers        : {layers}")
    print("=" * 100)

    for layer_idx in layers:
        print("\n" + "=" * 100)
        print(f"Layer {layer_idx}")
        print("=" * 100)
        base_entry, _ = layer_entry_stream(baseline, layer_idx, seq_lens, is_vllm=True)
        variant_entry, _ = layer_entry_stream(
            variant, layer_idx, seq_lens, is_vllm=True
        )
        if base_entry is not None and variant_entry is not None:
            print_stats_line("layer entry variant-base", variant_entry, base_entry)
        if megatron is not None:
            target_entry, _ = layer_entry_stream(
                megatron, layer_idx, seq_lens, is_vllm=False
            )
            if (
                base_entry is not None
                and variant_entry is not None
                and target_entry is not None
            ):
                print_stats_line(
                    "layer entry baseline-target", base_entry, target_entry
                )
                print_stats_line(
                    "layer entry variant-target", variant_entry, target_entry
                )

        base_map, variant_map = print_router_report(
            layer_idx=layer_idx,
            baseline=baseline,
            variant=variant,
            seq_lens=seq_lens,
            top_rows=args.top_rows,
        )
        route_changed = None
        if base_map is not None and variant_map is not None:
            base_map, variant_map = trim_pair(base_map, variant_map)
            route_changed = (base_map != variant_map).any(dim=1)

        for spec in G_TENSOR_SPECS:
            print_tensor_report(
                spec=spec,
                layer_idx=layer_idx,
                baseline=baseline,
                variant=variant,
                megatron=megatron,
                seq_lens=seq_lens,
                route_changed=route_changed,
                top_rows=args.top_rows,
            )


if __name__ == "__main__":
    main()
