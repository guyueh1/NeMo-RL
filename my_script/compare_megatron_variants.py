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

"""Compare two Megatron captures and optional vLLM targets."""

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
)


@dataclass(frozen=True)
class TensorSpec:
    label: str
    megatron_name: str
    megatron_outputs: bool
    megatron_selectors: tuple[str, ...]
    vllm_name: str | None = None
    vllm_outputs: bool = True
    vllm_selectors: tuple[str, ...] = ()


G_TENSOR_SPECS = (
    TensorSpec(
        label="mlp output",
        megatron_name="mlp",
        megatron_outputs=True,
        megatron_selectors=("item0", "output", "first"),
        vllm_name="mixer",
        vllm_outputs=True,
        vllm_selectors=("output", "item0", "first"),
    ),
    TensorSpec(
        label="raw router logits",
        megatron_name="mlp.router.gating",
        megatron_outputs=True,
        megatron_selectors=("output", "item0", "first"),
        vllm_name="mixer.gate",
        vllm_outputs=True,
        vllm_selectors=("output", "item0", "first"),
    ),
    TensorSpec(
        label="router sparse probs",
        megatron_name="mlp.router",
        megatron_outputs=True,
        megatron_selectors=("item0", "output", "first"),
    ),
    TensorSpec(
        label="shared fc2 output",
        megatron_name="mlp.shared_experts.linear_fc2",
        megatron_outputs=True,
        megatron_selectors=("output", "item0", "first"),
        vllm_name="mixer.shared_experts.down_proj",
        vllm_outputs=True,
        vllm_selectors=("output", "item0", "first"),
    ),
    TensorSpec(
        label="packed experts output",
        megatron_name="mlp.experts",
        megatron_outputs=True,
        megatron_selectors=("output", "item0", "first"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline Megatron capture.")
    parser.add_argument("--variant", required=True, help="Variant Megatron capture.")
    parser.add_argument(
        "--vllm",
        default=None,
        help="Optional vLLM capture used as the target for error deltas.",
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--variant-label", default="variant")
    parser.add_argument(
        "--layers",
        default="1-5,15-18",
        help="Layer list/ranges to inspect, for example '1-5,15-18'.",
    )
    parser.add_argument(
        "--top-rows",
        type=int,
        default=10,
        help="Rows/prompts to print for outlier reports.",
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


def print_stats_line(label: str, a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = trim_pair(a, b)
    stats = diff_stats(a, b)
    print(
        f"  {label:<36s} max={stats['max_abs_diff']:.6f} "
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


def top_error_delta_rows(
    *,
    label: str,
    baseline: torch.Tensor,
    variant: torch.Tensor,
    target: torch.Tensor,
    top_rows: int,
    route_changed: torch.Tensor | None = None,
) -> None:
    baseline, variant, target = trim_triple(baseline, variant, target)
    baseline_error = (baseline.float() - target.float()).abs()
    variant_error = (variant.float() - target.float()).abs()
    delta = variant_error - baseline_error
    flat_delta = delta.reshape(-1)
    print(
        f"    {label} error delta variant-baseline: "
        f"mean={float(flat_delta.mean()):.4e} "
        f"max_worse={float(flat_delta.max()):.6f} "
        f"max_better={float((-flat_delta).max()):.6f} "
        f"worse_frac={float((flat_delta > 0).float().mean()):.6f}"
    )

    row_delta = row_mean_abs(delta)
    row_base = row_mean_abs(baseline_error)
    row_variant = row_mean_abs(variant_error)
    row_direct = row_mean_abs(variant - baseline)
    row_direct_max = row_max_abs(variant - baseline)
    values, indices = torch.topk(row_delta, k=min(top_rows, row_delta.numel()))
    print(f"    top rows where variant worsens {label}:")
    print("      row changed delta_mean base_mean variant_mean vb_mean vb_max")
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
            f"{float(row_direct[row_idx]):>10.4e} "
            f"{float(row_direct_max[row_idx]):>8.6f}"
        )


def selected_tensor(
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


def ids_for_row(route_map: torch.Tensor, row_idx: int) -> str:
    ids = route_map[row_idx].nonzero(as_tuple=False).flatten().tolist()
    return ",".join(str(int(idx)) for idx in ids)


def megatron_routing(
    payload: dict[str, Any], layer_idx: int, seq_lens: list[int]
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    entry = get_layer_entry(payload, layer_idx, "mlp.router", outputs=True)
    probs, _ = select_tensor(entry, ("item0", "output", "first"))
    route_map, _ = select_tensor(entry, ("item1",))
    if probs is None:
        return None, None
    probs = normalize_router_layout(probs, seq_lens).float()
    if route_map is None:
        route_map = probs != 0
    else:
        route_map = normalize_router_layout(route_map, seq_lens).bool()
    return probs, route_map


def print_router_report(
    *,
    layer_idx: int,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    seq_lens: list[int],
    top_rows: int,
) -> torch.Tensor | None:
    baseline_probs, baseline_map = megatron_routing(baseline, layer_idx, seq_lens)
    variant_probs, variant_map = megatron_routing(variant, layer_idx, seq_lens)
    if (
        baseline_probs is None
        or baseline_map is None
        or variant_probs is None
        or variant_map is None
    ):
        print("  route selections: MISSING")
        return None

    baseline_probs, variant_probs = trim_pair(baseline_probs, variant_probs)
    baseline_map, variant_map = trim_pair(baseline_map, variant_map)
    route_changed = (baseline_map != variant_map).any(dim=1)
    row_match = (~route_changed).float().mean()
    entry_match = (baseline_map == variant_map).float().mean()
    prob_stats = diff_stats(variant_probs, baseline_probs)
    print(
        f"  route selections baseline-variant: rows={baseline_map.shape[0]} "
        f"changed={int(route_changed.sum().item())} "
        f"row_match={float(row_match):.6f} entry_match={float(entry_match):.6f} "
        f"prob_max={prob_stats['max_abs_diff']:.6f} "
        f"prob_mean={prob_stats['mean_abs_diff']:.4e}"
    )

    prob_l1 = (variant_probs - baseline_probs).abs().sum(dim=1)
    values, indices = torch.topk(prob_l1, k=min(top_rows, prob_l1.numel()))
    print("  top route/prob changed rows:")
    print("    row changed prob_l1 baseline_ids variant_ids")
    for value, row_idx in zip(values.tolist(), indices.tolist()):
        print(
            f"    {row_idx:>3d} {str(bool(route_changed[row_idx])):<7s} "
            f"{value:>8.6f} {ids_for_row(baseline_map, row_idx):<24s} "
            f"{ids_for_row(variant_map, row_idx):<24s}"
        )
    return route_changed


def compute_logprobs(payload: dict[str, Any], vocab: int | None = None) -> torch.Tensor:
    logits = payload.get("last_token_logits")
    if not isinstance(logits, torch.Tensor):
        raise KeyError("Megatron payload is missing last_token_logits")
    logits = logits.float()
    if vocab is not None:
        logits = logits[..., :vocab]
    return torch.log_softmax(logits, dim=-1)


def print_final_report(
    *,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    vllm: dict[str, Any] | None,
    top_rows: int,
) -> None:
    print("\nFinal next-token logprob report")
    if vllm is not None and isinstance(vllm.get("next_token_logprobs"), torch.Tensor):
        target = vllm["next_token_logprobs"].float()
        vocab = target.shape[-1]
    else:
        target = None
        vocab = None
    baseline_logprobs = compute_logprobs(baseline, vocab=vocab)
    variant_logprobs = compute_logprobs(variant, vocab=vocab)
    print_stats_line("variant-baseline", variant_logprobs, baseline_logprobs)
    if target is None:
        return
    print_stats_line("baseline-vllm", baseline_logprobs, target)
    print_stats_line("variant-vllm", variant_logprobs, target)
    top_error_delta_rows(
        label="final logprob",
        baseline=baseline_logprobs,
        variant=variant_logprobs,
        target=target,
        top_rows=top_rows,
    )


def print_layer_entry_report(
    *,
    layer_idx: int,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    vllm: dict[str, Any] | None,
    seq_lens: list[int],
    top_rows: int,
) -> None:
    baseline_entry, baseline_selector = layer_entry_stream(
        baseline, layer_idx, seq_lens, is_vllm=False
    )
    variant_entry, variant_selector = layer_entry_stream(
        variant, layer_idx, seq_lens, is_vllm=False
    )
    if baseline_entry is None or variant_entry is None:
        print(
            "  layer entry: MISSING "
            f"baseline={baseline_selector} variant={variant_selector}"
        )
        return
    print_stats_line("layer entry variant-baseline", variant_entry, baseline_entry)
    if vllm is None:
        return
    target_entry, target_selector = layer_entry_stream(
        vllm, layer_idx, seq_lens, is_vllm=True
    )
    if target_entry is None:
        print(f"    vLLM layer entry missing selector={target_selector}")
        return
    print_stats_line("layer entry baseline-vllm", baseline_entry, target_entry)
    print_stats_line("layer entry variant-vllm", variant_entry, target_entry)
    top_error_delta_rows(
        label="layer entry",
        baseline=baseline_entry,
        variant=variant_entry,
        target=target_entry,
        top_rows=top_rows,
    )


def print_tensor_report(
    *,
    spec: TensorSpec,
    layer_idx: int,
    baseline: dict[str, Any],
    variant: dict[str, Any],
    vllm: dict[str, Any] | None,
    seq_lens: list[int],
    route_changed: torch.Tensor | None,
    top_rows: int,
) -> None:
    baseline_tensor, baseline_selector = selected_tensor(
        baseline,
        layer_idx=layer_idx,
        name=spec.megatron_name,
        outputs=spec.megatron_outputs,
        selectors=spec.megatron_selectors,
        seq_lens=seq_lens,
    )
    variant_tensor, variant_selector = selected_tensor(
        variant,
        layer_idx=layer_idx,
        name=spec.megatron_name,
        outputs=spec.megatron_outputs,
        selectors=spec.megatron_selectors,
        seq_lens=seq_lens,
    )
    if baseline_tensor is None or variant_tensor is None:
        print(
            f"  {spec.label}: MISSING baseline={baseline_selector} "
            f"variant={variant_selector}"
        )
        return

    print_stats_line(f"{spec.label} variant-baseline", variant_tensor, baseline_tensor)
    if vllm is None or spec.vllm_name is None:
        return
    target_tensor, target_selector = selected_tensor(
        vllm,
        layer_idx=layer_idx,
        name=spec.vllm_name,
        outputs=spec.vllm_outputs,
        selectors=spec.vllm_selectors,
        seq_lens=seq_lens,
    )
    if target_tensor is None:
        print(f"    vLLM target missing selector={target_selector}")
        return
    print_stats_line(f"{spec.label} baseline-vllm", baseline_tensor, target_tensor)
    print_stats_line(f"{spec.label} variant-vllm", variant_tensor, target_tensor)
    top_error_delta_rows(
        label=spec.label,
        baseline=baseline_tensor,
        variant=variant_tensor,
        target=target_tensor,
        top_rows=top_rows,
        route_changed=route_changed,
    )


def main() -> None:
    args = parse_args()
    baseline = load_payload(args.baseline)
    variant = load_payload(args.variant)
    vllm = load_payload(args.vllm) if args.vllm else None
    seq_lens = baseline.get("seq_lens") or variant.get("seq_lens") or []
    layers = parse_layers(args.layers)

    print("=" * 100)
    print(f"baseline      : {args.baseline_label} {args.baseline}")
    print(f"variant       : {args.variant_label} {args.variant}")
    print(f"vllm target   : {args.vllm or 'None'}")
    print(f"seq_lens      : {seq_lens}")
    print(f"layers        : {layers}")
    print("=" * 100)

    print_final_report(
        baseline=baseline,
        variant=variant,
        vllm=vllm,
        top_rows=args.top_rows,
    )

    for layer_idx in layers:
        print("\n" + "=" * 100)
        print(f"Layer {layer_idx}")
        print("=" * 100)
        print_layer_entry_report(
            layer_idx=layer_idx,
            baseline=baseline,
            variant=variant,
            vllm=vllm,
            seq_lens=seq_lens,
            top_rows=args.top_rows,
        )
        route_changed = print_router_report(
            layer_idx=layer_idx,
            baseline=baseline,
            variant=variant,
            seq_lens=seq_lens,
            top_rows=args.top_rows,
        )
        for spec in G_TENSOR_SPECS:
            print_tensor_report(
                spec=spec,
                layer_idx=layer_idx,
                baseline=baseline,
                variant=variant,
                vllm=vllm,
                seq_lens=seq_lens,
                route_changed=route_changed,
                top_rows=args.top_rows,
            )


if __name__ == "__main__":
    main()
