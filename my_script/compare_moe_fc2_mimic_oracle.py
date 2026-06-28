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

"""Compare live Megatron vLLM-fc2 mimic output against exact vLLM oracle fc2."""

from __future__ import annotations

import argparse
import os

import torch
from compare import diff_stats, get_layer_entry, normalize_router_layout, select_tensor
from compare_vllm_moe_internals import (
    combine_expert_major,
    combine_expert_major_fp32,
    route_indices_from_router_map,
    squeeze_singletons,
    vllm_flat_indices_in_megatron_order,
)

DEFAULT_SESSION_DIR = os.path.join(
    "session",
    "20260625_121743",
    "nemotron_bf16_no_bi_triton_single_prompt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--mimic-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router_vllm_fc2_mimic_layer1_layers0_5_15_18.pt",
        ),
    )
    parser.add_argument(
        "--vllm-moe-internals",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "vllm_capture_nemotron3_nano_reference_conv_layer0_norm_router_scale_moe_internals.pt.moe_internals.pt",
        ),
    )
    parser.add_argument(
        "--router-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router.pt",
        ),
    )
    parser.add_argument(
        "--baseline-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router.pt",
        ),
    )
    parser.add_argument(
        "--oracle-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router_oracle_vllm_fc2_layer1_layers0_5_15_18.pt",
        ),
    )
    parser.add_argument("--topn", type=int, default=8)
    return parser.parse_args()


def print_stats(label, actual, expected):
    if actual.shape != expected.shape:
        print(f"{label:<56s} SHAPE {tuple(actual.shape)} vs {tuple(expected.shape)}")
        return None
    stats = diff_stats(actual, expected)
    print(
        f"{label:<56s} max={stats['max_abs_diff']:.6f} "
        f"mean={stats['mean_abs_diff']:.6e} cos={stats['cos_sim']:.9f} "
        f"shape={tuple(actual.shape)}"
    )
    return stats


def load_capture(path):
    print(f"[mimic-oracle] loading {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_packed_expert_output(payload, layer):
    candidates = (
        ("mlp.experts", ("output", "item0", "first")),
        ("mlp.experts.linear_fc2", ("output", "item0", "first")),
    )
    for module_name, selectors in candidates:
        tensor, selector = select_tensor(
            get_layer_entry(payload, layer, module_name, outputs=True),
            selectors,
        )
        if isinstance(tensor, torch.Tensor):
            return squeeze_singletons(tensor).contiguous(), module_name, selector
    raise KeyError(f"missing layer {layer} packed expert output")


def load_router_map(router_capture, layer):
    router_map, selector = select_tensor(
        get_layer_entry(router_capture, layer, "mlp.router", outputs=True),
        ("item1",),
    )
    if router_map is None:
        raise KeyError(f"missing layer {layer} router map")
    router_map = normalize_router_layout(router_map, router_capture["seq_lens"]).bool()
    return router_map, selector


def load_vllm_oracle_fc2(vllm_moe_path, router_capture, layer):
    vllm_moe = load_capture(vllm_moe_path)
    kernel_calls = vllm_moe.get("moe_kernel_calls", [])
    activation_calls = vllm_moe.get("moe_activation_calls", [])
    if len(kernel_calls) < 2 or not activation_calls:
        raise RuntimeError(
            "vLLM MoE internals must contain fc1/fc2 kernel calls and activation"
        )
    router_map, router_selector = load_router_map(router_capture, layer)
    num_flat_rows = activation_calls[0]["output"].shape[0]
    ordered_indices = vllm_flat_indices_in_megatron_order(
        kernel_calls[0],
        router_map,
        num_flat_rows,
    )
    fc2_output = (
        kernel_calls[1]["C_after"]
        .reshape(num_flat_rows, -1)[ordered_indices]
        .contiguous()
    )
    return fc2_output, router_map, router_selector, vllm_moe


def print_top_outliers(label, actual, expected, router_map, *, topn):
    if actual.shape != expected.shape:
        return
    token_indices, expert_indices = route_indices_from_router_map(router_map)
    diff = (actual.float() - expected.float()).abs()
    values, flat_indices = torch.topk(diff.flatten(), k=min(topn, diff.numel()))
    hidden = actual.shape[-1]
    print(f"[mimic-oracle] top {label} outliers")
    for rank, (value, flat_index) in enumerate(
        zip(values.tolist(), flat_indices.tolist()),
        start=1,
    ):
        row = flat_index // hidden
        col = flat_index % hidden
        print(
            "  "
            f"#{rank} row={row} token={int(token_indices[row])} "
            f"expert={int(expert_indices[row])} hidden={col} "
            f"diff={value:.6f} actual={float(actual[row, col]):.6f} "
            f"expected={float(expected[row, col]):.6f}"
        )


def compare_optional_capture(label, path, oracle_fc2, router_map, layer):
    if not path or not os.path.exists(path):
        print(f"[mimic-oracle] optional {label} capture missing: {path}")
        return None
    payload = load_capture(path)
    output, module_name, selector = load_packed_expert_output(payload, layer)
    print(
        f"[mimic-oracle] {label} selector module={module_name} "
        f"selector={selector} shape={tuple(output.shape)}"
    )
    print_stats(f"{label} packed fc2 vs oracle fc2", output, oracle_fc2)
    print_top_outliers(
        f"{label} packed-vs-oracle",
        output,
        oracle_fc2,
        router_map,
        topn=4,
    )
    return output


def print_route_reduction_stats(prefix, actual_fc2, expected_fc2, router_map):
    token_indices, _ = route_indices_from_router_map(router_map)
    num_tokens = router_map.shape[0]
    actual_bf16 = combine_expert_major(
        actual_fc2,
        token_indices,
        num_tokens,
        torch.bfloat16,
    )
    expected_bf16 = combine_expert_major(
        expected_fc2,
        token_indices,
        num_tokens,
        torch.bfloat16,
    )
    actual_fp32 = combine_expert_major_fp32(
        actual_fc2,
        token_indices,
        num_tokens,
        torch.bfloat16,
    )
    expected_fp32 = combine_expert_major_fp32(
        expected_fc2,
        token_indices,
        num_tokens,
        torch.bfloat16,
    )
    print_stats(f"{prefix} route bf16 combine", actual_bf16, expected_bf16)
    print_stats(f"{prefix} route fp32 combine", actual_fp32, expected_fp32)


def main():
    args = parse_args()
    mimic = load_capture(args.mimic_capture)
    router_capture = load_capture(args.router_capture)
    oracle_fc2, router_map, router_selector, vllm_moe = load_vllm_oracle_fc2(
        args.vllm_moe_internals,
        router_capture,
        args.layer,
    )
    mimic_fc2, mimic_module, mimic_selector = load_packed_expert_output(
        mimic,
        args.layer,
    )

    print(
        f"[mimic-oracle] layer={args.layer} router_selector={router_selector} "
        f"router_shape={tuple(router_map.shape)}"
    )
    print(
        f"[mimic-oracle] vllm kernels={len(vllm_moe.get('moe_kernel_calls', []))} "
        f"activations={len(vllm_moe.get('moe_activation_calls', []))}"
    )
    print(f"[mimic-oracle] mimic metadata={mimic.get('moe_vllm_fc2_mimic')}")
    print(
        f"[mimic-oracle] mimic selector module={mimic_module} "
        f"selector={mimic_selector} shape={tuple(mimic_fc2.shape)}"
    )

    print_stats("live mimic packed fc2 vs oracle fc2", mimic_fc2, oracle_fc2)
    print_route_reduction_stats(
        "live mimic vs oracle",
        mimic_fc2,
        oracle_fc2,
        router_map,
    )
    print_top_outliers(
        "live mimic packed-vs-oracle",
        mimic_fc2,
        oracle_fc2,
        router_map,
        topn=args.topn,
    )

    baseline_fc2 = compare_optional_capture(
        "baseline",
        args.baseline_capture,
        oracle_fc2,
        router_map,
        args.layer,
    )
    if baseline_fc2 is not None:
        print_stats(
            "live mimic packed fc2 vs baseline packed fc2",
            mimic_fc2,
            baseline_fc2,
        )
        print_route_reduction_stats(
            "live mimic vs baseline",
            mimic_fc2,
            baseline_fc2,
            router_map,
        )

    oracle_capture_fc2 = compare_optional_capture(
        "oracle_capture",
        args.oracle_capture,
        oracle_fc2,
        router_map,
        args.layer,
    )
    if oracle_capture_fc2 is not None:
        print_stats(
            "live mimic packed fc2 vs oracle-capture packed fc2",
            mimic_fc2,
            oracle_capture_fc2,
        )


if __name__ == "__main__":
    main()
