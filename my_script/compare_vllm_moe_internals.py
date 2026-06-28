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

"""Compare captured vLLM routed-MoE internals against Megatron expert hooks."""

from __future__ import annotations

import argparse
import os

import torch
from compare import diff_stats, get_layer_entry, normalize_router_layout, select_tensor

DEFAULT_SESSION_DIR = os.path.join(
    "session",
    "20260625_121743",
    "nemotron_bf16_no_bi_triton_single_prompt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-moe-internals",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "vllm_capture_nemotron3_nano_reference_conv_layer0_norm_router_scale_moe_internals.pt.moe_internals.pt",
        ),
    )
    parser.add_argument(
        "--vllm-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "vllm_capture_nemotron3_nano_reference_conv_layer0_norm_router_scale_moe_internals.pt",
        ),
    )
    parser.add_argument(
        "--megatron-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router.pt",
        ),
    )
    parser.add_argument("--layer", type=int, default=1)
    return parser.parse_args()


def squeeze_singletons(tensor):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    while tensor.dim() >= 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor.contiguous()


def load_module_tensor(payload, layer_idx, module_name, selectors, *, outputs=False):
    tensor, selector = select_tensor(
        get_layer_entry(payload, layer_idx, module_name, outputs=outputs),
        selectors,
    )
    if tensor is None:
        raise KeyError(
            f"missing layer={layer_idx} module={module_name!r} "
            f"selectors={selectors} outputs={outputs}"
        )
    return squeeze_singletons(tensor), selector


def print_stats(label, actual, expected):
    if actual.shape != expected.shape:
        print(f"{label:<48s} SHAPE {tuple(actual.shape)} vs {tuple(expected.shape)}")
        return
    stats = diff_stats(actual, expected)
    print(
        f"{label:<48s} max={stats['max_abs_diff']:.6f} "
        f"mean={stats['mean_abs_diff']:.6e} cos={stats['cos_sim']:.9f} "
        f"shape={tuple(actual.shape)}"
    )


def valid_sorted_flat_indices(call, num_flat_rows):
    sorted_token_ids = call["sorted_token_ids"].to(torch.long).flatten()
    return sorted_token_ids[sorted_token_ids < num_flat_rows].contiguous()


def vllm_flat_indices_in_megatron_order(call, router_map, num_flat_rows):
    top_k = int(call["top_k"])
    block_size = int(call["config"]["BLOCK_SIZE_M"])
    sorted_token_ids = call["sorted_token_ids"].to(torch.long).flatten()
    expert_ids = call["expert_ids"].to(torch.long).flatten()
    flat_by_token_expert = {}
    for block_idx, expert_id in enumerate(expert_ids.tolist()):
        if expert_id < 0:
            continue
        start = block_idx * block_size
        end = min(start + block_size, sorted_token_ids.numel())
        for flat_idx in sorted_token_ids[start:end].tolist():
            if flat_idx >= num_flat_rows:
                continue
            token_idx = flat_idx // top_k
            flat_by_token_expert[(int(token_idx), int(expert_id))] = int(flat_idx)

    num_tokens = router_map.shape[0]
    expert_major_flat = router_map.T.contiguous().reshape(-1).nonzero().flatten()
    ordered = []
    missing = []
    for flat in expert_major_flat.tolist():
        token_idx = flat % num_tokens
        expert_idx = flat // num_tokens
        mapped = flat_by_token_expert.get((int(token_idx), int(expert_idx)))
        if mapped is None:
            missing.append((int(token_idx), int(expert_idx)))
        else:
            ordered.append(mapped)
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        raise RuntimeError(
            f"missing {len(missing)} token/expert pairs in vLLM sorted metadata: "
            f"{preview}"
        )
    return torch.tensor(ordered, dtype=torch.long)


def token_indices_from_router_map(router_map):
    num_tokens = router_map.shape[0]
    expert_major_flat = router_map.T.contiguous().reshape(-1).nonzero().flatten()
    return (expert_major_flat % num_tokens).to(torch.long).contiguous()


def route_indices_from_router_map(router_map):
    num_tokens = router_map.shape[0]
    expert_major_flat = router_map.T.contiguous().reshape(-1).nonzero().flatten()
    token_indices = (expert_major_flat % num_tokens).to(torch.long).contiguous()
    expert_indices = (expert_major_flat // num_tokens).to(torch.long).contiguous()
    return token_indices, expert_indices


def combine_expert_major(packed_values, token_indices, num_tokens, dtype):
    combined = torch.zeros(num_tokens, packed_values.shape[-1], dtype=dtype)
    packed_values = packed_values.to(dtype).cpu()
    for row_idx, token_idx in enumerate(token_indices.tolist()):
        combined[token_idx] = (combined[token_idx] + packed_values[row_idx]).to(dtype)
    return combined


def combine_expert_major_fp32(packed_values, token_indices, num_tokens, output_dtype):
    combined = torch.zeros(num_tokens, packed_values.shape[-1], dtype=torch.float32)
    packed_values = packed_values.float().cpu()
    for row_idx, token_idx in enumerate(token_indices.tolist()):
        combined[token_idx] = combined[token_idx] + packed_values[row_idx]
    return combined.to(output_dtype)


def reduce_topk_bf16_sequential(values, *, reverse=False):
    values = values.to(torch.bfloat16).cpu()
    output = torch.zeros(values.shape[0], values.shape[-1], dtype=torch.bfloat16)
    indices = range(values.shape[1] - 1, -1, -1) if reverse else range(values.shape[1])
    for topk_idx in indices:
        output = (output + values[:, topk_idx, :]).to(torch.bfloat16)
    return output


def reduce_topk_fp32(values):
    return values.float().sum(dim=1).to(torch.bfloat16)


def replay_vllm_moe_sum_cuda(values):
    if not torch.cuda.is_available():
        return None

    import vllm._custom_ops as ops

    values_cuda = values.contiguous().cuda()
    output = torch.empty(
        values_cuda.shape[0],
        values_cuda.shape[-1],
        device=values_cuda.device,
        dtype=values_cuda.dtype,
    )
    ops.moe_sum(values_cuda, output)
    torch.cuda.synchronize()
    return output.cpu()


def add_then_cast(lhs, rhs, *, accumulate_dtype):
    lhs = lhs.to(accumulate_dtype).cpu()
    rhs = rhs.to(accumulate_dtype).cpu()
    return (lhs + rhs).to(torch.bfloat16)


def print_route_fc2_outliers(
    actual,
    expected,
    actual_input,
    expected_input,
    token_indices,
    expert_indices,
    *,
    topn=8,
):
    diff = (actual.float() - expected.float()).abs()
    values, flat_indices = torch.topk(diff.flatten(), k=min(topn, diff.numel()))
    hidden = actual.shape[-1]
    print("[moe-internals] top route-fc2 output outliers")
    for rank, (value, flat_index) in enumerate(
        zip(values.tolist(), flat_indices.tolist()),
        start=1,
    ):
        row = flat_index // hidden
        col = flat_index % hidden
        input_diff = (actual_input[row].float() - expected_input[row].float()).abs()
        print(
            "  "
            f"#{rank} row={row} token={int(token_indices[row])} "
            f"expert={int(expert_indices[row])} hidden={col} "
            f"diff={value:.6f} v={float(actual[row, col]):.6f} "
            f"m={float(expected[row, col]):.6f} "
            f"input_row_max={float(input_diff.max()):.6f} "
            f"input_row_mean={float(input_diff.mean()):.6e}"
        )


def print_fc2_input_outliers(
    actual,
    expected,
    raw_actual,
    packed_probs,
    token_indices,
    expert_indices,
    *,
    topn=8,
):
    diff = (actual.float() - expected.float()).abs()
    values, flat_indices = torch.topk(diff.flatten(), k=min(topn, diff.numel()))
    hidden = actual.shape[-1]
    print("[moe-internals] top fc2-input outliers")
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
            f"diff={value:.6f} v_weighted={float(actual[row, col]):.6f} "
            f"m={float(expected[row, col]):.6f} "
            f"v_raw={float(raw_actual[row, col]):.6f} "
            f"prob={float(packed_probs[row]):.6f}"
        )


def print_final_moe_outliers(
    v_final,
    m_final,
    v_route,
    m_route,
    v_shared,
    m_shared,
    *,
    topn=8,
):
    diff = (v_final.float() - m_final.float()).abs()
    values, flat_indices = torch.topk(diff.flatten(), k=min(topn, diff.numel()))
    hidden = v_final.shape[-1]
    print("[moe-internals] top final-MoE output outliers")
    for rank, (value, flat_index) in enumerate(
        zip(values.tolist(), flat_indices.tolist()),
        start=1,
    ):
        token = flat_index // hidden
        col = flat_index % hidden
        print(
            "  "
            f"#{rank} token={token} hidden={col} diff={value:.6f} "
            f"v_final={float(v_final[token, col]):.6f} "
            f"m_final={float(m_final[token, col]):.6f} "
            f"v_route={float(v_route[token, col]):.6f} "
            f"m_route={float(m_route[token, col]):.6f} "
            f"v_shared={float(v_shared[token, col]):.6f} "
            f"m_shared={float(m_shared[token, col]):.6f}"
        )


def main():
    args = parse_args()
    vllm = torch.load(args.vllm_moe_internals, map_location="cpu", weights_only=False)
    vllm_capture = torch.load(args.vllm_capture, map_location="cpu", weights_only=False)
    megatron = torch.load(args.megatron_capture, map_location="cpu", weights_only=False)

    kernel_calls = vllm.get("moe_kernel_calls", [])
    activation_calls = vllm.get("moe_activation_calls", [])
    print(
        f"[moe-internals] saved kernel={len(kernel_calls)} "
        f"total_kernel={vllm.get('moe_kernel_total_calls')} "
        f"saved_activation={len(activation_calls)} "
        f"total_activation={vllm.get('moe_activation_total_calls')}"
    )
    print(f"[moe-internals] modules={vllm.get('moe_modules', [])[:3]}")

    if len(kernel_calls) < 2 or not activation_calls:
        raise RuntimeError("Need at least fc1/fc2 kernel calls and activation capture")

    fc1_call = kernel_calls[0]
    fc2_call = kernel_calls[1]
    activation_call = activation_calls[0]

    print(
        "[moe-internals] fc1 "
        f"A={tuple(fc1_call['A'].shape)} C={tuple(fc1_call['C_after'].shape)} "
        f"top_k={fc1_call['top_k']} mul={fc1_call['mul_routed_weight']}"
    )
    print(
        "[moe-internals] act "
        f"in={tuple(activation_call['input'].shape)} "
        f"out={tuple(activation_call['output'].shape)} "
        f"activation={activation_call['activation']}"
    )
    print(
        "[moe-internals] fc2 "
        f"A={tuple(fc2_call['A'].shape)} C={tuple(fc2_call['C_after'].shape)} "
        f"top_k={fc2_call['top_k']} mul={fc2_call['mul_routed_weight']}"
    )

    num_flat_rows = activation_call["output"].shape[0]
    valid_indices = valid_sorted_flat_indices(fc1_call, num_flat_rows)
    v_fc1_expert_major = fc1_call["C_after"].reshape(num_flat_rows, -1)[valid_indices]
    v_activation_input_expert_major = activation_call["input"][valid_indices]
    v_activation_expert_major = activation_call["output"][valid_indices]
    v_fc2_expert_major = fc2_call["C_after"].reshape(num_flat_rows, -1)[valid_indices]

    m_fc1_output, m_fc1_output_selector = load_module_tensor(
        megatron,
        args.layer,
        "mlp.experts.linear_fc1",
        ("output", "item0", "first"),
        outputs=True,
    )
    m_fc2_input, m_fc2_input_selector = load_module_tensor(
        megatron,
        args.layer,
        "mlp.experts.linear_fc2",
        ("arg0", "first"),
        outputs=False,
    )
    m_fc2_output, m_fc2_output_selector = load_module_tensor(
        megatron,
        args.layer,
        "mlp.experts.linear_fc2",
        ("output", "item0", "first"),
        outputs=True,
    )
    v_moe_output, v_moe_selector = load_module_tensor(
        vllm_capture,
        args.layer,
        "mixer.experts",
        ("output", "first"),
        outputs=True,
    )
    v_shared_output, v_shared_selector = load_module_tensor(
        vllm_capture,
        args.layer,
        "mixer.shared_experts.down_proj",
        ("output", "first"),
        outputs=True,
    )
    m_mlp_output, m_mlp_selector = load_module_tensor(
        megatron,
        args.layer,
        "mlp",
        ("item0", "output", "first"),
        outputs=True,
    )
    m_shared_output, m_shared_selector = load_module_tensor(
        megatron,
        args.layer,
        "mlp.shared_experts.linear_fc2",
        ("output", "item0", "first"),
        outputs=True,
    )
    m_router_map, m_router_selector = select_tensor(
        get_layer_entry(megatron, args.layer, "mlp.router", outputs=True),
        ("item1",),
    )
    m_router_probs, m_router_probs_selector = select_tensor(
        get_layer_entry(megatron, args.layer, "mlp.router", outputs=True),
        ("item0", "output", "first"),
    )
    if m_router_map is None:
        raise KeyError("missing Megatron mlp.router routing map output")
    if m_router_probs is None:
        raise KeyError("missing Megatron mlp.router routing probs output")
    m_router_map = normalize_router_layout(m_router_map, megatron["seq_lens"]).bool()
    m_router_probs = normalize_router_layout(
        m_router_probs,
        megatron["seq_lens"],
    ).float()
    token_indices, expert_indices = route_indices_from_router_map(m_router_map)
    packed_probs = m_router_probs[token_indices, expert_indices].contiguous()

    ordered_indices = vllm_flat_indices_in_megatron_order(
        fc1_call,
        m_router_map,
        num_flat_rows,
    )
    v_fc1_megatron_order = fc1_call["C_after"].reshape(num_flat_rows, -1)[
        ordered_indices
    ]
    v_activation_input_megatron_order = activation_call["input"][ordered_indices]
    v_activation_megatron_order = activation_call["output"][ordered_indices]
    v_fc2_megatron_order = fc2_call["C_after"].reshape(num_flat_rows, -1)[
        ordered_indices
    ]
    v_activation_prob_fp32 = (
        v_activation_megatron_order.float() * packed_probs[:, None].float()
    ).to(torch.bfloat16)
    v_activation_prob_bf16 = (
        v_activation_megatron_order.to(torch.bfloat16)
        * packed_probs[:, None].to(torch.bfloat16)
    ).to(torch.bfloat16)

    print(
        f"[moe-internals] Megatron selectors: fc1_output={m_fc1_output_selector} "
        f"fc2_input={m_fc2_input_selector} fc2_output={m_fc2_output_selector} "
        f"router_probs={m_router_probs_selector} router_map={m_router_selector}"
    )
    print(
        f"[moe-internals] routed selectors: v_moe={v_moe_selector} "
        f"v_shared={v_shared_selector} m_mlp={m_mlp_selector} "
        f"m_shared={m_shared_selector}"
    )
    print_stats(
        "v fc1 output vLLM-sorted vs m fc1 output",
        v_fc1_expert_major,
        m_fc1_output,
    )
    print_stats(
        "v act input vLLM-sorted vs m fc1 output",
        v_activation_input_expert_major,
        m_fc1_output,
    )
    print_stats(
        "v activation vLLM-sorted vs m fc2 input",
        v_activation_expert_major,
        m_fc2_input,
    )
    print_stats(
        "v weighted fc2 vLLM-sorted vs m fc2 output",
        v_fc2_expert_major,
        m_fc2_output,
    )
    print_stats(
        "v fc1 output Megatron-order vs m fc1 output",
        v_fc1_megatron_order,
        m_fc1_output,
    )
    print_stats(
        "v act input Megatron-order vs m fc1 output",
        v_activation_input_megatron_order,
        m_fc1_output,
    )
    print_stats(
        "v activation Megatron-order vs m fc2 input",
        v_activation_megatron_order,
        m_fc2_input,
    )
    print_stats(
        "v activation*prob fp32 vs m fc2 input",
        v_activation_prob_fp32,
        m_fc2_input,
    )
    print_stats(
        "v activation*prob bf16 vs m fc2 input",
        v_activation_prob_bf16,
        m_fc2_input,
    )
    print_stats(
        "v weighted fc2 Megatron-order vs m fc2 output",
        v_fc2_megatron_order,
        m_fc2_output,
    )

    v_fc2_by_token_topk = fc2_call["C_after"].contiguous()
    v_routed_sum = v_fc2_by_token_topk.sum(dim=1).to(torch.bfloat16)
    v_routed_sum_fp32 = reduce_topk_fp32(v_fc2_by_token_topk)
    v_routed_sum_bf16 = reduce_topk_bf16_sequential(v_fc2_by_token_topk)
    v_routed_sum_bf16_reverse = reduce_topk_bf16_sequential(
        v_fc2_by_token_topk,
        reverse=True,
    )
    v_routed_sum_ops = replay_vllm_moe_sum_cuda(v_fc2_by_token_topk)
    v_routed_target = (v_moe_output - v_shared_output).to(torch.bfloat16)
    m_routed_target = (m_mlp_output - m_shared_output).to(torch.bfloat16)
    m_expert_major_sum = combine_expert_major(
        m_fc2_output,
        token_indices,
        m_router_map.shape[0],
        torch.bfloat16,
    )
    m_expert_major_sum_fp32 = combine_expert_major_fp32(
        m_fc2_output,
        token_indices,
        m_router_map.shape[0],
        torch.bfloat16,
    )
    print(
        "[moe-internals] v routed sum "
        f"shape={tuple(v_routed_sum.shape)} dtype={v_routed_sum.dtype} "
        f"max={float(v_routed_sum.float().abs().max()):.6f}"
    )
    print_stats(
        "v torch.sum topk vs v routed target",
        v_routed_sum,
        v_routed_target,
    )
    print_stats(
        "v fp32 topk sum vs v routed target",
        v_routed_sum_fp32,
        v_routed_target,
    )
    print_stats(
        "v bf16 topk sum vs v routed target",
        v_routed_sum_bf16,
        v_routed_target,
    )
    print_stats(
        "v bf16 reverse topk sum vs v routed target",
        v_routed_sum_bf16_reverse,
        v_routed_target,
    )
    if v_routed_sum_ops is None:
        print("[moe-internals] cuda ops.moe_sum replay unavailable")
    else:
        print_stats(
            "v cuda ops.moe_sum vs v routed target",
            v_routed_sum_ops,
            v_routed_target,
        )
        print_stats(
            "v cuda ops.moe_sum vs m routed target",
            v_routed_sum_ops,
            m_routed_target,
        )
    print_stats(
        "v torch.sum topk vs m routed target",
        v_routed_sum,
        m_routed_target,
    )
    print_stats(
        "m expert-major bf16 sum vs m routed target",
        m_expert_major_sum,
        m_routed_target,
    )
    print_stats(
        "m expert-major fp32 sum vs m routed target",
        m_expert_major_sum_fp32,
        m_routed_target,
    )
    print_stats(
        "m expert-major bf16 sum vs v routed target",
        m_expert_major_sum,
        v_routed_target,
    )
    print_stats(
        "m expert-major fp32 sum vs v routed target",
        m_expert_major_sum_fp32,
        v_routed_target,
    )
    print_stats(
        "v moe final vs m moe final",
        v_moe_output,
        m_mlp_output,
    )
    print_stats(
        "v fp32 route+shared vs v moe final",
        add_then_cast(
            v_routed_sum_fp32, v_shared_output, accumulate_dtype=torch.float32
        ),
        v_moe_output,
    )
    print_stats(
        "v bf16 route+shared vs v moe final",
        add_then_cast(
            v_routed_sum_fp32, v_shared_output, accumulate_dtype=torch.bfloat16
        ),
        v_moe_output,
    )
    if v_routed_sum_ops is not None:
        print_stats(
            "v ops route+shared vs v moe final",
            add_then_cast(
                v_routed_sum_ops, v_shared_output, accumulate_dtype=torch.float32
            ),
            v_moe_output,
        )
        print_stats(
            "v ops route+shared vs m moe final",
            add_then_cast(
                v_routed_sum_ops, v_shared_output, accumulate_dtype=torch.float32
            ),
            m_mlp_output,
        )
    print_stats(
        "m fp32 route+shared vs m moe final",
        add_then_cast(
            m_expert_major_sum_fp32, m_shared_output, accumulate_dtype=torch.float32
        ),
        m_mlp_output,
    )
    print_stats(
        "m bf16 route+shared vs m moe final",
        add_then_cast(
            m_expert_major_sum_fp32, m_shared_output, accumulate_dtype=torch.bfloat16
        ),
        m_mlp_output,
    )
    print_stats(
        "m fp32 route+shared vs v moe final",
        add_then_cast(
            m_expert_major_sum_fp32, m_shared_output, accumulate_dtype=torch.float32
        ),
        v_moe_output,
    )
    print_route_fc2_outliers(
        v_fc2_megatron_order,
        m_fc2_output,
        v_activation_megatron_order,
        m_fc2_input,
        token_indices,
        expert_indices,
    )
    print_fc2_input_outliers(
        v_activation_prob_fp32,
        m_fc2_input,
        v_activation_megatron_order,
        packed_probs,
        token_indices,
        expert_indices,
    )
    print_final_moe_outliers(
        v_moe_output,
        m_mlp_output,
        v_routed_sum_fp32,
        m_expert_major_sum_fp32,
        v_shared_output,
        m_shared_output,
    )


if __name__ == "__main__":
    main()
