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

"""Replay a NemotronH Megatron MoE layer from captured vLLM inputs."""

import argparse
import os

import torch
from compare import (
    diff_stats,
    first_tensor,
    get_layer_entry,
    normalize_router_layout,
    normalize_token_layout,
    select_tensor,
)
from compare_vllm_moe_internals import vllm_flat_indices_in_megatron_order
from megatron_forward import MODEL_ALIASES, NEMOTRON3_NANO_MODEL
from replay_mamba_scan import (
    load_megatron_model,
    packed_to_padded,
    padded_to_packed,
    print_stats,
)
from tensor_capture import find_decoder_layers

DEFAULT_SESSION_DIR = os.path.join(
    "session",
    "20260625_121743",
    "nemotron_bf16_no_bi_triton_single_prompt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nemotron3-nano",
        choices=sorted(MODEL_ALIASES) + [NEMOTRON3_NANO_MODEL],
    )
    parser.add_argument(
        "--vllm-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "vllm_capture_nemotron3_nano_reference_conv_layer0_norm_router_scale_layers0_5.pt",
        ),
    )
    parser.add_argument(
        "--megatron-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router_layers0_5.pt",
        ),
    )
    parser.add_argument(
        "--vllm-moe-internals",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "vllm_capture_nemotron3_nano_reference_conv_layer0_norm_router_scale_moe_internals.pt.moe_internals.pt",
        ),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument(
        "--save-vllm-fc2-weight",
        default=None,
        help="Optional path for Megatron layer fc2 weights in vLLM (E, out, in) layout.",
    )
    return parser.parse_args()


def load_selected_tensor(payload, layer_idx, module_name, selectors, *, outputs=False):
    tensor, selector = select_tensor(
        get_layer_entry(payload, layer_idx, module_name, outputs=outputs),
        selectors,
    )
    if tensor is None:
        raise KeyError(
            f"missing tensor layer={layer_idx} module={module_name!r} "
            f"selectors={selectors} outputs={outputs}"
        )
    return tensor, selector


def load_token_tensor(
    payload, layer_idx, module_name, selectors, seq_lens, *, outputs=False
):
    tensor, selector = load_selected_tensor(
        payload,
        layer_idx,
        module_name,
        selectors,
        outputs=outputs,
    )
    tensor = normalize_token_layout(tensor, seq_lens)
    if tensor.dim() >= 3 and tensor.shape[0] == sum(seq_lens) and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor.contiguous(), selector


def load_optional_token_tensor(
    payload, layer_idx, module_name, selectors, seq_lens, *, outputs=False
):
    tensor, selector = select_tensor(
        get_layer_entry(payload, layer_idx, module_name, outputs=outputs),
        selectors,
    )
    if tensor is None:
        return None, None
    tensor = normalize_token_layout(tensor, seq_lens)
    if tensor.dim() >= 3 and tensor.shape[0] == sum(seq_lens) and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor.contiguous(), selector


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    return value


class ForwardCapture:
    def __init__(self, modules):
        self.records = {
            name: {"inputs": [], "outputs": []}
            for name, module in modules.items()
            if module is not None
        }
        self.handles = []
        for name, module in modules.items():
            if module is None:
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name):
        def hook(module, args):  # noqa: ARG001
            self.records[name]["inputs"].append(detach_tree(args))

        return hook

    def _post_hook(self, name):
        def hook(module, args, output):  # noqa: ARG001
            self.records[name]["outputs"].append(detach_tree(output))

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def mlp_capture_modules(mlp):
    modules = {
        "router": mlp.router,
        "experts": mlp.experts,
        "shared_fc1": None,
        "shared_fc2": None,
        "expert_fc1": None,
        "expert_fc2": None,
    }
    if hasattr(mlp, "shared_experts"):
        modules["shared_fc1"] = getattr(mlp.shared_experts, "linear_fc1", None)
        modules["shared_fc2"] = getattr(mlp.shared_experts, "linear_fc2", None)
    modules["expert_fc1"] = getattr(mlp.experts, "linear_fc1", None)
    modules["expert_fc2"] = getattr(mlp.experts, "linear_fc2", None)
    return modules


def call_mlp(mlp, packed_input, seq_lens):
    padded_input = packed_to_padded(packed_input, seq_lens).cuda().contiguous()
    with torch.no_grad():
        output = mlp(padded_input)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return padded_to_packed(output, seq_lens).contiguous()


def call_mlp_with_capture(mlp, packed_input, seq_lens):
    capture = ForwardCapture(mlp_capture_modules(mlp))
    try:
        output = call_mlp(mlp, packed_input, seq_lens)
    finally:
        capture.close()
    return output, capture.records


def print_replay_stats(label, actual, expected):
    if actual.shape != expected.shape:
        print(f"{label:<42s} SHAPE {tuple(actual.shape)} vs {tuple(expected.shape)}")
        return
    stats = diff_stats(actual.cpu(), expected.cpu())
    print(
        f"{label:<42s} max={stats['max_abs_diff']:.6f} "
        f"mean={stats['mean_abs_diff']:.6e} cos={stats['cos_sim']:.9f} "
        f"shape={tuple(actual.shape)}"
    )


def bf16_add(left, right):
    return (left.cuda() + right.cuda()).to(torch.bfloat16).cpu()


def only_record(records, name, kind):
    values = records.get(name, {}).get(kind, [])
    if len(values) != 1:
        return None
    return values[0]


def record_output_tensor(records, name):
    return first_tensor(only_record(records, name, "outputs"))


def record_input_arg(records, name, index):
    args = only_record(records, name, "inputs")
    if not isinstance(args, tuple) or index >= len(args):
        return None
    return first_tensor(args[index])


def call_grouped_linear(linear, packed_input, tokens_per_expert):
    from megatron.core.typed_torch import apply_module

    with torch.no_grad():
        output = apply_module(linear)(
            packed_input.cuda().contiguous(),
            tokens_per_expert,
        )
    return first_tensor(output).detach().cpu()


def grouped_linear_weight(linear, num_experts, input_dim, output_dim):
    candidates = []
    expert_parameters = []
    for name, parameter in linear.named_parameters(recurse=True):
        candidates.append((name, tuple(parameter.shape)))
        if name.startswith("weight") and name[6:].isdigit():
            expert_idx = int(name[6:])
            if parameter.shape in ((output_dim, input_dim), (input_dim, output_dim)):
                expert_parameters.append((expert_idx, parameter))
        if parameter.dim() == 3 and parameter.shape[0] == num_experts:
            if parameter.shape[1:] in (
                (output_dim, input_dim),
                (input_dim, output_dim),
            ):
                return parameter, candidates
        if parameter.dim() == 2 and parameter.shape == (
            num_experts * output_dim,
            input_dim,
        ):
            return (
                parameter.view(
                    num_experts,
                    output_dim,
                    input_dim,
                ),
                candidates,
            )
        if parameter.dim() == 2 and parameter.shape == (
            num_experts * input_dim,
            output_dim,
        ):
            return (
                parameter.view(
                    num_experts,
                    input_dim,
                    output_dim,
                ),
                candidates,
            )
        if parameter.dim() == 2 and parameter.shape == (
            output_dim,
            num_experts * input_dim,
        ):
            return (
                parameter.view(
                    output_dim,
                    num_experts,
                    input_dim,
                ).permute(1, 0, 2),
                candidates,
            )
        if parameter.dim() == 2 and parameter.shape == (
            input_dim,
            num_experts * output_dim,
        ):
            return (
                parameter.view(
                    input_dim,
                    num_experts,
                    output_dim,
                ).permute(1, 0, 2),
                candidates,
            )
    expert_parameters = sorted(expert_parameters, key=lambda item: item[0])
    if len(expert_parameters) == num_experts and [
        idx for idx, _ in expert_parameters
    ] == list(range(num_experts)):
        return [parameter for _, parameter in expert_parameters], candidates
    return None, candidates


def grouped_linear_reference(
    linear,
    packed_input,
    tokens_per_expert,
    accumulate_dtype,
    output_dim,
):
    weight, candidates = grouped_linear_weight(
        linear,
        len(tokens_per_expert),
        packed_input.shape[-1],
        output_dim,
    )
    if weight is None:
        return None, candidates

    packed_input = packed_input.cuda().contiguous()
    if isinstance(weight, list):
        weights = [
            expert_weight.detach().cuda().contiguous() for expert_weight in weight
        ]
    else:
        weights = weight.detach().cuda().contiguous()
    outputs = []
    start = 0
    for expert_idx, count in enumerate(tokens_per_expert):
        end = start + int(count)
        if end == start:
            continue
        expert_input = packed_input[start:end].to(accumulate_dtype)
        expert_weight = weights[expert_idx].to(accumulate_dtype)
        if expert_weight.shape[1] == expert_input.shape[-1]:
            expert_output = torch.matmul(expert_input, expert_weight.T)
        elif expert_weight.shape[0] == expert_input.shape[-1]:
            expert_output = torch.matmul(expert_input, expert_weight)
        else:
            raise RuntimeError(
                "cannot align grouped-linear weight shape "
                f"{tuple(expert_weight.shape)} with input "
                f"{tuple(expert_input.shape)}"
            )
        outputs.append(expert_output.to(torch.bfloat16).cpu())
        start = end
    if start != packed_input.shape[0]:
        raise RuntimeError(
            f"tokens_per_expert sums to {start}, expected {packed_input.shape[0]}"
        )
    if not outputs:
        return None, candidates
    return torch.cat(outputs, dim=0).contiguous(), candidates


def print_internal_sensitivity(label, records_a, records_b):
    pairs = (
        (
            "router probs",
            record_output_tensor(records_a, "router"),
            record_output_tensor(records_b, "router"),
        ),
        (
            "expert packed input",
            record_input_arg(records_a, "experts", 0),
            record_input_arg(records_b, "experts", 0),
        ),
        (
            "expert fc1 output",
            record_output_tensor(records_a, "expert_fc1"),
            record_output_tensor(records_b, "expert_fc1"),
        ),
        (
            "expert fc2 input",
            record_input_arg(records_a, "expert_fc2", 0),
            record_input_arg(records_b, "expert_fc2", 0),
        ),
        (
            "expert fc2 output",
            record_output_tensor(records_a, "expert_fc2"),
            record_output_tensor(records_b, "expert_fc2"),
        ),
        (
            "experts output",
            record_output_tensor(records_a, "experts"),
            record_output_tensor(records_b, "experts"),
        ),
        (
            "shared fc1 output",
            record_output_tensor(records_a, "shared_fc1"),
            record_output_tensor(records_b, "shared_fc1"),
        ),
        (
            "shared fc2 output",
            record_output_tensor(records_a, "shared_fc2"),
            record_output_tensor(records_b, "shared_fc2"),
        ),
    )
    print(f"\n{label}")
    for name, left, right in pairs:
        if left is None or right is None or left.shape != right.shape:
            print(f"  {name:<28s} MISSING")
            continue
        print_replay_stats(f"  {name}", left, right)


def normalize_replay_router(tensor, seq_lens):
    tensor = normalize_router_layout(tensor, seq_lens)
    if tensor.dim() >= 3 and tensor.shape[0] == sum(seq_lens) and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor.contiguous()


def route_indices_from_map(router_map):
    num_tokens = router_map.shape[0]
    flat_indices = router_map.T.contiguous().reshape(-1).nonzero().flatten()
    token_indices = flat_indices % num_tokens
    expert_indices = flat_indices // num_tokens
    return token_indices.cpu(), expert_indices.cpu()


def tokens_per_expert_from_indices(expert_indices, num_experts):
    return (
        torch.bincount(
            expert_indices.to(torch.long),
            minlength=num_experts,
        )
        .cpu()
        .tolist()
    )


def combine_expert_major(packed_values, token_indices, num_tokens, dtype):
    combined = torch.zeros(num_tokens, packed_values.shape[-1], dtype=dtype)
    combined.index_add_(0, token_indices, packed_values.to(dtype).cpu())
    return combined


def combine_token_topk(packed_values, token_indices, expert_indices, topk_ids, dtype):
    num_tokens = topk_ids.shape[0]
    num_experts = int(expert_indices.max().item()) + 1
    row_lookup = torch.full((num_tokens, num_experts), -1, dtype=torch.long)
    row_lookup[token_indices, expert_indices] = torch.arange(token_indices.numel())
    combined = torch.zeros(num_tokens, packed_values.shape[-1], dtype=dtype)
    packed_values = packed_values.to(dtype).cpu()
    for token_idx in range(num_tokens):
        for expert_idx in topk_ids[token_idx].cpu().tolist():
            if expert_idx < 0 or expert_idx >= num_experts:
                continue
            row_idx = int(row_lookup[token_idx, expert_idx])
            if row_idx >= 0:
                combined[token_idx] = (combined[token_idx] + packed_values[row_idx]).to(
                    dtype
                )
    return combined


def packed_vllm_weighted_values(
    packed_output,
    packed_probs,
    token_indices,
    expert_indices,
    v_topk_ids,
    v_topk_weights,
):
    num_tokens = v_topk_ids.shape[0]
    topk = v_topk_ids.shape[1]
    values = torch.zeros_like(packed_probs.float())
    for row_idx, (token_idx, expert_idx) in enumerate(
        zip(token_indices, expert_indices)
    ):
        token_idx = int(token_idx)
        expert_idx = int(expert_idx)
        matches = (v_topk_ids[token_idx].cpu() == expert_idx).nonzero().flatten()
        if matches.numel() == 0:
            continue
        topk_idx = int(matches[0])
        if topk_idx < topk:
            values[row_idx] = v_topk_weights[token_idx, topk_idx].float()
    unweighted = packed_output.float() / packed_probs.float().clamp_min(
        1.0e-12
    ).unsqueeze(-1)
    return unweighted * values.unsqueeze(-1)


def load_vllm_moe_fc2_payload(path, router_map, packed_probs):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kernel_calls = payload.get("moe_kernel_calls", [])
    activation_calls = payload.get("moe_activation_calls", [])
    if len(kernel_calls) < 2 or not activation_calls:
        raise RuntimeError(
            "Need vLLM MoE internals with at least fc1/fc2 kernels and activation"
        )
    fc1_call = kernel_calls[0]
    fc2_call = kernel_calls[1]
    activation_call = activation_calls[0]
    num_flat_rows = activation_call["output"].shape[0]
    ordered_indices = vllm_flat_indices_in_megatron_order(
        fc1_call,
        router_map,
        num_flat_rows,
    )
    activation = activation_call["output"][ordered_indices].contiguous()
    weighted_fc2 = (
        fc2_call["C_after"].reshape(num_flat_rows, -1)[ordered_indices].contiguous()
    )
    weighted_activation_fp32 = (activation.float() * packed_probs[:, None].float()).to(
        torch.bfloat16
    )
    weighted_activation_bf16 = (
        activation.to(torch.bfloat16) * packed_probs[:, None].to(torch.bfloat16)
    ).to(torch.bfloat16)
    return {
        "activation": activation,
        "weighted_activation_fp32": weighted_activation_fp32,
        "weighted_activation_bf16": weighted_activation_bf16,
        "weighted_fc2": weighted_fc2,
        "fc1_call": fc1_call,
        "fc2_call": fc2_call,
        "num_flat_rows": num_flat_rows,
        "ordered_indices": ordered_indices,
        "fc1_mul_routed_weight": fc1_call["mul_routed_weight"],
        "fc2_mul_routed_weight": fc2_call["mul_routed_weight"],
        "top_k": fc1_call["top_k"],
    }


def as_vllm_fc2_weight(linear, num_experts, input_dim, output_dim):
    weight, candidates = grouped_linear_weight(
        linear,
        num_experts,
        input_dim,
        output_dim,
    )
    if weight is None:
        return None, candidates

    expert_weights = []
    if isinstance(weight, list):
        iterable = weight
    else:
        iterable = [weight[expert_idx] for expert_idx in range(weight.shape[0])]
    for expert_weight in iterable:
        expert_weight = expert_weight.detach()
        if expert_weight.shape == (output_dim, input_dim):
            oriented = expert_weight
        elif expert_weight.shape == (input_dim, output_dim):
            oriented = expert_weight.T
        else:
            raise RuntimeError(
                "cannot orient fc2 weight for vLLM kernel: "
                f"weight={tuple(expert_weight.shape)} "
                f"expected {(output_dim, input_dim)} or {(input_dim, output_dim)}"
            )
        expert_weights.append(oriented)
    return torch.stack(expert_weights, dim=0).to(
        torch.bfloat16
    ).contiguous(), candidates


def triton_compute_type(dtype):
    import triton.language as tl

    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"unsupported vLLM MoE replay dtype: {dtype}")


def cuda_value(value):
    if isinstance(value, torch.Tensor):
        return value.cuda().contiguous()
    return value


def bool_value(value):
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def replay_vllm_triton_fc2_kernel(fc2_call, fc2_weight):
    from vllm.model_executor.layers.fused_moe import fused_moe

    A = cuda_value(fc2_call["A"])
    C = torch.zeros_like(fc2_call["C_after"], device=A.device)
    B = fc2_weight.cuda().contiguous()
    fused_moe.invoke_fused_moe_triton_kernel(
        A,
        B,
        C,
        cuda_value(fc2_call["A_scale"]),
        cuda_value(fc2_call.get("B_scale")),
        cuda_value(fc2_call["topk_weights"]),
        cuda_value(fc2_call["sorted_token_ids"]),
        cuda_value(fc2_call["expert_ids"]),
        cuda_value(fc2_call["num_tokens_post_padded"]),
        bool_value(fc2_call["mul_routed_weight"]),
        int(fc2_call["top_k"]),
        dict(fc2_call["config"]),
        compute_type=triton_compute_type(A.dtype),
        use_fp8_w8a8=bool_value(fc2_call.get("use_fp8_w8a8", False)),
        use_int8_w8a8=bool_value(fc2_call.get("use_int8_w8a8", False)),
        use_int8_w8a16=bool_value(fc2_call.get("use_int8_w8a16", False)),
        use_int4_w4a16=bool_value(fc2_call.get("use_int4_w4a16", False)),
        per_channel_quant=bool_value(fc2_call.get("per_channel_quant", False)),
        block_shape=fc2_call.get("block_shape"),
        B_bias=None,
    )
    torch.cuda.synchronize()
    return C.detach().cpu()


def print_vllm_triton_fc2_replay(
    linear_fc2,
    vllm_fc2,
    m_fc2_output,
    router_map,
    save_fc2_weight_path=None,
):
    fc2_call = vllm_fc2["fc2_call"]
    fc2_weight, weight_candidates = as_vllm_fc2_weight(
        linear_fc2,
        router_map.shape[1],
        fc2_call["A"].shape[-1],
        m_fc2_output.shape[-1],
    )
    if fc2_weight is None:
        print(
            "[vllm-triton-fc2] grouped-linear weight unavailable; "
            f"parameter_candidates={weight_candidates}"
        )
        return
    if save_fc2_weight_path:
        os.makedirs(os.path.dirname(save_fc2_weight_path), exist_ok=True)
        torch.save(
            {
                "weight": fc2_weight.cpu(),
                "shape": tuple(fc2_weight.shape),
                "layout": "vllm_fc2_weight_E_out_in",
            },
            save_fc2_weight_path,
        )
        print(f"[vllm-triton-fc2] saved fc2 weight to {save_fc2_weight_path}")

    try:
        replay_c = replay_vllm_triton_fc2_kernel(fc2_call, fc2_weight)
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"[vllm-triton-fc2] replay unavailable: {error}")
        return

    flat_replay = replay_c.reshape(vllm_fc2["num_flat_rows"], -1)[
        vllm_fc2["ordered_indices"]
    ].contiguous()
    print(
        "\n[vllm-triton-fc2] "
        f"B={tuple(fc2_weight.shape)} A={tuple(fc2_call['A'].shape)} "
        f"C={tuple(fc2_call['C_after'].shape)}"
    )
    print_replay_stats(
        "vllm triton fc2 replay vs saved C",
        replay_c,
        fc2_call["C_after"],
    )
    print_replay_stats(
        "vllm triton fc2 replay vs v weighted fc2",
        flat_replay,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "vllm triton fc2 replay vs captured m fc2",
        flat_replay,
        m_fc2_output,
    )


def print_fc2_replay_variants(
    linear_fc2,
    records,
    seq_lens,
    vllm_moe_internals_path,
    save_fc2_weight_path=None,
):
    router_output = only_record(records, "router", "outputs")
    m_fc2_input = record_input_arg(records, "expert_fc2", 0)
    m_fc2_output = record_output_tensor(records, "expert_fc2")
    if (
        not isinstance(router_output, (list, tuple))
        or m_fc2_input is None
        or m_fc2_output is None
        or linear_fc2 is None
    ):
        print("\n[fc2-replay] missing router/fc2 capture")
        return

    router_probs = normalize_replay_router(first_tensor(router_output[0]), seq_lens)
    router_map = normalize_replay_router(
        first_tensor(router_output[1]), seq_lens
    ).bool()
    token_indices, expert_indices = route_indices_from_map(router_map)
    packed_probs = router_probs[token_indices, expert_indices].float().contiguous()
    tokens_per_expert = tokens_per_expert_from_indices(
        expert_indices,
        router_map.shape[1],
    )
    vllm_fc2 = load_vllm_moe_fc2_payload(
        vllm_moe_internals_path,
        router_map,
        packed_probs,
    )
    print(
        "\n[fc2-replay] "
        f"top_k={vllm_fc2['top_k']} "
        f"fc1_mul={vllm_fc2['fc1_mul_routed_weight']} "
        f"fc2_mul={vllm_fc2['fc2_mul_routed_weight']} "
        f"routes={int(token_indices.numel())}"
    )
    print_replay_stats(
        "v activation*prob fp32 vs m fc2 input",
        vllm_fc2["weighted_activation_fp32"],
        m_fc2_input,
    )
    print_replay_stats(
        "v activation*prob bf16 vs m fc2 input",
        vllm_fc2["weighted_activation_bf16"],
        m_fc2_input,
    )
    print_vllm_triton_fc2_replay(
        linear_fc2,
        vllm_fc2,
        m_fc2_output,
        router_map,
        save_fc2_weight_path=save_fc2_weight_path,
    )

    replay_m_input = call_grouped_linear(linear_fc2, m_fc2_input, tokens_per_expert)
    replay_v_weighted_fp32 = call_grouped_linear(
        linear_fc2,
        vllm_fc2["weighted_activation_fp32"],
        tokens_per_expert,
    )
    replay_v_weighted_bf16 = call_grouped_linear(
        linear_fc2,
        vllm_fc2["weighted_activation_bf16"],
        tokens_per_expert,
    )
    replay_v_raw = call_grouped_linear(
        linear_fc2,
        vllm_fc2["activation"].to(torch.bfloat16),
        tokens_per_expert,
    )
    replay_v_post_prob_fp32 = (replay_v_raw.float() * packed_probs[:, None].float()).to(
        torch.bfloat16
    )
    replay_v_post_prob_bf16 = (
        replay_v_raw.to(torch.bfloat16) * packed_probs[:, None].to(torch.bfloat16)
    ).to(torch.bfloat16)
    torch_ref_m_bf16, weight_candidates = grouped_linear_reference(
        linear_fc2,
        m_fc2_input,
        tokens_per_expert,
        torch.bfloat16,
        m_fc2_output.shape[-1],
    )
    torch_ref_m_fp32, _ = grouped_linear_reference(
        linear_fc2,
        m_fc2_input,
        tokens_per_expert,
        torch.float32,
        m_fc2_output.shape[-1],
    )
    torch_ref_v_weighted_bf16, _ = grouped_linear_reference(
        linear_fc2,
        vllm_fc2["weighted_activation_fp32"],
        tokens_per_expert,
        torch.bfloat16,
        m_fc2_output.shape[-1],
    )
    torch_ref_v_weighted_fp32, _ = grouped_linear_reference(
        linear_fc2,
        vllm_fc2["weighted_activation_fp32"],
        tokens_per_expert,
        torch.float32,
        m_fc2_output.shape[-1],
    )
    torch_ref_v_raw_fp32, _ = grouped_linear_reference(
        linear_fc2,
        vllm_fc2["activation"].to(torch.bfloat16),
        tokens_per_expert,
        torch.float32,
        m_fc2_output.shape[-1],
    )

    print_replay_stats(
        "direct fc2(m input) vs captured m fc2",
        replay_m_input,
        m_fc2_output,
    )
    print_replay_stats(
        "direct fc2(v act*prob fp32) vs captured m fc2",
        replay_v_weighted_fp32,
        m_fc2_output,
    )
    print_replay_stats(
        "direct fc2(v act*prob bf16) vs captured m fc2",
        replay_v_weighted_bf16,
        m_fc2_output,
    )
    print_replay_stats(
        "direct fc2(v act*prob fp32) vs v weighted fc2",
        replay_v_weighted_fp32,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "direct fc2(v act*prob bf16) vs v weighted fc2",
        replay_v_weighted_bf16,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "direct fc2(v raw)*prob fp32 vs v weighted fc2",
        replay_v_post_prob_fp32,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "direct fc2(v raw)*prob bf16 vs v weighted fc2",
        replay_v_post_prob_bf16,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "direct fc2(v raw)*prob fp32 vs captured m fc2",
        replay_v_post_prob_fp32,
        m_fc2_output,
    )
    if (
        torch_ref_m_bf16 is None
        or torch_ref_m_fp32 is None
        or torch_ref_v_weighted_bf16 is None
        or torch_ref_v_weighted_fp32 is None
        or torch_ref_v_raw_fp32 is None
    ):
        print(
            "[fc2-replay] grouped-linear torch reference unavailable; "
            f"parameter_candidates={weight_candidates}"
        )
        return

    torch_ref_v_post_prob_fp32 = (
        torch_ref_v_raw_fp32.float() * packed_probs[:, None].float()
    ).to(torch.bfloat16)
    print_replay_stats(
        "torch bf16 fc2(m input) vs captured m fc2",
        torch_ref_m_bf16,
        m_fc2_output,
    )
    print_replay_stats(
        "torch fp32 fc2(m input) vs captured m fc2",
        torch_ref_m_fp32,
        m_fc2_output,
    )
    print_replay_stats(
        "torch bf16 fc2(v act*prob) vs v weighted fc2",
        torch_ref_v_weighted_bf16,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "torch fp32 fc2(v act*prob) vs v weighted fc2",
        torch_ref_v_weighted_fp32,
        vllm_fc2["weighted_fc2"],
    )
    print_replay_stats(
        "torch fp32 fc2(v act*prob) vs captured m fc2",
        torch_ref_v_weighted_fp32,
        m_fc2_output,
    )
    print_replay_stats(
        "torch fp32 fc2(v raw)*prob vs v weighted fc2",
        torch_ref_v_post_prob_fp32,
        vllm_fc2["weighted_fc2"],
    )


def print_routed_combine_variants(
    v_routed_target,
    replay_routed,
    records,
    seq_lens,
    v_topk_ids,
    v_topk_weights,
):
    router_output = only_record(records, "router", "outputs")
    packed_output = record_output_tensor(records, "experts")
    packed_probs = record_input_arg(records, "experts", 2)
    if (
        not isinstance(router_output, (list, tuple))
        or packed_output is None
        or packed_probs is None
    ):
        print("\n[routed-combine] missing router/expert capture")
        return
    router_probs = normalize_replay_router(first_tensor(router_output[0]), seq_lens)
    router_map = normalize_replay_router(
        first_tensor(router_output[1]), seq_lens
    ).bool()
    token_indices, expert_indices = route_indices_from_map(router_map)
    packed_prob_ref = router_probs[token_indices, expert_indices].float()
    print_replay_stats(
        "packed probs vs router probs", packed_probs.float(), packed_prob_ref
    )

    if packed_output.shape[0] != token_indices.numel():
        print(
            "[routed-combine] shape mismatch "
            f"packed={tuple(packed_output.shape)} routes={int(token_indices.numel())}"
        )
        return

    variants = [
        (
            "expert-major fp32",
            combine_expert_major(
                packed_output, token_indices, router_map.shape[0], torch.float32
            ),
        ),
        (
            "expert-major bf16",
            combine_expert_major(
                packed_output, token_indices, router_map.shape[0], torch.bfloat16
            ),
        ),
        (
            "token-topk bf16",
            combine_token_topk(
                packed_output, token_indices, expert_indices, v_topk_ids, torch.bfloat16
            ),
        ),
        (
            "token-topk fp32",
            combine_token_topk(
                packed_output, token_indices, expert_indices, v_topk_ids, torch.float32
            ),
        ),
    ]
    vprob_values = packed_vllm_weighted_values(
        packed_output,
        packed_probs,
        token_indices,
        expert_indices,
        v_topk_ids,
        v_topk_weights,
    )
    variants.extend(
        [
            (
                "vprob expert-major fp32",
                combine_expert_major(
                    vprob_values, token_indices, router_map.shape[0], torch.float32
                ),
            ),
            (
                "vprob token-topk bf16",
                combine_token_topk(
                    vprob_values,
                    token_indices,
                    expert_indices,
                    v_topk_ids,
                    torch.bfloat16,
                ),
            ),
        ]
    )
    print("\n[routed-combine] variants vs saved vLLM routed-only target")
    for label, combined in variants:
        print_replay_stats(f"  {label}", combined, v_routed_target)
    print("\n[routed-combine] variants vs replay routed output")
    for label, combined in variants:
        print_replay_stats(f"  {label}", combined, replay_routed)


def main():
    args = parse_args()
    model_ref = MODEL_ALIASES.get(args.model, args.model)
    vllm = torch.load(args.vllm_capture, map_location="cpu", weights_only=False)
    megatron = torch.load(args.megatron_capture, map_location="cpu", weights_only=False)
    seq_lens = vllm.get("seq_lens") or megatron.get("seq_lens")
    if not seq_lens:
        raise KeyError("captures do not contain seq_lens")

    v_mlp_input, v_mlp_input_selector = load_token_tensor(
        vllm, args.layer, "mixer", ("arg0", "first"), seq_lens
    )
    m_mlp_input, m_mlp_input_selector = load_token_tensor(
        megatron, args.layer, "mlp", ("arg0", "first"), seq_lens
    )
    v_mlp_output, v_mlp_output_selector = load_token_tensor(
        vllm, args.layer, "mixer", ("output", "item0", "first"), seq_lens, outputs=True
    )
    v_shared_output, v_shared_output_selector = load_optional_token_tensor(
        vllm,
        args.layer,
        "mixer.shared_experts.down_proj",
        ("output", "item0", "first"),
        seq_lens,
        outputs=True,
    )
    v_topk_weights, _ = load_optional_token_tensor(
        vllm,
        args.layer,
        "mixer.experts.router.select_experts",
        ("item0", "output", "first"),
        seq_lens,
        outputs=True,
    )
    v_topk_ids, _ = load_optional_token_tensor(
        vllm,
        args.layer,
        "mixer.experts.router.select_experts",
        ("item1",),
        seq_lens,
        outputs=True,
    )
    m_mlp_output, m_mlp_output_selector = load_token_tensor(
        megatron,
        args.layer,
        "mlp",
        ("item0", "output", "first"),
        seq_lens,
        outputs=True,
    )
    v_layer_input, _ = load_token_tensor(
        vllm,
        args.layer,
        "<layer>",
        ("kw.hidden_states+kw.residual", "kw.hidden_states", "first"),
        seq_lens,
    )
    m_layer_input, _ = load_token_tensor(
        megatron, args.layer, "<layer>", ("kw.hidden_states", "first"), seq_lens
    )
    v_layer_output, _ = load_token_tensor(
        vllm,
        args.layer,
        "<layer>",
        ("item0+item1", "output", "first"),
        seq_lens,
        outputs=True,
    )
    m_layer_output, _ = load_token_tensor(
        megatron,
        args.layer,
        "<layer>",
        ("item0+item1", "output", "first"),
        seq_lens,
        outputs=True,
    )

    print(
        "[replay-moe] "
        f"seq_lens={seq_lens} layer={args.layer} "
        f"v_mlp_input={v_mlp_input_selector} m_mlp_input={m_mlp_input_selector} "
        f"v_mlp_output={v_mlp_output_selector} m_mlp_output={m_mlp_output_selector}"
    )
    print(f"[replay-moe] v_shared_output={v_shared_output_selector}")
    print_stats("saved v input vs saved m input", v_mlp_input, m_mlp_input)
    print_stats("saved v output vs saved m output", v_mlp_output, m_mlp_output)

    model = load_megatron_model(model_ref)
    layers = find_decoder_layers(model)
    mlp = layers[args.layer].mlp

    replay_from_m_input, m_replay_records = call_mlp_with_capture(
        mlp, m_mlp_input, seq_lens
    )
    replay_from_v_input, v_replay_records = call_mlp_with_capture(
        mlp, v_mlp_input, seq_lens
    )

    print_replay_stats(
        "Megatron MLP(m input) vs saved m output", replay_from_m_input, m_mlp_output
    )
    print_replay_stats(
        "Megatron MLP(v input) vs saved v output", replay_from_v_input, v_mlp_output
    )
    print_replay_stats(
        "Megatron MLP(v input) vs saved m output", replay_from_v_input, m_mlp_output
    )
    print_replay_stats(
        "Megatron MLP(v input) vs MLP(m input)",
        replay_from_v_input,
        replay_from_m_input,
    )

    replay_v_layer_output = bf16_add(v_layer_input, replay_from_v_input)
    replay_m_layer_output = bf16_add(m_layer_input, replay_from_v_input)
    print_replay_stats(
        "bf16(v layer input + replay) vs v layer out",
        replay_v_layer_output,
        v_layer_output,
    )
    print_replay_stats(
        "bf16(m layer input + replay) vs m layer out",
        replay_m_layer_output,
        m_layer_output,
    )

    print_internal_sensitivity(
        "[replay-moe] Megatron internal sensitivity: v input vs m input",
        v_replay_records,
        m_replay_records,
    )

    replay_shared_output = record_output_tensor(v_replay_records, "shared_fc2")
    if (
        replay_shared_output is not None
        and v_shared_output is not None
        and v_topk_ids is not None
        and v_topk_weights is not None
    ):
        replay_shared_output = normalize_token_layout(replay_shared_output, seq_lens)
        if (
            replay_shared_output.dim() >= 3
            and replay_shared_output.shape[0] == sum(seq_lens)
            and replay_shared_output.shape[1] == 1
        ):
            replay_shared_output = replay_shared_output.squeeze(1)
        replay_routed = replay_from_v_input.float().cpu() - replay_shared_output.float()
        v_routed_target = v_mlp_output.float() - v_shared_output.float()
        print_replay_stats(
            "Megatron shared(v input) vs saved v shared",
            replay_shared_output,
            v_shared_output,
        )
        print_replay_stats(
            "Megatron routed(v input) vs saved v routed", replay_routed, v_routed_target
        )
        print_routed_combine_variants(
            v_routed_target,
            replay_routed,
            v_replay_records,
            seq_lens,
            v_topk_ids.to(torch.long),
            v_topk_weights.float(),
        )
        print_fc2_replay_variants(
            getattr(mlp.experts, "linear_fc2", None),
            v_replay_records,
            seq_lens,
            args.vllm_moe_internals,
            save_fc2_weight_path=args.save_vllm_fc2_weight,
        )
    else:
        print(
            "[replay-moe] missing shared/topk capture; skipping routed combine variants"
        )


if __name__ == "__main__":
    main()
