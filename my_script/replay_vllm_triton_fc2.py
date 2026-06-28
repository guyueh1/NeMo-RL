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

"""Replay vLLM's routed-MoE Triton fc2 kernel from captured call metadata."""

import argparse
import os

import torch
from compare import get_layer_entry, normalize_router_layout, select_tensor
from compare_vllm_moe_internals import (
    load_module_tensor,
    print_stats,
    vllm_flat_indices_in_megatron_order,
)

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
        "--megatron-capture",
        default=os.path.join(
            DEFAULT_SESSION_DIR,
            "megatron_capture_nemotron3_nano_mambaprefill_router_layers0_5.pt",
        ),
    )
    parser.add_argument(
        "--fc2-weight",
        default=os.path.join(DEFAULT_SESSION_DIR, "layer1_fc2_weight_vllm_layout.pt"),
    )
    parser.add_argument("--layer", type=int, default=1)
    return parser.parse_args()


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
    fused_moe.invoke_fused_moe_triton_kernel(
        A,
        fc2_weight.cuda().contiguous(),
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


def load_router_map(megatron, layer):
    router_map, selector = select_tensor(
        get_layer_entry(megatron, layer, "mlp.router", outputs=True),
        ("item1",),
    )
    if router_map is None:
        raise KeyError("missing Megatron mlp.router routing map output")
    router_map = normalize_router_layout(router_map, megatron["seq_lens"]).bool()
    return router_map, selector


def load_fc2_weight(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "weight" in payload:
        weight = payload["weight"]
    else:
        weight = payload
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"fc2 weight payload is not a tensor: {type(weight)!r}")
    if weight.dim() != 3:
        raise ValueError(f"expected fc2 weight rank 3, got {tuple(weight.shape)}")
    return weight.contiguous()


def main():
    args = parse_args()
    vllm = torch.load(args.vllm_moe_internals, map_location="cpu", weights_only=False)
    megatron = torch.load(args.megatron_capture, map_location="cpu", weights_only=False)
    fc2_weight = load_fc2_weight(args.fc2_weight)

    kernel_calls = vllm.get("moe_kernel_calls", [])
    activation_calls = vllm.get("moe_activation_calls", [])
    if len(kernel_calls) < 2 or not activation_calls:
        raise RuntimeError("Need at least fc1/fc2 kernel calls and activation capture")
    fc1_call = kernel_calls[0]
    fc2_call = kernel_calls[1]
    num_flat_rows = activation_calls[0]["output"].shape[0]
    router_map, router_selector = load_router_map(megatron, args.layer)
    ordered_indices = vllm_flat_indices_in_megatron_order(
        fc1_call,
        router_map,
        num_flat_rows,
    )
    m_fc2_output, m_fc2_selector = load_module_tensor(
        megatron,
        args.layer,
        "mlp.experts.linear_fc2",
        ("output", "item0", "first"),
        outputs=True,
    )

    print(
        "[vllm-triton-fc2] "
        f"layer={args.layer} router={router_selector} m_fc2={m_fc2_selector} "
        f"A={tuple(fc2_call['A'].shape)} B={tuple(fc2_weight.shape)} "
        f"C={tuple(fc2_call['C_after'].shape)} top_k={fc2_call['top_k']} "
        f"mul={fc2_call['mul_routed_weight']}"
    )

    replay_c = replay_vllm_triton_fc2_kernel(fc2_call, fc2_weight)
    saved_c = fc2_call["C_after"]
    replay_ordered = replay_c.reshape(num_flat_rows, -1)[ordered_indices].contiguous()
    saved_ordered = saved_c.reshape(num_flat_rows, -1)[ordered_indices].contiguous()
    print_stats("vllm triton replay full C vs saved C", replay_c, saved_c)
    print_stats(
        "vllm triton replay ordered vs saved v fc2",
        replay_ordered,
        saved_ordered,
    )
    print_stats(
        "vllm triton replay ordered vs captured m fc2",
        replay_ordered,
        m_fc2_output,
    )
    print_stats("saved v fc2 ordered vs captured m fc2", saved_ordered, m_fc2_output)


if __name__ == "__main__":
    main()
