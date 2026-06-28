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

"""Replay captured vLLM NemotronH Mamba causal-conv inputs with reference math."""

import argparse
import os

import torch
import torch.nn.functional as F

DEFAULT_CAPTURE = os.path.join(
    "session",
    "20260625_121743",
    "nemotron_bf16_no_bi_triton_single_prompt",
    "vllm_capture_nemotron3_nano_mamba_internals.pt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-capture", default=DEFAULT_CAPTURE)
    parser.add_argument("--conv-call-index", type=int, default=0)
    return parser.parse_args()


def diff_stats(actual, expected):
    actual_f = actual.detach().float().reshape(-1)
    expected_f = expected.detach().float().reshape(-1)
    diff = (actual_f - expected_f).abs()
    denom = actual_f.norm() * expected_f.norm()
    cos = (
        torch.dot(actual_f, expected_f) / denom
        if denom != 0
        else torch.tensor(float("nan"))
    )
    return {
        "max": float(diff.max().item()) if diff.numel() else 0.0,
        "mean": float(diff.mean().item()) if diff.numel() else 0.0,
        "cos": float(cos.item()),
    }


def print_stats(label, actual, expected):
    stats = diff_stats(actual.cpu(), expected.cpu())
    print(
        f"{label:<34s} max={stats['max']:.6f} "
        f"mean={stats['mean']:.6e} cos={stats['cos']:.9f} "
        f"shape={tuple(actual.shape)}"
    )


def reference_conv(
    x, weight, bias, cu_seqlens, *, flip_weight=False, out_dtype=torch.bfloat16
):
    total_tokens, conv_dim = x.shape
    width = weight.shape[1]
    out = torch.empty((total_tokens, conv_dim), dtype=out_dtype, device=x.device)
    weight = weight.flip(1) if flip_weight else weight
    x_f = x.float()
    weight_f = weight.float()
    bias_f = bias.float()

    for seq_idx in range(cu_seqlens.numel() - 1):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        for pos in range(end - start):
            acc = bias_f.clone()
            for tap_idx in range(width):
                src_pos = pos - (width - 1) + tap_idx
                if src_pos >= 0:
                    acc = acc + x_f[start + src_pos] * weight_f[:, tap_idx]
            out[start + pos] = F.silu(acc).to(out_dtype)
    return out


def main():
    args = parse_args()
    payload = torch.load(args.vllm_capture, map_location="cpu", weights_only=False)
    calls = payload.get("mamba_conv1d_calls", [])
    if args.conv_call_index >= len(calls):
        raise IndexError(
            f"conv call {args.conv_call_index} out of range for {len(calls)} calls"
        )
    call = calls[args.conv_call_index]
    metadata = call.get("metadata") or {}

    x = call["input_token_major"].cuda()
    output = call["output_token_major"].cuda()
    weight = call["weight"].cuda()
    bias = call["bias"].cuda()
    cu_seqlens = call["query_start_loc"].cuda()
    has_initial_state = call.get("has_initial_state")
    if isinstance(has_initial_state, torch.Tensor):
        has_initial_state_value = has_initial_state.cpu().tolist()
    else:
        has_initial_state_value = has_initial_state

    print(
        f"[conv-replay] capture={args.vllm_capture} call={args.conv_call_index} "
        f"x={tuple(x.shape)} weight={tuple(weight.shape)} output={tuple(output.shape)}"
    )
    print(f"[conv-replay] query_start_loc={cu_seqlens.cpu().tolist()}")
    print(f"[conv-replay] has_initial_state={has_initial_state_value}")
    for field in (
        "num_prefill_tokens",
        "chunk_size",
        "prep_initial_states",
        "block_idx_first_scheduled_token_p",
        "block_idx_last_scheduled_token",
        "block_idx_last_computed_token",
        "num_computed_tokens_p",
    ):
        value = metadata.get(field)
        if isinstance(value, torch.Tensor):
            value = value.cpu().tolist()
        print(f"[conv-replay] metadata {field}={value}")

    ref = reference_conv(x, weight, bias, cu_seqlens)
    print_stats("reference conv vs vLLM", ref, output)
    ref_flipped = reference_conv(x, weight, bias, cu_seqlens, flip_weight=True)
    print_stats("reference flipped vs vLLM", ref_flipped, output)


if __name__ == "__main__":
    main()
