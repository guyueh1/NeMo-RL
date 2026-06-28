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

"""Replay captured vLLM NemotronH Mamba SSD-scan inputs with Megatron's op."""

import argparse
import math
import os

import torch
from megatron.core.ssm.ops.ssd_combined import mamba_chunk_scan_combined_varlen

DEFAULT_CAPTURE = os.path.join(
    "session",
    "20260625_121743",
    "nemotron_bf16_no_bi_triton_single_prompt",
    "vllm_capture_nemotron3_nano_mamba_internals.pt",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-capture", default=DEFAULT_CAPTURE)
    parser.add_argument("--scan-call-index", type=int, default=0)
    return parser.parse_args()


def cuda_tensor(value, dtype=None):
    if value is None:
        return None
    value = value.cuda()
    if dtype is not None:
        value = value.to(dtype)
    return value.contiguous()


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


def dtype_candidates(captured_state_dtype):
    candidates = [("none", None), ("bf16", torch.bfloat16), ("fp32", torch.float32)]
    if captured_state_dtype == "torch.float32":
        candidates.insert(0, ("captured-fp32", torch.float32))
    elif captured_state_dtype == "torch.bfloat16":
        candidates.insert(0, ("captured-bf16", torch.bfloat16))
    return candidates


def main():
    args = parse_args()
    payload = torch.load(args.vllm_capture, map_location="cpu", weights_only=False)
    scan_calls = payload.get("mamba_scan_calls", [])
    conv_calls = payload.get("mamba_conv1d_calls", [])
    if args.scan_call_index >= len(scan_calls):
        raise IndexError(
            f"scan call {args.scan_call_index} out of range for {len(scan_calls)} calls"
        )
    scan = scan_calls[args.scan_call_index]
    conv = (
        conv_calls[args.scan_call_index]
        if args.scan_call_index < len(conv_calls)
        else {}
    )
    metadata = conv.get("metadata") or {}

    chunk_size = metadata.get("chunk_size")
    if chunk_size is None:
        chunk_size = 128
        print("[scan-replay] chunk_size missing from capture; using 128")

    x = cuda_tensor(scan["x"])
    dt = cuda_tensor(scan["dt"])
    A = cuda_tensor(scan["A"])
    B = cuda_tensor(scan["B"])
    C = cuda_tensor(scan["C"])
    D = cuda_tensor(scan.get("D"))
    dt_bias = cuda_tensor(scan.get("dt_bias"))
    seq_idx = cuda_tensor(scan.get("seq_idx"))
    cu_seqlens = cuda_tensor(scan.get("cu_seqlens"))
    cu_chunk_seqlens = cuda_tensor(scan.get("cu_chunk_seqlens"))
    last_chunk_indices = cuda_tensor(scan.get("last_chunk_indices"))
    expected = cuda_tensor(scan["out_after"])
    captured_state_dtype = scan.get("state_dtype")

    print(
        f"[scan-replay] capture={args.vllm_capture} call={args.scan_call_index} "
        f"chunk_size={chunk_size} captured_state_dtype={captured_state_dtype}"
    )
    print(
        f"[scan-replay] x={tuple(x.shape)} dt={tuple(dt.shape)} "
        f"B={tuple(B.shape)} C={tuple(C.shape)} expected={tuple(expected.shape)}"
    )
    if cu_seqlens is not None:
        print(f"[scan-replay] cu_seqlens={cu_seqlens.cpu().tolist()}")
    if cu_chunk_seqlens is not None:
        print(f"[scan-replay] cu_chunk_seqlens={cu_chunk_seqlens.cpu().tolist()}")
    if last_chunk_indices is not None:
        print(f"[scan-replay] last_chunk_indices={last_chunk_indices.cpu().tolist()}")

    for label, state_dtype in dtype_candidates(captured_state_dtype):
        initial_states = scan.get("initial_states")
        initial_states = cuda_tensor(initial_states, dtype=state_dtype)
        out = torch.zeros_like(x)
        result = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
            D=D,
            z=None,
            dt_bias=dt_bias,
            initial_states=initial_states,
            return_intermediate_states=isinstance(scan.get("result"), tuple),
            dt_softplus=True,
            dt_limit=(0.0, math.inf),
            state_dtype=state_dtype,
        )
        print_stats(f"megatron scan {label}", out, expected)
        if isinstance(result, torch.Tensor):
            print(
                f"[scan-replay] result {label} shape={tuple(result.shape)} dtype={result.dtype}"
            )
        else:
            print(f"[scan-replay] result {label} type={type(result).__name__}")


if __name__ == "__main__":
    main()
