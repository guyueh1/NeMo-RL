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

"""Compare random Q/K/V attention paths with the GSM8K batch-32 shapes.

This is a focused diagnostic for the vLLM-vs-Megatron attention mismatch.  It
generates identical random Q/K/V tensors, then compares:

1. Direct packed FA2 vs the Megatron layout path that packs from ``(B,S,H,D)``.
2. Direct packed FA2 run as one batch vs split into vLLM-like request chunks.
3. vLLM paged-KV FA2 vs the Megatron direct packed path.
4. vLLM paged-KV FA2 split into vLLM-like request chunks vs the Megatron path.

Run inside the Lyris container with the vLLM worker environment, for example:

    /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
        my_script/compare_random_sdpa_qkv.py --iters 10
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

GSM8K32_SEQ_LENS = [
    44,
    63,
    44,
    47,
    45,
    55,
    45,
    53,
    31,
    41,
    40,
    65,
    56,
    49,
    44,
    55,
    46,
    69,
    51,
    52,
    42,
    79,
    63,
    42,
    45,
    58,
    55,
    85,
    44,
    47,
    74,
    41,
]


@dataclass
class Metrics:
    max_abs: float
    mean_abs: float
    rms_abs: float
    mean_rel: float
    norm_mean_abs: float
    norm_rms_abs: float
    equal: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seq-lens", default=None)
    parser.add_argument("--seq-lens-from", type=Path, default=None)
    parser.add_argument("--split-request-counts", default="1,31")
    parser.add_argument("--skip-paged", action="store_true")
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def load_seq_lens(args: argparse.Namespace) -> list[int]:
    if args.seq_lens:
        return [int(item) for item in args.seq_lens.split(",") if item]
    if args.seq_lens_from is not None:
        payload = torch.load(args.seq_lens_from, map_location="cpu", weights_only=False)
        seq_lens = payload.get("seq_lens")
        if not seq_lens:
            raise ValueError(f"{args.seq_lens_from} does not contain seq_lens")
        return [int(item) for item in seq_lens]
    return list(GSM8K32_SEQ_LENS)


def cu_seqlens(seq_lens: list[int], device: torch.device) -> torch.Tensor:
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(lens, 0, dtype=torch.int32),
        ]
    )


def pack_bshd(tensor: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
    return torch.cat(
        [tensor[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(seq_lens)],
        dim=0,
    ).contiguous()


def scatter_bshd(
    packed: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    seq_lens: list[int],
) -> torch.Tensor:
    output = packed.new_zeros((batch_size, seq_len, packed.size(1), packed.size(2)))
    offset = 0
    for batch_idx, cur_seq_len in enumerate(seq_lens):
        next_offset = offset + cur_seq_len
        output[batch_idx, :cur_seq_len] = packed[offset:next_offset]
        offset = next_offset
    return output


def flash_attn_varlen_func():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func as func

    return func


def direct_packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_lens: list[int],
    causal: bool,
    fa_version: int,
    num_splits: int,
) -> torch.Tensor:
    func = flash_attn_varlen_func()
    max_seq_len = max(seq_lens)
    cu = cu_seqlens(seq_lens, q.device)
    out = func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_seq_len,
        max_seqlen_k=max_seq_len,
        softmax_scale=1.0 / math.sqrt(q.size(-1)),
        causal=causal,
        fa_version=fa_version,
        num_splits=num_splits,
        deterministic=False,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def megatron_layout_attention(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *,
    seq_lens: list[int],
    causal: bool,
    fa_version: int,
    num_splits: int,
) -> torch.Tensor:
    q = pack_bshd(q_bshd, seq_lens)
    k = pack_bshd(k_bshd, seq_lens)
    v = pack_bshd(v_bshd, seq_lens)
    out = direct_packed_attention(
        q,
        k,
        v,
        seq_lens=seq_lens,
        causal=causal,
        fa_version=fa_version,
        num_splits=num_splits,
    )
    scattered = scatter_bshd(
        out,
        batch_size=q_bshd.size(0),
        seq_len=q_bshd.size(1),
        seq_lens=seq_lens,
    )
    return pack_bshd(scattered, seq_lens)


def build_vllm_block_metadata(
    seq_lens: list[int],
    *,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    max_blocks_per_request = math.ceil(max(seq_lens) / block_size)
    block_table = torch.zeros(
        (len(seq_lens), max_blocks_per_request), dtype=torch.int32, device=device
    )
    slot_mapping: list[int] = []
    next_block = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        num_blocks = math.ceil(seq_len / block_size)
        blocks = torch.arange(
            next_block,
            next_block + num_blocks,
            dtype=torch.int32,
            device=device,
        )
        block_table[batch_idx, :num_blocks] = blocks
        for pos in range(seq_len):
            slot_mapping.append(
                (next_block + pos // block_size) * block_size + pos % block_size
            )
        next_block += num_blocks
    return (
        block_table,
        torch.tensor(slot_mapping, dtype=torch.int64, device=device),
        next_block,
    )


def gather_cache_by_slot_mapping(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    flat_cache = cache.reshape(-1, cache.size(-2), cache.size(-1))
    return flat_cache.index_select(0, slot_mapping).contiguous()


def build_vllm_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_lens: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

    block_table, slot_mapping, num_blocks = build_vllm_block_metadata(
        seq_lens,
        block_size=block_size,
        device=k.device,
    )
    key_cache = torch.empty(
        (num_blocks, block_size, k.size(1), k.size(2)), dtype=k.dtype, device=k.device
    )
    value_cache = torch.empty_like(key_cache)
    scale = torch.ones((), dtype=torch.float32, device=k.device)
    reshape_and_cache_flash(
        k,
        v,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        scale,
        scale,
    )
    return key_cache, value_cache, block_table, slot_mapping


def vllm_paged_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_lens: list[int],
    block_size: int,
    causal: bool,
    fa_version: int,
    num_splits: int,
) -> torch.Tensor:
    func = flash_attn_varlen_func()
    key_cache, value_cache, block_table, _ = build_vllm_cache(
        k,
        v,
        seq_lens=seq_lens,
        block_size=block_size,
    )

    out = torch.empty_like(q)
    result = func(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens(seq_lens, q.device),
        max_seqlen_q=max(seq_lens),
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=q.device),
        max_seqlen_k=max(seq_lens),
        softmax_scale=1.0 / math.sqrt(q.size(-1)),
        causal=causal,
        block_table=block_table,
        fa_version=fa_version,
        num_splits=num_splits,
    )
    if isinstance(result, tuple):
        return result[0]
    return out


def split_attention(
    attention_fn,
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *,
    seq_lens: list[int],
    split_request_counts: list[int],
    **kwargs,
) -> torch.Tensor:
    outputs = []
    start = 0
    for count in split_request_counts:
        end = start + count
        if end > len(seq_lens):
            raise ValueError("split_request_counts exceeds batch size")
        cur_seq_lens = seq_lens[start:end]
        q = pack_bshd(q_bshd[start:end], cur_seq_lens)
        k = pack_bshd(k_bshd[start:end], cur_seq_lens)
        v = pack_bshd(v_bshd[start:end], cur_seq_lens)
        outputs.append(attention_fn(q, k, v, seq_lens=cur_seq_lens, **kwargs))
        start = end
    if start != len(seq_lens):
        raise ValueError("split_request_counts must sum to batch size")
    return torch.cat(outputs, dim=0)


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> Metrics:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    ref_abs = expected_f.abs()
    eps = 1e-7
    return Metrics(
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        rms_abs=float(torch.sqrt(torch.mean(diff * diff)).item()),
        mean_rel=float((diff / torch.clamp(ref_abs, min=eps)).mean().item()),
        norm_mean_abs=float(
            diff.mean().div(torch.clamp(ref_abs.mean(), min=eps)).item()
        ),
        norm_rms_abs=float(
            torch.sqrt(torch.mean(diff * diff))
            .div(torch.clamp(torch.sqrt(torch.mean(expected_f * expected_f)), min=eps))
            .item()
        ),
        equal=bool(torch.equal(actual, expected)),
    )


def generate_qkv(
    *,
    seed: int,
    seq_lens: list[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    q = torch.randn(
        (batch_size, max_seq_len, num_q_heads, head_dim),
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    k = torch.randn(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    v = torch.randn(
        (batch_size, max_seq_len, num_kv_heads, head_dim),
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    return q, k, v


def summarize(
    all_metrics: dict[str, list[Metrics]],
) -> dict[str, dict[str, float | bool]]:
    summary: dict[str, dict[str, float | bool]] = {}
    metric_names = (
        "max_abs",
        "mean_abs",
        "rms_abs",
        "mean_rel",
        "norm_mean_abs",
        "norm_rms_abs",
    )
    for name, values in all_metrics.items():
        summary[name] = {
            f"avg_{metric_name}": sum(getattr(item, metric_name) for item in values)
            / len(values)
            for metric_name in metric_names
        }
        summary[name].update(
            {
                f"max_{metric_name}": max(getattr(item, metric_name) for item in values)
                for metric_name in metric_names
            }
        )
        summary[name]["all_equal"] = all(item.equal for item in values)
    return summary


def print_metrics(name: str, metrics: Metrics) -> None:
    print(
        f"{name}: equal={metrics.equal} "
        f"max_abs={metrics.max_abs:.9g} mean_abs={metrics.mean_abs:.9g} "
        f"rms_abs={metrics.rms_abs:.9g} mean_rel={metrics.mean_rel:.9g} "
        f"norm_mean_abs={metrics.norm_mean_abs:.9g} "
        f"norm_rms_abs={metrics.norm_rms_abs:.9g}"
    )


def main() -> None:
    args = parse_args()
    seq_lens = load_seq_lens(args)
    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)
    split_request_counts = [
        int(item) for item in args.split_request_counts.split(",") if item
    ]
    if sum(split_request_counts) != len(seq_lens):
        raise ValueError(
            f"split_request_counts={split_request_counts} must sum to "
            f"batch size {len(seq_lens)}"
        )

    print(
        "shape: "
        f"batch={len(seq_lens)} total_tokens={sum(seq_lens)} "
        f"min_seq={min(seq_lens)} max_seq={max(seq_lens)} "
        f"q=({sum(seq_lens)}, {args.num_q_heads}, {args.head_dim}) "
        f"kv=({sum(seq_lens)}, {args.num_kv_heads}, {args.head_dim}) "
        f"dtype={args.dtype} causal={args.causal} "
        f"fa_version={args.fa_version} num_splits={args.num_splits}"
    )

    all_metrics: dict[str, list[Metrics]] = {
        "direct_packed_vs_megatron_layout": [],
        "direct_split_vs_direct_packed": [],
    }
    if not args.skip_paged:
        all_metrics["paged_cache_k_vs_original"] = []
        all_metrics["paged_cache_v_vs_original"] = []
        all_metrics["vllm_paged_vs_megatron_layout"] = []
        all_metrics["vllm_paged_split_vs_megatron_layout"] = []

    with torch.no_grad():
        for iteration in range(args.iters):
            seed = args.seed + iteration
            q_bshd, k_bshd, v_bshd = generate_qkv(
                seed=seed,
                seq_lens=seq_lens,
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
            )
            q = pack_bshd(q_bshd, seq_lens)
            k = pack_bshd(k_bshd, seq_lens)
            v = pack_bshd(v_bshd, seq_lens)
            direct = direct_packed_attention(
                q,
                k,
                v,
                seq_lens=seq_lens,
                causal=args.causal,
                fa_version=args.fa_version,
                num_splits=args.num_splits,
            )
            megatron = megatron_layout_attention(
                q_bshd,
                k_bshd,
                v_bshd,
                seq_lens=seq_lens,
                causal=args.causal,
                fa_version=args.fa_version,
                num_splits=args.num_splits,
            )
            direct_split = split_attention(
                direct_packed_attention,
                q_bshd,
                k_bshd,
                v_bshd,
                seq_lens=seq_lens,
                split_request_counts=split_request_counts,
                causal=args.causal,
                fa_version=args.fa_version,
                num_splits=args.num_splits,
            )

            torch.cuda.synchronize()
            all_metrics["direct_packed_vs_megatron_layout"].append(
                compare_tensors(direct, megatron)
            )
            all_metrics["direct_split_vs_direct_packed"].append(
                compare_tensors(direct_split, direct)
            )

            if not args.skip_paged:
                key_cache, value_cache, _, slot_mapping = build_vllm_cache(
                    k,
                    v,
                    seq_lens=seq_lens,
                    block_size=args.block_size,
                )
                gathered_k = gather_cache_by_slot_mapping(key_cache, slot_mapping)
                gathered_v = gather_cache_by_slot_mapping(value_cache, slot_mapping)
                paged = vllm_paged_attention(
                    q,
                    k,
                    v,
                    seq_lens=seq_lens,
                    block_size=args.block_size,
                    causal=args.causal,
                    fa_version=args.fa_version,
                    num_splits=args.num_splits,
                )
                paged_split = split_attention(
                    vllm_paged_attention,
                    q_bshd,
                    k_bshd,
                    v_bshd,
                    seq_lens=seq_lens,
                    split_request_counts=split_request_counts,
                    block_size=args.block_size,
                    causal=args.causal,
                    fa_version=args.fa_version,
                    num_splits=args.num_splits,
                )
                torch.cuda.synchronize()
                all_metrics["paged_cache_k_vs_original"].append(
                    compare_tensors(gathered_k, k)
                )
                all_metrics["paged_cache_v_vs_original"].append(
                    compare_tensors(gathered_v, v)
                )
                all_metrics["vllm_paged_vs_megatron_layout"].append(
                    compare_tensors(paged, megatron)
                )
                all_metrics["vllm_paged_split_vs_megatron_layout"].append(
                    compare_tensors(paged_split, megatron)
                )

            print(f"iteration={iteration} seed={seed}")
            for name, values in all_metrics.items():
                if len(values) == iteration + 1:
                    print_metrics(name, values[-1])

    summary = summarize(all_metrics)
    print("summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
