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

"""Inspect residual-stream outliers at a captured decoder layer boundary."""

from __future__ import annotations

import argparse

import torch
from compare import diff_stats, get_layer_entry, normalize_token_layout, select_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", required=True)
    parser.add_argument("--megatron", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--v-block-output", default="mixer.out_proj")
    parser.add_argument("--m-block-output", default="mixer.out_proj")
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def tensor_from(payload, layer_idx, name, selectors, *, outputs=False):
    entry = get_layer_entry(payload, layer_idx, name, outputs=outputs)
    tensor, selector = select_tensor(entry, selectors)
    if tensor is None:
        raise KeyError(f"missing tensor {name} selectors={selectors} outputs={outputs}")
    seq_lens = payload.get("seq_lens") or []
    tensor = normalize_token_layout(tensor, seq_lens)
    if len(seq_lens) == 1 and tensor.dim() >= 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor, selector


def row_to_prompt_pos(row_idx, seq_lens):
    offset = 0
    for prompt_idx, seq_len in enumerate(seq_lens):
        if row_idx < offset + seq_len:
            return prompt_idx, row_idx - offset
        offset += seq_len
    return -1, -1


def print_stats(label, a, b):
    stats = diff_stats(a, b)
    print(
        f"{label:<34s} max={stats['max_abs_diff']:.6f} "
        f"mean={stats['mean_abs_diff']:.6e} cos={stats['cos_sim']:.6f} "
        f"shape={tuple(a.shape)} dtype=({a.dtype},{b.dtype})"
    )


def bf16_add(a, b):
    return (a.to(torch.bfloat16) + b.to(torch.bfloat16)).to(torch.bfloat16)


def main():
    args = parse_args()
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)
    m = torch.load(args.megatron, map_location="cpu", weights_only=False)
    seq_lens = v.get("seq_lens") or m.get("seq_lens") or []

    v_input, v_input_sel = tensor_from(
        v,
        args.layer,
        "<layer>",
        ("kw.hidden_states+kw.residual", "arg1+arg2", "kw.hidden_states", "arg1"),
    )
    m_input, m_input_sel = tensor_from(
        m, args.layer, "<layer>", ("kw.hidden_states", "arg0")
    )
    v_block, v_block_sel = tensor_from(
        v,
        args.layer,
        args.v_block_output,
        ("output", "item0", "first"),
        outputs=True,
    )
    m_block, m_block_sel = tensor_from(
        m,
        args.layer,
        args.m_block_output,
        ("output", "item0", "first"),
        outputs=True,
    )
    v_output, v_output_sel = tensor_from(
        v,
        args.layer,
        "<layer>",
        ("item0+item1", "output", "first"),
        outputs=True,
    )
    m_output, m_output_sel = tensor_from(
        m,
        args.layer,
        "<layer>",
        ("item0+item1", "output", "first"),
        outputs=True,
    )

    v_readd = bf16_add(v_input, v_block)
    m_readd = bf16_add(m_input, m_block)

    print(f"layer={args.layer} seq_lens={seq_lens}")
    print(
        "selectors: "
        f"input=({v_input_sel},{m_input_sel}) "
        f"block=({v_block_sel},{m_block_sel}) "
        f"output=({v_output_sel},{m_output_sel})"
    )
    print_stats("layer input stream", v_input, m_input)
    print_stats("block output", v_block, m_block)
    print_stats("layer output stream", v_output, m_output)
    print_stats("v output vs bf16(input+block)", v_output, v_readd)
    print_stats("m output vs bf16(input+block)", m_output, m_readd)
    print_stats("bf16(input+block)", v_readd, m_readd)

    diff = (v_output.float() - m_output.float()).abs()
    flat = diff.reshape(-1)
    top_values, top_indices = torch.topk(flat, k=min(args.topk, flat.numel()))
    width = v_output.shape[-1]
    input_diff = (v_input.float() - m_input.float()).abs()
    block_diff = (v_block.float() - m_block.float()).abs()

    print("\nTop layer-output residual-stream diffs")
    print(
        "rank diff row prompt pos col v_out m_out "
        "v_in m_in in_diff v_block m_block block_diff "
        "v_readd m_readd readd_diff"
    )
    for rank, (value, flat_idx) in enumerate(zip(top_values, top_indices), start=1):
        row = int(flat_idx.item() // width)
        col = int(flat_idx.item() % width)
        prompt_idx, pos = row_to_prompt_pos(row, seq_lens)
        print(
            f"{rank:>4d} {float(value):>11.6f} {row:>4d} "
            f"{prompt_idx:>6d} {pos:>4d} {col:>5d} "
            f"{float(v_output[row, col]):>11.6f} "
            f"{float(m_output[row, col]):>11.6f} "
            f"{float(v_input[row, col]):>11.6f} "
            f"{float(m_input[row, col]):>11.6f} "
            f"{float(input_diff[row, col]):>9.6f} "
            f"{float(v_block[row, col]):>11.6f} "
            f"{float(m_block[row, col]):>11.6f} "
            f"{float(block_diff[row, col]):>10.6f} "
            f"{float(v_readd[row, col]):>11.6f} "
            f"{float(m_readd[row, col]):>11.6f} "
            f"{float((v_readd[row, col].float() - m_readd[row, col].float()).abs()):>10.6f}"
        )


if __name__ == "__main__":
    main()
