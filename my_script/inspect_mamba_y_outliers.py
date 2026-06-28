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

"""Inspect NemotronH layer-0 Mamba scan/gated-RMSNorm outliers."""

from __future__ import annotations

import argparse

import torch
from compare import diff_stats, get_layer_entry, normalize_token_layout, select_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", required=True)
    parser.add_argument("--megatron", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def tensor_from(payload, layer_idx, name, selectors, *, outputs=False):
    entry = get_layer_entry(payload, layer_idx, name, outputs=outputs)
    tensor, selector = select_tensor(entry, selectors)
    if tensor is None:
        raise KeyError(f"missing tensor {name} selectors={selectors} outputs={outputs}")
    seq_lens = payload.get("seq_lens") or []
    return normalize_token_layout(tensor, seq_lens).float(), selector


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
        f"shape={tuple(a.shape)}"
    )


def mamba_in_proj_chunks(total_dim):
    # Nemotron 3 Nano: z, x are d_inner=4096, B/C are n_groups*d_state=8*128,
    # and dt is nheads=64.
    d_inner = 4096
    bc_dim = 1024
    dt_dim = total_dim - 2 * d_inner - 2 * bc_dim
    if dt_dim <= 0:
        raise ValueError(f"unexpected Mamba in_proj dim: {total_dim}")
    bounds = []
    start = 0
    for name, size in (
        ("z", d_inner),
        ("x_pre_conv", d_inner),
        ("B", bc_dim),
        ("C", bc_dim),
        ("dt", dt_dim),
    ):
        end = start + size
        bounds.append((name, start, end))
        start = end
    if start != total_dim:
        raise ValueError(f"chunk sizes sum to {start}, expected {total_dim}")
    return bounds


def main():
    args = parse_args()
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)
    m = torch.load(args.megatron, map_location="cpu", weights_only=False)
    seq_lens = v.get("seq_lens") or m.get("seq_lens") or []

    layer_idx = args.layer
    v_norm_out, v_norm_sel = tensor_from(
        v, layer_idx, "norm", ("output", "item0", "first"), outputs=True
    )
    m_norm_out, m_norm_sel = tensor_from(
        m,
        layer_idx,
        "mixer.in_proj.norm",
        ("output", "item0", "first"),
        outputs=True,
    )
    v_proj, v_proj_sel = tensor_from(
        v, layer_idx, "mixer.in_proj", ("output", "item0", "first"), outputs=True
    )
    m_proj, m_proj_sel = tensor_from(
        m,
        layer_idx,
        "mixer.in_proj.linear",
        ("output", "item0", "first"),
        outputs=True,
    )
    v_y, v_y_sel = tensor_from(v, layer_idx, "mixer.norm", ("arg0", "first"))
    m_y, m_y_sel = tensor_from(m, layer_idx, "mixer.norm", ("arg0", "first"))
    v_z, v_z_sel = tensor_from(v, layer_idx, "mixer.norm", ("arg1",))
    m_z, m_z_sel = tensor_from(m, layer_idx, "mixer.norm", ("arg1",))
    v_norm2, v_norm2_sel = tensor_from(
        v, layer_idx, "mixer.norm", ("output", "item0", "first"), outputs=True
    )
    m_norm2, m_norm2_sel = tensor_from(
        m, layer_idx, "mixer.norm", ("output", "item0", "first"), outputs=True
    )

    print(f"layer={layer_idx} seq_lens={seq_lens}")
    print(
        "selectors: "
        f"norm=({v_norm_sel},{m_norm_sel}) "
        f"proj=({v_proj_sel},{m_proj_sel}) "
        f"y=({v_y_sel},{m_y_sel}) z=({v_z_sel},{m_z_sel}) "
        f"gated_norm=({v_norm2_sel},{m_norm2_sel})"
    )
    print_stats("pre-mixer norm output", v_norm_out, m_norm_out)
    print_stats("in_proj output", v_proj, m_proj)
    for name, start, end in mamba_in_proj_chunks(v_proj.shape[-1]):
        print_stats(f"in_proj chunk {name}", v_proj[:, start:end], m_proj[:, start:end])
    print_stats("gated norm y input", v_y, m_y)
    print_stats("gated norm z input", v_z, m_z)
    print_stats("gated norm output", v_norm2, m_norm2)

    diff = (v_y - m_y).abs()
    flat = diff.reshape(-1)
    top_values, top_indices = torch.topk(flat, k=min(args.topk, flat.numel()))
    width = v_y.shape[-1]
    proj_diff = (v_proj - m_proj).abs()

    print("\nTop gated-norm y-input diffs")
    print(
        "rank diff row prompt pos col head dim v_y m_y row_y_max row_proj_max z_row_max"
    )
    for rank, (value, flat_idx) in enumerate(zip(top_values, top_indices), start=1):
        row = int(flat_idx.item() // width)
        col = int(flat_idx.item() % width)
        prompt_idx, pos = row_to_prompt_pos(row, seq_lens)
        row_y_max = float(diff[row].max())
        row_proj_max = float(proj_diff[row].max())
        z_row_max = float((v_z[row] - m_z[row]).abs().max())
        print(
            f"{rank:>4d} {float(value):>11.6f} {row:>4d} "
            f"{prompt_idx:>6d} {pos:>4d} {col:>5d} "
            f"{col // 64:>4d} {col % 64:>3d} "
            f"{float(v_y[row, col]):>11.6f} {float(m_y[row, col]):>11.6f} "
            f"{row_y_max:>11.6f} {row_proj_max:>12.6f} {z_row_max:>10.6f}"
        )


if __name__ == "__main__":
    main()
