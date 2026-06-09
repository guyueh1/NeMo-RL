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

"""Compare saved NeMo-RL vLLM/Megatron tensor dump artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _cap(path: Path) -> dict[str, Any]:
    return _load(path)["capture"]


def _to_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0, :]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor after squeeze, got {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


def _get_by_path(value: Any, path: str) -> torch.Tensor:
    cur = value
    if path == "":
        if not isinstance(cur, torch.Tensor):
            raise TypeError(f"empty path did not resolve to a tensor: {type(cur)}")
        return cur
    for part in path.split("."):
        if part.startswith("[") and part.endswith("]"):
            cur = cur[int(part[1:-1])]
        else:
            cur = cur[part]
    if not isinstance(cur, torch.Tensor):
        raise TypeError(f"{path} did not resolve to a tensor: {type(cur)}")
    return cur


def _module_tensor(
    cap: dict[str, Any],
    *,
    kind: str,
    layer_idx: int,
    module_name: str,
    tensor_path: str,
) -> torch.Tensor:
    key = f"module_{kind}s_by_layer"
    return _get_by_path(cap[key][layer_idx][module_name], tensor_path)


def _metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs = _to_2d(lhs)
    rhs = _to_2d(rhs)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    diff = lhs - rhs
    abs_diff = diff.abs()
    denom = torch.maximum(
        torch.maximum(lhs.abs(), rhs.abs()),
        torch.full_like(abs_diff, 1e-12),
    )
    rel = abs_diff / denom
    equal = (
        lhs.to(torch.bfloat16)
        .view(torch.int16)
        .eq(rhs.to(torch.bfloat16).view(torch.int16))
    )
    return {
        "shape": list(lhs.shape),
        "allclose_atol0_rtol0": bool(torch.equal(lhs, rhs)),
        "bf16_bitwise_equal_fraction": float(equal.float().mean().item()),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
        "numel": int(lhs.numel()),
        "num_nonzero": int((abs_diff != 0).sum().item()),
    }


def _slice_policy(tensor: torch.Tensor, offset: int, length: int) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        return tensor[offset : offset + length]
    return tensor[offset : offset + length]


def _compare_item(
    *,
    name: str,
    gen_cap: dict[str, Any],
    policy_cap: dict[str, Any],
    layer_idx: int,
    gen_kind: str,
    gen_module: str,
    gen_path: str,
    policy_kind: str,
    policy_module: str,
    policy_path: str,
    offset: int,
    length: int,
) -> dict[str, Any]:
    gen_tensor = _module_tensor(
        gen_cap,
        kind=gen_kind,
        layer_idx=layer_idx,
        module_name=gen_module,
        tensor_path=gen_path,
    )
    policy_tensor = _module_tensor(
        policy_cap,
        kind=policy_kind,
        layer_idx=layer_idx,
        module_name=policy_module,
        tensor_path=policy_path,
    )
    gen_tensor = gen_tensor[:length]
    policy_tensor = _slice_policy(policy_tensor, offset, length)
    item = _metrics(gen_tensor, policy_tensor)
    item.update(
        {
            "name": name,
            "layer_idx": layer_idx,
            "offset": offset,
            "generation": {
                "kind": gen_kind,
                "module": gen_module,
                "path": gen_path,
            },
            "policy": {
                "kind": policy_kind,
                "module": policy_module,
                "path": policy_path,
            },
        }
    )
    return item


def _layer_entry_metric(
    *,
    gen_cap: dict[str, Any],
    policy_cap: dict[str, Any],
    layer_idx: int,
    offset: int,
    length: int,
) -> dict[str, Any]:
    return _compare_item(
        name=f"layer{layer_idx}.entry_hidden",
        gen_cap=gen_cap,
        policy_cap=policy_cap,
        layer_idx=layer_idx,
        gen_kind="input",
        gen_module="<layer>",
        gen_path="args.[1]",
        policy_kind="input",
        policy_module="<layer>",
        policy_path="kwargs.hidden_states",
        offset=offset,
        length=length,
    )


def _layer0_items() -> list[dict[str, str]]:
    return [
        {
            "name": "layer0.entry_hidden",
            "gen_kind": "input",
            "gen_module": "<layer>",
            "gen_path": "args.[1]",
            "policy_kind": "input",
            "policy_module": "<layer>",
            "policy_path": "kwargs.hidden_states",
        },
        {
            "name": "layer0.input_layernorm.input",
            "gen_kind": "input",
            "gen_module": "input_layernorm",
            "gen_path": "args.[0]",
            "policy_kind": "input",
            "policy_module": "input_layernorm",
            "policy_path": "args.[0]",
        },
        {
            "name": "layer0.input_layernorm.output",
            "gen_kind": "output",
            "gen_module": "input_layernorm",
            "gen_path": "",
            "policy_kind": "output",
            "policy_module": "input_layernorm",
            "policy_path": "",
        },
        {
            "name": "layer0.qkv.input",
            "gen_kind": "input",
            "gen_module": "self_attn.qkv_proj",
            "gen_path": "args.[0]",
            "policy_kind": "input",
            "policy_module": "self_attention.linear_qkv",
            "policy_path": "args.[0]",
        },
        {
            "name": "layer0.qkv.output",
            "gen_kind": "output",
            "gen_module": "self_attn.qkv_proj",
            "gen_path": "[0]",
            "policy_kind": "output",
            "policy_module": "self_attention.linear_qkv",
            "policy_path": "[0]",
        },
        {
            "name": "layer0.attn.q_input",
            "gen_kind": "input",
            "gen_module": "self_attn.attn",
            "gen_path": "args.[0]",
            "policy_kind": "input",
            "policy_module": "self_attention.core_attention",
            "policy_path": "args.[0]",
        },
        {
            "name": "layer0.attn.k_input",
            "gen_kind": "input",
            "gen_module": "self_attn.attn",
            "gen_path": "args.[1]",
            "policy_kind": "input",
            "policy_module": "self_attention.core_attention",
            "policy_path": "args.[1]",
        },
        {
            "name": "layer0.attn.v_input",
            "gen_kind": "input",
            "gen_module": "self_attn.attn",
            "gen_path": "args.[2]",
            "policy_kind": "input",
            "policy_module": "self_attention.core_attention",
            "policy_path": "args.[2]",
        },
        {
            "name": "layer0.attn.output",
            "gen_kind": "output",
            "gen_module": "self_attn.attn",
            "gen_path": "",
            "policy_kind": "output",
            "policy_module": "self_attention.core_attention",
            "policy_path": "",
        },
        {
            "name": "layer0.o_proj.output",
            "gen_kind": "output",
            "gen_module": "self_attn.o_proj",
            "gen_path": "[0]",
            "policy_kind": "output",
            "policy_module": "self_attention.linear_proj",
            "policy_path": "[0]",
        },
        {
            "name": "layer0.pre_mlp_norm.output",
            "gen_kind": "output",
            "gen_module": "post_attention_layernorm",
            "gen_path": "[0]",
            "policy_kind": "output",
            "policy_module": "pre_mlp_layernorm",
            "policy_path": "",
        },
        {
            "name": "layer0.mlp_fc1.output",
            "gen_kind": "output",
            "gen_module": "mlp.gate_up_proj",
            "gen_path": "[0]",
            "policy_kind": "output",
            "policy_module": "mlp.linear_fc1",
            "policy_path": "[0]",
        },
        {
            "name": "layer0.silu_mul.output",
            "gen_kind": "output",
            "gen_module": "mlp.act_fn",
            "gen_path": "",
            "policy_kind": "input",
            "policy_module": "mlp.linear_fc2",
            "policy_path": "args.[0]",
        },
        {
            "name": "layer0.mlp_fc2.output",
            "gen_kind": "output",
            "gen_module": "mlp.down_proj",
            "gen_path": "[0]",
            "policy_kind": "output",
            "policy_module": "mlp.linear_fc2",
            "policy_path": "[0]",
        },
        {
            "name": "layer0.mlp.output",
            "gen_kind": "output",
            "gen_module": "mlp",
            "gen_path": "",
            "policy_kind": "output",
            "policy_module": "mlp",
            "policy_path": "[0]",
        },
        {
            "name": "layer0.output_hidden",
            "gen_kind": "output",
            "gen_module": "<layer>",
            "gen_path": "[0]",
            "policy_kind": "output",
            "policy_module": "<layer>",
            "policy_path": "[0]",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base = args.dump_dir
    comparison = _load(next(base.glob("*comparison*.pt")))
    prompt_lens = [
        int(row.nonzero(as_tuple=False).squeeze(-1)[0].item())
        for row in comparison["token_mask"]
    ]
    input_lengths = [int(v) for v in comparison["logprob_input_lengths"].tolist()]
    gen_paths = sorted(base.glob("*generation*.pt"))
    policy_paths = sorted(base.glob("*policy*.pt"))
    policy_by_rank = {_load(path)["metadata"]["rank"]: path for path in policy_paths}

    row_by_prompt_len = {prompt_len: row for row, prompt_len in enumerate(prompt_lens)}
    pairs = []
    for gen_path in gen_paths:
        gen_cap = _cap(gen_path)
        length = int(
            gen_cap["module_inputs_by_layer"][0]["input_layernorm"]["args"][0].shape[0]
        )
        row = row_by_prompt_len[length]
        rank = row
        pairs.append(
            {
                "row": row,
                "rank": rank,
                "length": length,
                "input_length": input_lengths[row],
                "right_offset": 0,
                "left_offset": 168 - input_lengths[row],
                "generation_path": gen_path,
                "policy_path": policy_by_rank[rank],
            }
        )

    pair_results = []
    for pair in pairs:
        gen_cap = _cap(pair["generation_path"])
        policy_cap = _cap(pair["policy_path"])
        offsets = {
            "right_padded_offset0": pair["right_offset"],
            "left_padded_offset": pair["left_offset"],
        }
        offset_results = {}
        for offset_name, offset in offsets.items():
            layer_entries = [
                _layer_entry_metric(
                    gen_cap=gen_cap,
                    policy_cap=policy_cap,
                    layer_idx=layer_idx,
                    offset=offset,
                    length=pair["length"],
                )
                for layer_idx in range(32)
            ]
            layer0 = [
                _compare_item(
                    name=item["name"],
                    gen_cap=gen_cap,
                    policy_cap=policy_cap,
                    layer_idx=0,
                    offset=offset,
                    length=pair["length"],
                    gen_kind=item["gen_kind"],
                    gen_module=item["gen_module"],
                    gen_path=item["gen_path"],
                    policy_kind=item["policy_kind"],
                    policy_module=item["policy_module"],
                    policy_path=item["policy_path"],
                )
                for item in _layer0_items()
            ]
            offset_results[offset_name] = {
                "offset": offset,
                "layer_entries": layer_entries,
                "layer0_modules": layer0,
            }
        pair_result = {
            **{k: v for k, v in pair.items() if not k.endswith("_path")},
            "generation_file": pair["generation_path"].name,
            "policy_file": pair["policy_path"].name,
            "offset_results": offset_results,
        }
        pair_results.append(pair_result)

    summary = {"pairs": pair_results}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))

    for pair in pair_results:
        print(
            f"PAIR row={pair['row']} rank={pair['rank']} len={pair['length']} "
            f"gen={pair['generation_file']} policy={pair['policy_file']}"
        )
        for offset_name, offset_data in pair["offset_results"].items():
            first_layer_entry = next(
                (
                    item
                    for item in offset_data["layer_entries"]
                    if item["max_abs"] != 0.0
                ),
                None,
            )
            first_layer0_module = next(
                (
                    item
                    for item in offset_data["layer0_modules"]
                    if item["max_abs"] != 0.0
                ),
                None,
            )
            best_entry0 = offset_data["layer_entries"][0]
            print(
                f"  {offset_name} offset={offset_data['offset']} "
                f"entry0 max_abs={best_entry0['max_abs']:.6e} "
                f"mean_abs={best_entry0['mean_abs']:.6e} "
                f"bf16_equal={best_entry0['bf16_bitwise_equal_fraction']:.6f}"
            )
            if first_layer_entry:
                print(
                    "    first differing layer entry: "
                    f"{first_layer_entry['name']} max_abs={first_layer_entry['max_abs']:.6e} "
                    f"mean_abs={first_layer_entry['mean_abs']:.6e} "
                    f"bf16_equal={first_layer_entry['bf16_bitwise_equal_fraction']:.6f}"
                )
            if first_layer0_module:
                print(
                    "    first differing layer0 module: "
                    f"{first_layer0_module['name']} max_abs={first_layer0_module['max_abs']:.6e} "
                    f"mean_abs={first_layer0_module['mean_abs']:.6e} "
                    f"bf16_equal={first_layer0_module['bf16_bitwise_equal_fraction']:.6f}"
                )


if __name__ == "__main__":
    main()
