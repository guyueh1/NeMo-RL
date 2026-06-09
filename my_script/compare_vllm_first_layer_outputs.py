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

"""Compare first-layer module output tensors between two vLLM captures."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import torch
from compare_vllm_first_layer_inputs import (
    TensorDiff,
    best_prompt_idx,
    choose_worker_prompt_map,
    compare_tensor,
    extract_capture,
    iter_tensors,
    layer0_input_calls,
    load_pt,
    print_diff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standalone",
        required=True,
        help="Standalone vllm_forward.py .pt payload with embedded debug capture.",
    )
    parser.add_argument(
        "--nemo-glob",
        required=True,
        help="Glob for NeMo generation debug tensor capture .pt files.",
    )
    parser.add_argument(
        "--token-dump",
        required=True,
        help="First-batch token dump containing sample_prompt_token_ids_list.",
    )
    parser.add_argument(
        "--prompt-key",
        default="sample_prompt_token_ids_list",
        help="Prompt token ids key in --token-dump.",
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Optional layer-0 module name to compare, such as post_attention_layernorm.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Report the first module with max_abs above this threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of worst tensor diffs to print.",
    )
    return parser.parse_args()


def layer0_output_calls(capture: dict[str, Any]) -> dict[str, list[Any]]:
    calls_by_layer = capture.get("module_output_calls_by_layer", {})
    return calls_by_layer.get(0) or calls_by_layer.get("0") or {}


def output_tensors(output: Any) -> dict[str, torch.Tensor]:
    return {path: tensor for path, tensor in iter_tensors(output, "output")}


def cat_standalone_output_tensors(
    calls: list[Any], *, chunk_count: int, num_layers: int
) -> dict[str, torch.Tensor]:
    if len(calls) == chunk_count:
        selected_calls = calls
    elif num_layers > 0 and len(calls) == chunk_count * num_layers:
        selected_calls = [
            calls[chunk_idx * num_layers] for chunk_idx in range(chunk_count)
        ]
    else:
        selected_calls = calls[:chunk_count]

    tensors_by_path: dict[str, list[torch.Tensor]] = {}
    for output in selected_calls:
        for path, tensor in output_tensors(output).items():
            tensors_by_path.setdefault(path, []).append(tensor)

    concatenated: dict[str, torch.Tensor] = {}
    for path, tensors in tensors_by_path.items():
        if len(tensors) == 1:
            concatenated[path] = tensors[0]
            continue
        if all(
            tensor.ndim > 0 and tensor.shape[1:] == tensors[0].shape[1:]
            for tensor in tensors
        ):
            concatenated[path] = torch.cat(tensors, dim=0)
    return concatenated


def prompt_lengths_from_token_dump(
    token_dump: dict[str, Any], prompt_key: str
) -> list[int]:
    prompt_token_ids = token_dump[prompt_key]
    return [len(row) for row in prompt_token_ids]


def choose_worker_prompt_map_from_outputs(
    nemo_files: list[str],
    nemo_captures: dict[str, dict[str, Any]],
    standalone_tensors_by_module: dict[str, dict[str, torch.Tensor]],
    prompt_lengths: list[int],
) -> dict[str, int | None]:
    mapping = choose_worker_prompt_map(
        nemo_files, nemo_captures, standalone_tensors_by_module, prompt_lengths
    )
    if all(prompt_idx is not None for prompt_idx in mapping.values()):
        return mapping

    standalone_tensors = standalone_tensors_by_module.get("<layer>", {})
    standalone_hidden = standalone_tensors.get("output")
    if standalone_hidden is None:
        standalone_hidden = standalone_tensors.get("output.0")
    if standalone_hidden is None:
        return mapping

    output_mapping: dict[str, int | None] = {}
    for nemo_file in nemo_files:
        nemo_calls = layer0_output_calls(nemo_captures[nemo_file])
        nemo_outputs = nemo_calls.get("<layer>", [])
        if not nemo_outputs:
            output_mapping[nemo_file] = mapping[nemo_file]
            continue
        nemo_tensors = output_tensors(nemo_outputs[0])
        nemo_hidden = nemo_tensors.get("output")
        if nemo_hidden is None:
            nemo_hidden = nemo_tensors.get("output.0")
        if nemo_hidden is None:
            output_mapping[nemo_file] = mapping[nemo_file]
            continue
        prompt_idx, _ = best_prompt_idx(nemo_hidden, standalone_hidden, prompt_lengths)
        output_mapping[nemo_file] = prompt_idx
    return output_mapping


def main() -> None:
    args = parse_args()
    token_dump = load_pt(args.token_dump)
    prompt_lengths = prompt_lengths_from_token_dump(token_dump, args.prompt_key)

    standalone_payload = load_pt(args.standalone)
    standalone_capture = extract_capture(standalone_payload)
    standalone_input_calls = layer0_input_calls(standalone_capture)
    standalone_output_calls = layer0_output_calls(standalone_capture)
    standalone_layer_calls = standalone_input_calls.get("<layer>", [])
    chunk_count = len(standalone_layer_calls)
    num_layers = int(standalone_capture.get("num_layers", 0))
    standalone_tensors_by_module = {
        module: cat_standalone_output_tensors(
            calls, chunk_count=chunk_count, num_layers=num_layers
        )
        for module, calls in standalone_output_calls.items()
    }

    nemo_files = sorted(glob.glob(args.nemo_glob))
    if not nemo_files:
        raise FileNotFoundError(args.nemo_glob)
    nemo_captures = {path: extract_capture(load_pt(path)) for path in nemo_files}
    worker_prompt_map = choose_worker_prompt_map_from_outputs(
        nemo_files, nemo_captures, standalone_tensors_by_module, prompt_lengths
    )

    module_order = list(standalone_output_calls)
    if args.module is not None:
        module_order = [module for module in module_order if module == args.module]
        if not module_order:
            raise KeyError(
                f"Standalone capture does not include module {args.module!r}"
            )

    print(f"standalone={args.standalone}")
    print(f"nemo_files={len(nemo_files)}")
    print(f"prompt_lengths={prompt_lengths} total={sum(prompt_lengths)}")
    print(f"standalone_prefill_chunks={chunk_count} num_layers={num_layers}")
    print("worker_prompt_map:")
    for nemo_file in nemo_files:
        print(f"  {Path(nemo_file).name}: prompt_idx={worker_prompt_map[nemo_file]}")

    all_diffs: list[TensorDiff] = []
    first_divergent: TensorDiff | None = None
    for module in module_order:
        standalone_tensors = standalone_tensors_by_module.get(module, {})
        if not standalone_tensors:
            continue
        module_diffs: list[TensorDiff] = []
        for nemo_file in nemo_files:
            nemo_outputs_by_module = layer0_output_calls(nemo_captures[nemo_file])
            nemo_output_list = nemo_outputs_by_module.get(module, [])
            if not nemo_output_list:
                continue
            nemo_tensors = output_tensors(nemo_output_list[0])
            for tensor_path, nemo_tensor in sorted(nemo_tensors.items()):
                standalone_tensor = standalone_tensors.get(tensor_path)
                if standalone_tensor is None:
                    module_diffs.append(
                        TensorDiff(
                            module=module,
                            tensor_path=tensor_path,
                            nemo_file=nemo_file,
                            prompt_idx=worker_prompt_map[nemo_file],
                            nemo_shape=tuple(nemo_tensor.shape),
                            standalone_shape=(),
                            mean_abs=float("nan"),
                            max_abs=float("nan"),
                            mean_rel=float("nan"),
                            max_rel=float("nan"),
                            exact=False,
                            comparable=False,
                            reason="missing standalone tensor",
                        )
                    )
                    continue
                module_diffs.append(
                    compare_tensor(
                        module=module,
                        tensor_path=tensor_path,
                        nemo_file=nemo_file,
                        nemo_tensor=nemo_tensor,
                        standalone_tensor=standalone_tensor,
                        prompt_idx=worker_prompt_map[nemo_file],
                        prompt_lengths=prompt_lengths,
                    )
                )

        comparable = [diff for diff in module_diffs if diff.comparable]
        all_diffs.extend(module_diffs)
        if not comparable:
            print(f"module={module} comparable=0")
            continue

        worst = max(comparable, key=lambda item: item.max_abs)
        exact_count = sum(1 for diff in comparable if diff.exact)
        print(
            f"module={module} tensors={len(comparable)} exact={exact_count}/"
            f"{len(comparable)} worst_max_abs={worst.max_abs:.6e} "
            f"worst_mean_abs={worst.mean_abs:.6e} "
            f"worst_tensor={worst.tensor_path} "
            f"worst_file={Path(worst.nemo_file).name} "
            f"prompt={worst.prompt_idx}"
        )
        if first_divergent is None and worst.max_abs > args.threshold:
            first_divergent = worst

    if first_divergent is None:
        print(f"first_divergent_module=NONE threshold={args.threshold}")
    else:
        print_diff("first_divergent", first_divergent)

    comparable_diffs = [diff for diff in all_diffs if diff.comparable]
    worst_diffs = sorted(comparable_diffs, key=lambda item: item.max_abs, reverse=True)
    print(f"top_{args.top_k}_worst_diffs:")
    for diff in worst_diffs[: args.top_k]:
        print_diff("  diff", diff)

    skipped = [diff for diff in all_diffs if not diff.comparable]
    if skipped:
        print(f"skipped={len(skipped)}")
        for diff in skipped[: args.top_k]:
            print(
                f"  skipped module={diff.module} tensor={diff.tensor_path} "
                f"reason={diff.reason} nemo_shape={diff.nemo_shape} "
                f"standalone_shape={diff.standalone_shape} "
                f"file={Path(diff.nemo_file).name}"
            )


if __name__ == "__main__":
    main()
