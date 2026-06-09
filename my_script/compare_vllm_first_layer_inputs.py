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

"""Compare first-layer module input tensors between NeMo vLLM and standalone vLLM."""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TensorDiff:
    module: str
    tensor_path: str
    nemo_file: str
    prompt_idx: int | None
    nemo_shape: tuple[int, ...]
    standalone_shape: tuple[int, ...]
    mean_abs: float
    max_abs: float
    mean_rel: float
    max_rel: float
    exact: bool
    comparable: bool
    reason: str = ""


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


def load_pt(path: str | Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_capture(payload: dict[str, Any]) -> dict[str, Any]:
    if "capture" in payload and isinstance(payload["capture"], dict):
        return payload["capture"]
    if "module_input_calls_by_layer" in payload:
        return payload
    raise KeyError(f"Could not find debug tensor capture keys: {sorted(payload)[:20]}")


def layer0_input_calls(capture: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    calls_by_layer = capture.get("module_input_calls_by_layer", {})
    return calls_by_layer.get(0) or calls_by_layer.get("0") or {}


def iter_tensors(value: Any, prefix: str) -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, (tuple, list)):
        result: list[tuple[str, torch.Tensor]] = []
        for idx, item in enumerate(value):
            result.extend(iter_tensors(item, f"{prefix}.{idx}"))
        return result
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(iter_tensors(value[key], f"{prefix}.{key}"))
        return result
    return []


def call_tensors(call: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = {}
    for path, tensor in iter_tensors(call.get("args", ()), "args"):
        tensors[path] = tensor
    for path, tensor in iter_tensors(call.get("kwargs", {}), "kwargs"):
        tensors[path] = tensor
    return tensors


def cat_standalone_tensors(
    calls: list[dict[str, Any]], *, chunk_count: int, num_layers: int
) -> dict[str, torch.Tensor]:
    if len(calls) == chunk_count:
        selected_calls = calls
    elif num_layers > 0 and len(calls) == chunk_count * num_layers:
        # Some vLLM modules, notably rotary embeddings, are shared across all
        # decoder layers. The hook is registered while walking layer 0, but it
        # fires once per layer. Pick the layer-0 call from each prefill chunk.
        selected_calls = [
            calls[chunk_idx * num_layers] for chunk_idx in range(chunk_count)
        ]
    else:
        selected_calls = calls[:chunk_count]

    tensors_by_path: dict[str, list[torch.Tensor]] = {}
    for call in selected_calls:
        for path, tensor in call_tensors(call).items():
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


def tensor_stats(
    nemo_tensor: torch.Tensor, standalone_tensor: torch.Tensor
) -> dict[str, Any]:
    nemo_float = nemo_tensor.to(torch.float32)
    standalone_float = standalone_tensor.to(torch.float32)
    diff = nemo_float - standalone_float
    abs_diff = diff.abs()
    denom = torch.maximum(
        torch.maximum(nemo_float.abs(), standalone_float.abs()),
        torch.full_like(abs_diff, 1e-12),
    )
    rel_diff = abs_diff / denom
    return {
        "mean_abs": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
        "mean_rel": float(rel_diff.mean().item()),
        "max_rel": float(rel_diff.max().item()),
        "exact": bool(torch.equal(nemo_tensor, standalone_tensor)),
    }


def prompt_slices(prompt_lengths: list[int]) -> list[slice]:
    starts = [0]
    for length in prompt_lengths[:-1]:
        starts.append(starts[-1] + length)
    return [
        slice(start, start + length) for start, length in zip(starts, prompt_lengths)
    ]


def standalone_prompt_slice(
    standalone_tensor: torch.Tensor,
    nemo_tensor: torch.Tensor,
    prompt_idx: int,
    prompt_lengths: list[int],
) -> torch.Tensor | None:
    if standalone_tensor.shape == nemo_tensor.shape:
        return standalone_tensor

    if nemo_tensor.ndim == 0 or standalone_tensor.ndim == 0:
        return None

    if standalone_tensor.shape[0] == sum(prompt_lengths):
        slices = prompt_slices(prompt_lengths)
        sliced = standalone_tensor[slices[prompt_idx]]
        if sliced.shape == nemo_tensor.shape:
            return sliced

    if standalone_tensor.shape[0] == len(prompt_lengths) and nemo_tensor.shape[0] == 1:
        sliced = standalone_tensor[prompt_idx : prompt_idx + 1]
        if sliced.shape == nemo_tensor.shape:
            return sliced

    return None


def best_prompt_idx(
    nemo_tensor: torch.Tensor,
    standalone_tensor: torch.Tensor,
    prompt_lengths: list[int],
) -> tuple[int | None, dict[str, Any] | None]:
    best_idx = None
    best_stats = None
    for prompt_idx in range(len(prompt_lengths)):
        sliced = standalone_prompt_slice(
            standalone_tensor, nemo_tensor, prompt_idx, prompt_lengths
        )
        if sliced is None:
            continue
        stats = tensor_stats(nemo_tensor, sliced)
        if best_stats is None or (stats["max_abs"], stats["mean_abs"]) < (
            best_stats["max_abs"],
            best_stats["mean_abs"],
        ):
            best_idx = prompt_idx
            best_stats = stats
    return best_idx, best_stats


def choose_worker_prompt_map(
    nemo_files: list[str],
    nemo_captures: dict[str, dict[str, Any]],
    standalone_tensors_by_module: dict[str, dict[str, torch.Tensor]],
    prompt_lengths: list[int],
) -> dict[str, int | None]:
    standalone_tensors = standalone_tensors_by_module.get("<layer>", {})
    standalone_hidden = standalone_tensors.get("args.1")
    if standalone_hidden is None:
        standalone_hidden = standalone_tensors.get("args.0")
    mapping: dict[str, int | None] = {}
    if standalone_hidden is None:
        return {path: None for path in nemo_files}

    for nemo_file in nemo_files:
        nemo_calls = layer0_input_calls(nemo_captures[nemo_file])
        nemo_layer_call = nemo_calls.get("<layer>", [{}])[0]
        nemo_tensors = call_tensors(nemo_layer_call)
        nemo_hidden = nemo_tensors.get("args.1")
        if nemo_hidden is None:
            nemo_hidden = nemo_tensors.get("args.0")
        if nemo_hidden is None:
            mapping[nemo_file] = None
            continue
        prompt_idx, _ = best_prompt_idx(nemo_hidden, standalone_hidden, prompt_lengths)
        mapping[nemo_file] = prompt_idx
    return mapping


def compare_tensor(
    *,
    module: str,
    tensor_path: str,
    nemo_file: str,
    nemo_tensor: torch.Tensor,
    standalone_tensor: torch.Tensor,
    prompt_idx: int | None,
    prompt_lengths: list[int],
) -> TensorDiff:
    if prompt_idx is None:
        best_idx, stats = best_prompt_idx(
            nemo_tensor, standalone_tensor, prompt_lengths
        )
        if stats is None:
            return TensorDiff(
                module=module,
                tensor_path=tensor_path,
                nemo_file=nemo_file,
                prompt_idx=None,
                nemo_shape=tuple(nemo_tensor.shape),
                standalone_shape=tuple(standalone_tensor.shape),
                mean_abs=float("nan"),
                max_abs=float("nan"),
                mean_rel=float("nan"),
                max_rel=float("nan"),
                exact=False,
                comparable=False,
                reason="shape mismatch",
            )
        prompt_idx = best_idx
    else:
        sliced = standalone_prompt_slice(
            standalone_tensor, nemo_tensor, prompt_idx, prompt_lengths
        )
        if sliced is None:
            return TensorDiff(
                module=module,
                tensor_path=tensor_path,
                nemo_file=nemo_file,
                prompt_idx=prompt_idx,
                nemo_shape=tuple(nemo_tensor.shape),
                standalone_shape=tuple(standalone_tensor.shape),
                mean_abs=float("nan"),
                max_abs=float("nan"),
                mean_rel=float("nan"),
                max_rel=float("nan"),
                exact=False,
                comparable=False,
                reason="shape mismatch",
            )
        stats = tensor_stats(nemo_tensor, sliced)

    return TensorDiff(
        module=module,
        tensor_path=tensor_path,
        nemo_file=nemo_file,
        prompt_idx=prompt_idx,
        nemo_shape=tuple(nemo_tensor.shape),
        standalone_shape=tuple(standalone_tensor.shape),
        mean_abs=stats["mean_abs"],
        max_abs=stats["max_abs"],
        mean_rel=stats["mean_rel"],
        max_rel=stats["max_rel"],
        exact=stats["exact"],
        comparable=True,
    )


def print_diff(prefix: str, diff: TensorDiff) -> None:
    print(
        f"{prefix} module={diff.module} tensor={diff.tensor_path} "
        f"prompt={diff.prompt_idx} exact={diff.exact} "
        f"mean_abs={diff.mean_abs:.6e} max_abs={diff.max_abs:.6e} "
        f"mean_rel={diff.mean_rel:.6e} max_rel={diff.max_rel:.6e} "
        f"nemo_shape={diff.nemo_shape} standalone_shape={diff.standalone_shape} "
        f"file={Path(diff.nemo_file).name}"
    )


def main() -> None:
    args = parse_args()
    token_dump = load_pt(args.token_dump)
    prompt_token_ids = token_dump[args.prompt_key]
    prompt_lengths = [len(row) for row in prompt_token_ids]

    standalone_payload = load_pt(args.standalone)
    standalone_capture = extract_capture(standalone_payload)
    standalone_calls = layer0_input_calls(standalone_capture)
    module_order = list(standalone_calls)
    standalone_layer_calls = standalone_calls.get("<layer>", [])
    chunk_count = len(standalone_layer_calls)
    num_layers = int(standalone_capture.get("num_layers", 0))
    standalone_tensors_by_module = {
        module: cat_standalone_tensors(
            calls, chunk_count=chunk_count, num_layers=num_layers
        )
        for module, calls in standalone_calls.items()
    }

    nemo_files = sorted(glob.glob(args.nemo_glob))
    if not nemo_files:
        raise FileNotFoundError(args.nemo_glob)
    nemo_captures = {path: extract_capture(load_pt(path)) for path in nemo_files}
    worker_prompt_map = choose_worker_prompt_map(
        nemo_files, nemo_captures, standalone_tensors_by_module, prompt_lengths
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
            nemo_calls = layer0_input_calls(nemo_captures[nemo_file])
            nemo_call_list = nemo_calls.get(module, [])
            if not nemo_call_list:
                continue
            nemo_tensors = call_tensors(nemo_call_list[0])
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
