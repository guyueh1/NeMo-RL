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

"""Compare residual-stream and final-logprob error distributions across runs."""

import argparse
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from compare import diff_stats, last_token_offsets, layer_entry_stream


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: str


@dataclass(frozen=True)
class DiffSummary:
    n: int
    max_abs: float
    mean_abs: float
    q50: float
    q90: float
    q99: float
    q999: float
    nonzero_frac: float
    cos_sim: float


@dataclass(frozen=True)
class BaselineDelta:
    mean_delta: float
    max_worse: float
    max_better: float
    worse_frac: float
    better_frac: float
    q99_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron", required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="vLLM capture to compare. May be supplied multiple times.",
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        help="Run label used as the baseline for better/worse deltas.",
    )
    parser.add_argument(
        "--layers",
        default="0-5",
        help="Layer list/ranges to inspect, for example '0-5,8'.",
    )
    parser.add_argument(
        "--top-prompts",
        type=int,
        default=5,
        help="Number of worst last-token prompts to print per layer.",
    )
    return parser.parse_args()


def parse_run_spec(value: str) -> RunSpec:
    if "=" not in value:
        raise ValueError(f"--run must be LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    if not label or not path:
        raise ValueError(f"--run must be LABEL=PATH, got {value!r}")
    return RunSpec(label=label, path=path)


def parse_layers(value: str) -> list[int]:
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_payload(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dict payload")
    return payload


def trim_pair(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != b.dim():
        min_dims = min(a.dim(), b.dim())
        a = a.reshape(-1, *a.shape[-min_dims + 1 :]) if min_dims > 1 else a.reshape(-1)
        b = b.reshape(-1, *b.shape[-min_dims + 1 :]) if min_dims > 1 else b.reshape(-1)
    shape = tuple(min(left, right) for left, right in zip(a.shape, b.shape))
    slices = tuple(slice(0, size) for size in shape)
    return a[slices], b[slices]


def quantile(diff: torch.Tensor, q: float) -> float:
    flat = diff.reshape(-1)
    if flat.numel() == 0:
        return float("nan")
    return float(torch.quantile(flat, q))


def summarize(a: torch.Tensor, b: torch.Tensor) -> tuple[DiffSummary, torch.Tensor]:
    a, b = trim_pair(a, b)
    diff = (a.float() - b.float()).abs()
    flat = diff.reshape(-1)
    stats = diff_stats(a, b)
    summary = DiffSummary(
        n=flat.numel(),
        max_abs=float(flat.max()) if flat.numel() else float("nan"),
        mean_abs=float(flat.mean()) if flat.numel() else float("nan"),
        q50=quantile(diff, 0.50),
        q90=quantile(diff, 0.90),
        q99=quantile(diff, 0.99),
        q999=quantile(diff, 0.999),
        nonzero_frac=float((flat > 0).float().mean()) if flat.numel() else float("nan"),
        cos_sim=stats["cos_sim"],
    )
    return summary, diff


def summarize_delta(diff: torch.Tensor, baseline_diff: torch.Tensor) -> BaselineDelta:
    diff, baseline_diff = trim_pair(diff, baseline_diff)
    delta = diff.float() - baseline_diff.float()
    flat = delta.reshape(-1)
    if flat.numel() == 0:
        return BaselineDelta(
            mean_delta=float("nan"),
            max_worse=float("nan"),
            max_better=float("nan"),
            worse_frac=float("nan"),
            better_frac=float("nan"),
            q99_delta=float("nan"),
        )
    return BaselineDelta(
        mean_delta=float(flat.mean()),
        max_worse=float(flat.max()),
        max_better=float((-flat).max()),
        worse_frac=float((flat > 0).float().mean()),
        better_frac=float((flat < 0).float().mean()),
        q99_delta=float(torch.quantile(flat, 0.99)),
    )


def print_summary_header(prefix: str) -> None:
    print(
        f"{prefix:>12s} {'n':>11s} {'max':>11s} {'mean':>11s} "
        f"{'q50':>11s} {'q90':>11s} {'q99':>11s} {'q999':>11s} "
        f"{'nz_frac':>9s} {'cos':>10s}"
    )


def print_summary(label: str, summary: DiffSummary) -> None:
    print(
        f"{label:>12s} {summary.n:>11d} {summary.max_abs:>11.6f} "
        f"{summary.mean_abs:>11.4e} {summary.q50:>11.6f} "
        f"{summary.q90:>11.6f} {summary.q99:>11.6f} "
        f"{summary.q999:>11.6f} {summary.nonzero_frac:>9.6f} "
        f"{summary.cos_sim:>10.6f}"
    )


def print_delta_header(prefix: str) -> None:
    print(
        f"{prefix:>12s} {'mean_delta':>12s} {'max_worse':>11s} "
        f"{'max_better':>11s} {'worse%':>8s} {'better%':>8s} {'q99_delta':>11s}"
    )


def print_delta(label: str, delta: BaselineDelta) -> None:
    print(
        f"{label:>12s} {delta.mean_delta:>12.4e} {delta.max_worse:>11.6f} "
        f"{delta.max_better:>11.6f} {100.0 * delta.worse_frac:>8.3f} "
        f"{100.0 * delta.better_frac:>8.3f} {delta.q99_delta:>11.6f}"
    )


def compute_m_logprobs(megatron: dict, vocab: int) -> torch.Tensor:
    logits = megatron.get("last_token_logits")
    if not isinstance(logits, torch.Tensor):
        raise KeyError("Megatron payload is missing last_token_logits")
    logits = logits.float()[..., :vocab]
    return torch.log_softmax(logits, dim=-1)


def logprob_tensor(
    run: dict, megatron_logprobs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    v_logprobs = run.get("next_token_logprobs")
    if not isinstance(v_logprobs, torch.Tensor):
        raise KeyError("vLLM payload is missing next_token_logprobs")
    n = min(v_logprobs.shape[0], megatron_logprobs.shape[0])
    vocab = min(v_logprobs.shape[-1], megatron_logprobs.shape[-1])
    return v_logprobs[:n, :vocab].float(), megatron_logprobs[:n, :vocab].float()


def print_worst_prompts(
    diff: torch.Tensor, seq_lens: Iterable[int], top_k: int
) -> None:
    seq_lens = list(seq_lens)
    if top_k <= 0 or not seq_lens or diff.dim() < 2:
        return
    offsets = last_token_offsets(seq_lens)
    if diff.shape[0] <= int(offsets[-1]):
        return
    last = diff[offsets]
    prompt_max = last.reshape(last.shape[0], -1).max(dim=1).values
    prompt_mean = last.reshape(last.shape[0], -1).mean(dim=1)
    top_values, top_indices = torch.topk(prompt_max, k=min(top_k, prompt_max.numel()))
    parts = []
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        parts.append(f"p{index}:{value:.6f}/{float(prompt_mean[index]):.2e}")
    print(f"      worst last-token prompts: {' '.join(parts)}")


def main() -> None:
    args = parse_args()
    run_specs = [parse_run_spec(value) for value in args.run]
    if not run_specs:
        raise ValueError("provide at least one --run LABEL=PATH")

    megatron = load_payload(args.megatron)
    runs = [(spec, load_payload(spec.path)) for spec in run_specs]
    baseline_label = args.baseline_label or run_specs[0].label
    layers = parse_layers(args.layers)
    seq_lens = runs[0][1].get("seq_lens") or megatron.get("seq_lens") or []

    v_vocab = min(
        int(run.get("next_token_logprobs").shape[-1])
        for _, run in runs
        if isinstance(run.get("next_token_logprobs"), torch.Tensor)
    )
    m_logprobs = compute_m_logprobs(megatron, vocab=v_vocab)

    print("=" * 120)
    print(f"Megatron capture: {args.megatron}")
    print(f"Baseline label  : {baseline_label}")
    print(f"Seq lens        : {seq_lens}")
    print(f"Layers          : {layers}")
    print("=" * 120)

    print("\nFinal next-token logprob distributions vs Megatron")
    print_summary_header("run")
    logprob_diffs = {}
    for spec, run in runs:
        v_logprobs, m_logprobs_trimmed = logprob_tensor(run, m_logprobs)
        summary, diff = summarize(v_logprobs, m_logprobs_trimmed)
        logprob_diffs[spec.label] = diff
        print_summary(spec.label, summary)

    if baseline_label in logprob_diffs:
        print("\nFinal logprob deltas versus baseline abs error")
        print_delta_header("run")
        baseline_diff = logprob_diffs[baseline_label]
        for spec, _ in runs:
            if spec.label == baseline_label:
                continue
            print_delta(
                spec.label,
                summarize_delta(logprob_diffs[spec.label], baseline_diff),
            )

    for layer_idx in layers:
        print("\n" + "=" * 120)
        print(f"Layer {layer_idx} entry residual-stream distributions vs Megatron")
        print_summary_header("run")
        layer_diffs = {}
        for spec, run in runs:
            v_tensor, v_selector = layer_entry_stream(
                run, layer_idx, seq_lens, is_vllm=True
            )
            m_tensor, m_selector = layer_entry_stream(
                megatron, layer_idx, seq_lens, is_vllm=False
            )
            if v_tensor is None or m_tensor is None:
                print(
                    f"{spec.label:>12s} MISSING "
                    f"v_selector={v_selector} m_selector={m_selector}"
                )
                continue
            summary, diff = summarize(v_tensor, m_tensor)
            layer_diffs[spec.label] = diff
            print_summary(spec.label, summary)
            print_worst_prompts(diff, seq_lens, args.top_prompts)

        if baseline_label in layer_diffs:
            print(f"\nLayer {layer_idx} deltas versus baseline abs error")
            print_delta_header("run")
            baseline_diff = layer_diffs[baseline_label]
            for spec, _ in runs:
                if spec.label == baseline_label or spec.label not in layer_diffs:
                    continue
                print_delta(
                    spec.label,
                    summarize_delta(layer_diffs[spec.label], baseline_diff),
                )


if __name__ == "__main__":
    main()
