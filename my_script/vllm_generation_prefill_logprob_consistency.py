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

r"""Compare vLLM generation logprobs against vLLM prefill prompt logprobs.

This diagnostic uses only vLLM:

1. Tokenize one or more prompts.
2. Generate a response of ``--max-new-tokens`` tokens and record the selected
   generation logprob for each generated token in one batch.
3. Concatenate each ``prompt + response`` and run vLLM again with
   ``prompt_logprobs=0`` in one batch.
4. Compare the selected generation logprobs against the prompt logprobs for the
   same generated-token positions in the concatenated sequence.

Batch-invariant mode is enabled by default and must be selected before vLLM is
imported:

    uv run --extra vllm python \
        my_script/vllm_generation_prefill_logprob_consistency.py
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Required so apply_model() can ship our hook-installer closure to the worker
# process via pickle (the default msgpack encoder rejects functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Parse the batch-invariant flag before importing vLLM so the worker process
# inherits the correct environment.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--no-batch-invariant", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()
if not _pre_args.no_batch_invariant:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"


def ensure_vllm_distribution_metadata() -> None:
    """Let source-tree vLLM imports satisfy importlib.metadata.version("vllm")."""
    try:
        importlib.metadata.version("vllm")
        return
    except importlib.metadata.PackageNotFoundError:
        pass

    metadata_root = Path(tempfile.gettempdir()) / "vllm_source_tree_metadata"
    dist_info = metadata_root / "vllm-0.0.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    metadata_path = dist_info / "METADATA"
    if not metadata_path.exists():
        metadata_path.write_text(
            "Metadata-Version: 2.1\nName: vllm\nVersion: 0.0.0\n",
            encoding="utf-8",
        )
    if str(metadata_root) not in sys.path:
        sys.path.insert(0, str(metadata_root))


ensure_vllm_distribution_metadata()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PROMPT = (
    "Solve this carefully: A rectangle has width 7 and height 9. "
    "What is its area? Answer in one short sentence."
)
DEFAULT_PROMPTS = [
    DEFAULT_PROMPT,
    "A baker has 12 muffins and sells 5. How many muffins remain?",
    "Write one concise sentence about why water freezes at low temperature.",
    "If a train travels 60 miles in 2 hours, what is its average speed?",
    "Name the capital of France and include only a short answer.",
    "Convert 3 kilograms to grams. Answer with the number and unit.",
    "A triangle has sides 3, 4, and 5. What kind of triangle is it?",
    "Complete the pattern with the next number: 2, 4, 8, 16,",
]


def install_rmsnorm_bi_residual_patch(model: Any) -> None:
    """Route fused add+RMSNorm through vLLM's batch-invariant Triton kernel."""
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant
    from vllm.model_executor.layers.layernorm import RMSNorm

    orig_forward_cuda = RMSNorm.forward_cuda

    def patched_forward_cuda(self: Any, x: Any, residual: Any | None = None) -> Any:
        if residual is not None:
            residual.add_(x)
            return (
                rms_norm_batch_invariant(
                    residual, self.weight.data, self.variance_epsilon
                ),
                residual,
            )
        return orig_forward_cuda(self, x, residual)

    RMSNorm.forward_cuda = patched_forward_cuda

    patched_count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module._forward_method = module.forward_cuda
            patched_count += 1
    print(f"[vllm-patch] rebound {patched_count} RMSNorm modules")
    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate tokens with vLLM, prefill prompt+response with vLLM, "
            "and compare selected-token logprobs."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help=(
            "Prompt to run. May be repeated. If omitted, built-in prompts are "
            "used according to --num-prompts."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of built-in prompts to use when --prompt is omitted.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--no-batch-invariant", action="store_true")
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--skip-rmsnorm-bi-residual-patch", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be at least 1")
    return args


def get_logprob_value(logprobs: dict[int, Any] | None, token_id: int) -> float:
    """Read the selected-token logprob from vLLM's logprob dictionary."""
    if logprobs is None:
        raise RuntimeError(f"missing logprob dictionary for token_id={token_id}")
    logprob = logprobs.get(token_id)
    if logprob is None:
        available = ", ".join(str(key) for key in list(logprobs)[:8])
        raise RuntimeError(
            f"selected token_id={token_id} not present in logprobs; "
            f"first available keys: {available}"
        )
    return float(logprob.logprob)


def prompt_logprobs_index(
    prompt_logprobs: list[Any], token_position: int, num_tokens: int
) -> int:
    """Return the prompt_logprobs index for a full-sequence token position."""
    if len(prompt_logprobs) == num_tokens:
        return token_position
    if len(prompt_logprobs) == num_tokens - 1:
        return token_position - 1
    raise RuntimeError(
        f"unexpected prompt_logprobs length {len(prompt_logprobs)} for "
        f"{num_tokens} prompt tokens"
    )


def make_default_prompts(num_prompts: int) -> list[str]:
    """Return built-in prompts, cycling when more are requested than defined."""
    prompts = []
    for prompt_index in range(num_prompts):
        base_prompt = DEFAULT_PROMPTS[prompt_index % len(DEFAULT_PROMPTS)]
        if prompt_index < len(DEFAULT_PROMPTS):
            prompts.append(base_prompt)
        else:
            prompts.append(f"{base_prompt} Use example #{prompt_index + 1}.")
    return prompts


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate difference metrics for comparison rows."""
    return {
        "mean_abs_diff": sum(row["abs_diff"] for row in rows) / len(rows),
        "max_abs_diff": max(row["abs_diff"] for row in rows),
        "mean_rel_diff": sum(row["rel_diff"] for row in rows) / len(rows),
        "max_rel_diff": max(row["rel_diff"] for row in rows),
    }


def make_sampling_params(
    *,
    args: argparse.Namespace,
    max_tokens: int,
    logprobs: int | None,
    prompt_logprobs: int | None,
) -> SamplingParams:
    """Build vLLM sampling parameters for generation or prompt-logprob prefill."""
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        seed=args.seed,
        ignore_eos=True,
    )


def main() -> None:
    """Run the diagnostic."""
    args = parse_args()
    tokenizer_name_or_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=args.trust_remote_code
    )
    prompts = (
        args.prompt
        if args.prompt is not None
        else make_default_prompts(args.num_prompts)
    )
    prompt_token_ids_list = [
        tokenizer.encode(prompt, add_special_tokens=True) for prompt in prompts
    ]

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": tokenizer_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": not args.no_enforce_eager,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_model_len": args.max_model_len,
    }
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend

    print(
        "[setup] "
        f"model={args.model} dtype={args.dtype} "
        f"batch_invariant={os.environ.get('VLLM_BATCH_INVARIANT', '0')} "
        f"attention_backend={args.attention_backend or 'default'}"
    )
    print(f"[prompts] count={len(prompts)}")
    for prompt_index, (prompt, prompt_token_ids) in enumerate(
        zip(prompts, prompt_token_ids_list)
    ):
        print(
            f"[prompt {prompt_index:02d}] "
            f"token_count={len(prompt_token_ids)} text={prompt!r}"
        )

    llm = LLM(**llm_kwargs)
    if (
        os.environ.get("VLLM_BATCH_INVARIANT") == "1"
        and not args.skip_rmsnorm_bi_residual_patch
    ):
        llm.apply_model(install_rmsnorm_bi_residual_patch)

    generation_params = make_sampling_params(
        args=args,
        max_tokens=args.max_new_tokens,
        logprobs=0,
        prompt_logprobs=None,
    )
    generation_outputs = llm.generate(
        [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids_list],
        sampling_params=generation_params,
        use_tqdm=False,
    )

    generation_results: list[dict[str, Any]] = []
    full_token_ids_list: list[list[int]] = []
    for prompt_index, output in enumerate(generation_outputs):
        generation = output.outputs[0]
        generated_token_ids = list(generation.token_ids)
        generation_logprobs = [
            get_logprob_value(step_logprobs, token_id)
            for step_logprobs, token_id in zip(generation.logprobs, generated_token_ids)
        ]
        generation_results.append(
            {
                "prompt_index": prompt_index,
                "generated_token_ids": generated_token_ids,
                "generated_text": generation.text,
                "generation_logprobs": generation_logprobs,
            }
        )
        full_token_ids_list.append(
            prompt_token_ids_list[prompt_index] + generated_token_ids
        )

    prefill_params = make_sampling_params(
        args=args,
        max_tokens=1,
        logprobs=None,
        prompt_logprobs=0,
    )
    prefill_outputs = llm.generate(
        [{"prompt_token_ids": token_ids} for token_ids in full_token_ids_list],
        sampling_params=prefill_params,
        use_tqdm=False,
    )

    all_rows: list[dict[str, Any]] = []
    prompt_results: list[dict[str, Any]] = []
    for prompt_index, (generation_result, prefill_output) in enumerate(
        zip(generation_results, prefill_outputs)
    ):
        prefill_prompt_logprobs = prefill_output.prompt_logprobs
        if prefill_prompt_logprobs is None:
            raise RuntimeError(
                f"vLLM did not return prompt_logprobs for prompt {prompt_index}"
            )

        rows: list[dict[str, Any]] = []
        generated_token_ids = generation_result["generated_token_ids"]
        generation_logprobs = generation_result["generation_logprobs"]
        for generation_index, (token_id, generation_logprob) in enumerate(
            zip(generated_token_ids, generation_logprobs)
        ):
            token_position = len(prompt_token_ids_list[prompt_index]) + generation_index
            logprobs_index = prompt_logprobs_index(
                prefill_prompt_logprobs,
                token_position,
                len(full_token_ids_list[prompt_index]),
            )
            prefill_logprob = get_logprob_value(
                prefill_prompt_logprobs[logprobs_index], token_id
            )
            abs_diff = abs(generation_logprob - prefill_logprob)
            rel_diff = abs_diff / max(
                abs(generation_logprob), abs(prefill_logprob), 1e-12
            )
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "index": generation_index,
                    "token_id": token_id,
                    "token_text": tokenizer.decode([token_id]),
                    "generation_logprob": generation_logprob,
                    "prefill_logprob": prefill_logprob,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )
        summary = summarize_rows(rows)
        all_rows.extend(rows)
        prompt_results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompts[prompt_index],
                "prompt_token_ids": prompt_token_ids_list[prompt_index],
                "generated_token_ids": generated_token_ids,
                "generated_text": generation_result["generated_text"],
                "rows": rows,
                "summary": summary,
            }
        )

    overall_summary = summarize_rows(all_rows)

    for prompt_result in prompt_results:
        prompt_index = prompt_result["prompt_index"]
        summary = prompt_result["summary"]
        print(
            f"[generation {prompt_index:02d}] "
            f"token_count={len(prompt_result['generated_token_ids'])} "
            f"text={prompt_result['generated_text']!r}"
        )
        print(
            f"[summary {prompt_index:02d}] "
            f"mean_abs_diff={summary['mean_abs_diff']:.9e} "
            f"max_abs_diff={summary['max_abs_diff']:.9e} "
            f"mean_rel_diff={summary['mean_rel_diff']:.9e} "
            f"max_rel_diff={summary['max_rel_diff']:.9e}"
        )
    print(
        "[summary all] "
        f"tokens={len(all_rows)} "
        f"mean_abs_diff={overall_summary['mean_abs_diff']:.9e} "
        f"max_abs_diff={overall_summary['max_abs_diff']:.9e} "
        f"mean_rel_diff={overall_summary['mean_rel_diff']:.9e} "
        f"max_rel_diff={overall_summary['max_rel_diff']:.9e}"
    )

    if args.output is not None:
        payload = {
            "model": args.model,
            "tokenizer": tokenizer_name_or_path,
            "prompts": prompts,
            "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1",
            "attention_backend": args.attention_backend,
            "results": prompt_results,
            "rows": all_rows,
            "summary": overall_summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[output] wrote {args.output}")


if __name__ == "__main__":
    main()
