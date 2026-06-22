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

import argparse
import os
import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ray
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import (
    GRPOSaveState,
    MasterConfig,
    _should_use_async_rollouts,
    _should_use_nemo_gym,
    refit_policy_generation,
    setup,
)
from nemo_rl.algorithms.loss import ClippedPGLossDataDict
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.dataloader import MultipleDataloaderWrapper
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import Logger, get_next_experiment_dir
from nemo_rl.utils.timer import Timer

TokenizerType = PreTrainedTokenizerBase


@dataclass
class LogprobComparisonTotals:
    token_count: int = 0
    abs_sum: float = 0.0
    rel_sum: float = 0.0
    signed_sum: float = 0.0
    max_abs: float = 0.0
    max_rel: float = 0.0

    def update(self, metrics: dict[str, Any]) -> None:
        count = int(metrics["num_compared_tokens"])
        self.token_count += count
        self.abs_sum += float(metrics["sum_abs_diff"])
        self.rel_sum += float(metrics["sum_rel_diff"])
        self.signed_sum += float(metrics["sum_signed_diff"])
        self.max_abs = max(self.max_abs, float(metrics["max_abs_diff"]))
        self.max_rel = max(self.max_rel, float(metrics["max_rel_diff"]))

    def as_metrics(self) -> dict[str, float | int]:
        if self.token_count == 0:
            return {
                "num_compared_tokens": 0,
                "mean_abs_diff": 0.0,
                "mean_rel_diff": 0.0,
                "mean_signed_diff": 0.0,
                "max_abs_diff": 0.0,
                "max_rel_diff": 0.0,
            }

        return {
            "num_compared_tokens": self.token_count,
            "mean_abs_diff": self.abs_sum / self.token_count,
            "mean_rel_diff": self.rel_sum / self.token_count,
            "mean_signed_diff": self.signed_sum / self.token_count,
            "max_abs_diff": self.max_abs,
            "max_rel_diff": self.max_rel,
        }


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run GRPO rollouts and compare generation logprobs against policy "
            "forward logprobs without training."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a GRPO YAML config file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of worst-token differences to print per step.",
    )
    parser.add_argument(
        "--tensor-dump-dir",
        type=str,
        default=None,
        help="Directory where generation and policy tensor captures will be saved.",
    )
    parser.add_argument(
        "--tensor-dump-prefix",
        type=str,
        default="logprob_comparison",
        help="Filename prefix for tensor dump artifacts.",
    )
    parser.add_argument(
        "--tensor-dump-max-steps",
        type=int,
        default=1,
        help=(
            "Number of comparison steps to dump tensors for. Use a negative value "
            "to dump every step."
        ),
    )
    parser.add_argument(
        "--tensor-dump-max-calls-per-module",
        type=int,
        default=1,
        help="Maximum tensor snapshots to retain per hooked module per step.",
    )
    parser.add_argument(
        "--vllm-prefill-check-first-batch",
        action="store_true",
        help=(
            "For the first rollout batch, compare vLLM rollout logprobs against "
            "vLLM prompt logprobs from scoring prompt+response as one prefill."
        ),
    )
    parser.add_argument(
        "--first-batch-token-dump",
        type=str,
        default=None,
        help=(
            "For the first rollout batch, save prompt/response token ids and "
            "per-response-token prefix rows that can be replayed by the "
            "offline vLLM/Megatron forward scripts."
        ),
    )
    parser.add_argument(
        "--inspect-vllm-layernorm-impl",
        action="store_true",
        help=(
            "After vLLM generation setup, print the runtime implementation bound "
            "to layer 0 post_attention_layernorm."
        ),
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _set_message_log_training_fields(batch: BatchedDataDict[DatumSpec]) -> None:
    for message_log in batch["message_log"]:
        for message in message_log:
            if message["role"] == "assistant":
                message["token_loss_mask"] = torch.ones_like(message["token_ids"])
            else:
                message["token_loss_mask"] = torch.zeros_like(message["token_ids"])

            if "generation_logprobs" not in message:
                message["generation_logprobs"] = torch.zeros_like(
                    message["token_ids"], dtype=torch.float32
                )


def _build_logprob_data(
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    master_config: MasterConfig,
) -> tuple[
    BatchedDataDict[ClippedPGLossDataDict],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _set_message_log_training_fields(batch)
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    extra_multimodal_data = flat_messages.get_multimodal_dict(as_tensors=False)
    logprob_data = BatchedDataDict[ClippedPGLossDataDict](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat_messages["token_loss_mask"],
            "sample_mask": batch["loss_multiplier"],
            **extra_multimodal_data,
        }
    )
    logprob_data.to("cpu")
    return (
        logprob_data,
        flat_messages["generation_logprobs"].to(torch.float32),
        flat_messages["token_loss_mask"],
        flat_messages["token_ids"],
    )


def _build_calibration_data(
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    master_config: MasterConfig,
) -> BatchedDataDict[ClippedPGLossDataDict]:
    calib_flat, calib_input_lengths = batched_message_log_to_flat_message(
        batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    calibration_data = BatchedDataDict[ClippedPGLossDataDict](
        {
            "input_ids": calib_flat["token_ids"],
            "input_lengths": calib_input_lengths,
        }
    )
    calibration_data.update(calib_flat.get_multimodal_dict(as_tensors=False))
    calibration_data.to("cpu")
    return calibration_data


def _compare_logprobs(
    *,
    generation_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    if generation_logprobs.shape[0] != policy_logprobs.shape[0]:
        raise ValueError(
            "Generation and policy logprobs have different batch sizes: "
            f"{generation_logprobs.shape} vs {policy_logprobs.shape}"
        )

    original_generation_shape = tuple(generation_logprobs.shape)
    original_policy_shape = tuple(policy_logprobs.shape)
    seq_len = min(generation_logprobs.shape[1], policy_logprobs.shape[1])
    shape_mismatch = original_generation_shape != original_policy_shape

    generation_shifted = generation_logprobs[:, 1:seq_len].to(torch.float32)
    policy_shifted = policy_logprobs[:, 1:seq_len].to(torch.float32)
    token_mask_shifted = token_mask[:, 1:seq_len]
    input_ids_shifted = input_ids[:, 1:seq_len]
    compare_mask = (token_mask_shifted * sample_mask.unsqueeze(-1)).bool()

    diff = policy_shifted - generation_shifted
    abs_diff = diff.abs()
    denom = torch.maximum(
        torch.maximum(policy_shifted.abs(), generation_shifted.abs()),
        torch.full_like(abs_diff, 1e-12),
    )
    rel_diff = abs_diff / denom

    if not compare_mask.any():
        return {
            "num_compared_tokens": 0,
            "sum_abs_diff": 0.0,
            "sum_rel_diff": 0.0,
            "sum_signed_diff": 0.0,
            "mean_abs_diff": 0.0,
            "mean_rel_diff": 0.0,
            "mean_signed_diff": 0.0,
            "max_abs_diff": 0.0,
            "max_rel_diff": 0.0,
            "shape_mismatch": shape_mismatch,
            "generation_shape": original_generation_shape,
            "policy_shape": original_policy_shape,
            "worst_tokens": [],
        }

    masked_abs = abs_diff[compare_mask]
    masked_rel = rel_diff[compare_mask]
    masked_signed = diff[compare_mask]
    masked_positions = compare_mask.nonzero(as_tuple=False)

    worst_tokens = []
    num_worst = min(top_k, masked_abs.numel())
    if num_worst > 0:
        top_abs, top_indices = torch.topk(masked_abs, num_worst)
        for rank, (abs_value, masked_index) in enumerate(
            zip(top_abs.tolist(), top_indices.tolist()), start=1
        ):
            row, shifted_pos = masked_positions[masked_index].tolist()
            pos = shifted_pos + 1
            worst_tokens.append(
                {
                    "rank": rank,
                    "sample_idx": int(row),
                    "position": int(pos),
                    "token_id": int(input_ids_shifted[row, shifted_pos].item()),
                    "generation_logprob": float(
                        generation_shifted[row, shifted_pos].item()
                    ),
                    "policy_logprob": float(policy_shifted[row, shifted_pos].item()),
                    "abs_diff": float(abs_value),
                    "rel_diff": float(masked_rel[masked_index].item()),
                }
            )

    return {
        "num_compared_tokens": int(masked_abs.numel()),
        "sum_abs_diff": float(masked_abs.sum().item()),
        "sum_rel_diff": float(masked_rel.sum().item()),
        "sum_signed_diff": float(masked_signed.sum().item()),
        "mean_abs_diff": float(masked_abs.mean().item()),
        "mean_rel_diff": float(masked_rel.mean().item()),
        "mean_signed_diff": float(masked_signed.mean().item()),
        "max_abs_diff": float(masked_abs.max().item()),
        "max_rel_diff": float(masked_rel.max().item()),
        "shape_mismatch": shape_mismatch,
        "generation_shape": original_generation_shape,
        "policy_shape": original_policy_shape,
        "worst_tokens": worst_tokens,
    }


def _logger_metrics(
    comparison_metrics: dict[str, Any], rollout_metrics: dict[str, Any]
) -> dict[str, int | float | bool]:
    metrics = {
        key: value
        for key, value in comparison_metrics.items()
        if isinstance(value, (int, float, bool))
    }
    for key, value in rollout_metrics.items():
        if isinstance(value, (int, float, bool)):
            metrics[f"rollout/{key}"] = value
    return metrics


def _print_step_summary(
    *,
    step: int,
    max_num_steps: int,
    comparison_metrics: dict[str, Any],
    rollout_metrics: dict[str, Any],
) -> None:
    print(
        "[logprob-comparison] "
        f"step={step}/{max_num_steps} "
        f"tokens={comparison_metrics['num_compared_tokens']} "
        f"mean_abs={comparison_metrics['mean_abs_diff']:.6e} "
        f"max_abs={comparison_metrics['max_abs_diff']:.6e} "
        f"mean_rel={comparison_metrics['mean_rel_diff']:.6e} "
        f"max_rel={comparison_metrics['max_rel_diff']:.6e} "
        f"mean_gen_tokens={rollout_metrics.get('mean_gen_tokens_per_sample', 0.0):.2f}",
        flush=True,
    )
    if comparison_metrics["shape_mismatch"]:
        print(
            "[logprob-comparison] shape mismatch: "
            f"generation={comparison_metrics['generation_shape']} "
            f"policy={comparison_metrics['policy_shape']}",
            flush=True,
        )
    if comparison_metrics["worst_tokens"]:
        print("[logprob-comparison] worst token diffs:", flush=True)
        for item in comparison_metrics["worst_tokens"]:
            print(
                "  "
                f"#{item['rank']} sample={item['sample_idx']} "
                f"pos={item['position']} token={item['token_id']} "
                f"gen={item['generation_logprob']:.8e} "
                f"policy={item['policy_logprob']:.8e} "
                f"abs={item['abs_diff']:.8e} "
                f"rel={item['rel_diff']:.8e}",
                flush=True,
            )


def _latest_timer_seconds(timer: Timer, label: str) -> float | None:
    try:
        return timer.get_latest_elapsed(label)
    except (KeyError, IndexError):
        return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _print_timing_summary(
    *,
    step: int,
    timer: Timer,
    rollout_metrics: dict[str, Any],
    batch_size: int,
    master_config: MasterConfig,
) -> None:
    rollout_seconds = _latest_timer_seconds(timer, "rollout_generation")
    total_generated_tokens = (
        float(rollout_metrics.get("mean_gen_tokens_per_sample", 0.0)) * batch_size
    )
    if rollout_seconds is not None and rollout_seconds > 0.0:
        rollout_tokens_per_second = total_generated_tokens / rollout_seconds
    else:
        rollout_tokens_per_second = 0.0

    num_gpus = int(master_config.cluster["num_nodes"]) * int(
        master_config.cluster["gpus_per_node"]
    )
    rollout_tokens_per_second_per_gpu = (
        rollout_tokens_per_second / num_gpus if num_gpus > 0 else 0.0
    )

    print(
        "[logprob-comparison-timing] "
        f"step={step} "
        f"prepare={_format_seconds(_latest_timer_seconds(timer, 'prepare_for_generation/total'))} "
        f"rollout={_format_seconds(rollout_seconds)} "
        f"vllm_prefill_check={_format_seconds(_latest_timer_seconds(timer, 'vllm_prefill_check'))} "
        f"logprob_data={_format_seconds(_latest_timer_seconds(timer, 'logprob_data_processing'))} "
        f"logprob_prep={_format_seconds(_latest_timer_seconds(timer, 'logprob_inference_prep'))} "
        f"policy_logprobs={_format_seconds(_latest_timer_seconds(timer, 'policy_logprobs'))} "
        f"gen_tokens={total_generated_tokens:.0f} "
        f"gen_tokens_per_sec={rollout_tokens_per_second:.2f} "
        f"gen_tokens_per_sec_per_gpu={rollout_tokens_per_second_per_gpu:.2f}",
        flush=True,
    )


def _print_vllm_prefill_check_summary(
    *,
    step: int,
    comparison_metrics: dict[str, Any],
) -> None:
    print(
        "[vllm-prefill-check] "
        f"step={step} "
        f"tokens={comparison_metrics['num_compared_tokens']} "
        f"mean_abs={comparison_metrics['mean_abs_diff']:.6e} "
        f"max_abs={comparison_metrics['max_abs_diff']:.6e} "
        f"mean_rel={comparison_metrics['mean_rel_diff']:.6e} "
        f"max_rel={comparison_metrics['max_rel_diff']:.6e} "
        f"mean_signed={comparison_metrics['mean_signed_diff']:.6e}",
        flush=True,
    )
    if comparison_metrics["shape_mismatch"]:
        print(
            "[vllm-prefill-check] shape mismatch: "
            f"generation={comparison_metrics['generation_shape']} "
            f"prefill={comparison_metrics['policy_shape']}",
            flush=True,
        )
    if comparison_metrics["worst_tokens"]:
        print("[vllm-prefill-check] worst token diffs:", flush=True)
        for item in comparison_metrics["worst_tokens"]:
            print(
                "  "
                f"#{item['rank']} sample={item['sample_idx']} "
                f"pos={item['position']} token={item['token_id']} "
                f"generation={item['generation_logprob']:.8e} "
                f"prefill={item['policy_logprob']:.8e} "
                f"abs={item['abs_diff']:.8e} "
                f"rel={item['rel_diff']:.8e}",
                flush=True,
            )


def _should_dump_tensor_step(
    tensor_dump_dir: Optional[str],
    tensor_dump_max_steps: int,
    tensor_dump_steps_done: int,
) -> bool:
    if tensor_dump_dir is None:
        return False
    return tensor_dump_max_steps < 0 or tensor_dump_steps_done < tensor_dump_max_steps


def _visit_debug_results(result: Any, paths: list[str], hook_counts: list[int]) -> None:
    if isinstance(result, dict):
        if "path" in result:
            paths.append(str(result["path"]))
        if "num_hooks" in result:
            hook_counts.append(int(result["num_hooks"]))
        return
    if isinstance(result, list):
        for item in result:
            _visit_debug_results(item, paths, hook_counts)


def _print_debug_result_summary(label: str, action: str, result: Any) -> None:
    paths: list[str] = []
    hook_counts: list[int] = []
    _visit_debug_results(result, paths, hook_counts)
    if paths:
        shown_paths = ", ".join(paths[:4])
        extra = "" if len(paths) <= 4 else f", ... ({len(paths)} files total)"
        print(
            f"[tensor-dump] {label} {action}: {shown_paths}{extra}",
            flush=True,
        )
        return
    if hook_counts:
        print(
            f"[tensor-dump] {label} {action}: installed {sum(hook_counts)} hooks "
            f"across {len(hook_counts)} workers",
            flush=True,
        )
        return
    print(f"[tensor-dump] {label} {action}: {result}", flush=True)


def _call_debug_method(
    target: Any,
    *,
    label: str,
    method_name: str,
    **kwargs: Any,
) -> Any:
    method = getattr(target, method_name, None)
    if method is None:
        raise RuntimeError(
            f"{label} does not expose {method_name}; cannot dump tensors for it."
        )
    result = method(**kwargs)
    _print_debug_result_summary(label, method_name, result)
    return result


def _install_tensor_dump_hooks(
    *,
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    tensor_dump_dir: Optional[str],
    tensor_dump_max_steps: int,
    tensor_dump_max_calls_per_module: int,
) -> None:
    if tensor_dump_dir is None or tensor_dump_max_steps == 0:
        return

    Path(tensor_dump_dir).mkdir(parents=True, exist_ok=True)
    print(
        "[tensor-dump] Installing hooks. "
        f"dir={tensor_dump_dir} "
        f"max_calls_per_module={tensor_dump_max_calls_per_module}",
        flush=True,
    )
    if policy_generation is not policy:
        _call_debug_method(
            policy_generation,
            label="generation",
            method_name="install_debug_tensor_hooks",
            max_calls_per_module=tensor_dump_max_calls_per_module,
        )
    _call_debug_method(
        policy,
        label="policy",
        method_name="install_debug_tensor_hooks",
        max_calls_per_module=tensor_dump_max_calls_per_module,
    )


def _save_tensor_dump(
    *,
    target: Any,
    label: str,
    tensor_dump_dir: str,
    tensor_dump_prefix: str,
    step: int,
) -> Any:
    return _call_debug_method(
        target,
        label=label,
        method_name="save_debug_tensor_capture",
        output_dir=tensor_dump_dir,
        prefix=f"{tensor_dump_prefix}_{label}",
        step=step,
    )


def _save_comparison_tensors(
    *,
    tensor_dump_dir: str,
    tensor_dump_prefix: str,
    step: int,
    logprob_data: BatchedDataDict[ClippedPGLossDataDict],
    generation_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    input_ids: torch.Tensor,
    comparison_metrics: dict[str, Any],
) -> None:
    path = Path(tensor_dump_dir) / f"{tensor_dump_prefix}_comparison_step{step:06d}.pt"
    payload = {
        "logprob_input_ids": logprob_data["input_ids"].detach().cpu(),
        "logprob_input_lengths": logprob_data["input_lengths"].detach().cpu(),
        "sample_mask": logprob_data["sample_mask"].detach().cpu(),
        "generation_logprobs": generation_logprobs.detach().cpu(),
        "policy_logprobs": policy_logprobs.detach().cpu(),
        "token_mask": token_mask.detach().cpu(),
        "input_ids": input_ids.detach().cpu(),
        "comparison_metrics": comparison_metrics,
    }
    torch.save(payload, path)
    print(f"[tensor-dump] comparison tensors saved: {path}", flush=True)


def _save_first_batch_token_dump(
    *,
    path: str,
    step: int,
    logprob_data: BatchedDataDict[ClippedPGLossDataDict],
    generation_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: TokenizerType,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_ids_cpu = input_ids.detach().cpu()
    input_lengths_cpu = logprob_data["input_lengths"].detach().cpu().to(torch.long)
    token_mask_cpu = token_mask.detach().cpu().bool()
    sample_mask_cpu = logprob_data["sample_mask"].detach().cpu()
    generation_logprobs_cpu = generation_logprobs.detach().cpu().to(torch.float32)

    sample_rows: list[dict[str, Any]] = []
    prompt_token_ids_list: list[list[int]] = []
    response_token_ids_list: list[list[int]] = []
    full_token_ids_list: list[list[int]] = []
    offline_prefix_token_ids_list: list[list[int]] = []
    offline_target_token_ids: list[int] = []
    offline_generation_logprobs: list[float] = []
    offline_metadata: list[dict[str, int]] = []

    for row in range(input_ids_cpu.size(0)):
        input_length = int(input_lengths_cpu[row].item())
        full_token_ids = [
            int(token_id) for token_id in input_ids_cpu[row, :input_length].tolist()
        ]
        response_positions = [
            int(pos)
            for pos in token_mask_cpu[row, :input_length]
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        ]
        prompt_end = response_positions[0] if response_positions else input_length
        prompt_token_ids = full_token_ids[:prompt_end]
        response_token_ids = [full_token_ids[pos] for pos in response_positions]

        prompt_token_ids_list.append(prompt_token_ids)
        response_token_ids_list.append(response_token_ids)
        full_token_ids_list.append(full_token_ids)
        sample_rows.append(
            {
                "sample_idx": row,
                "input_length": input_length,
                "prompt_length": prompt_end,
                "num_response_tokens": len(response_token_ids),
                "response_positions": response_positions,
                "sample_mask": float(sample_mask_cpu[row].item()),
            }
        )

        for response_index, token_position in enumerate(response_positions):
            if token_position == 0:
                continue
            target_token_id = full_token_ids[token_position]
            offline_prefix_token_ids_list.append(full_token_ids[:token_position])
            offline_target_token_ids.append(target_token_id)
            if token_position < generation_logprobs_cpu.size(1):
                offline_generation_logprobs.append(
                    float(generation_logprobs_cpu[row, token_position].item())
                )
            else:
                offline_generation_logprobs.append(float("nan"))
            offline_metadata.append(
                {
                    "offline_idx": len(offline_metadata),
                    "sample_idx": row,
                    "response_index": response_index,
                    "position": token_position,
                    "target_token_id": target_token_id,
                    "prefix_length": token_position,
                }
            )

    payload = {
        "step": step,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "input_ids": input_ids_cpu,
        "input_lengths": input_lengths_cpu,
        "token_mask": token_mask_cpu,
        "sample_mask": sample_mask_cpu,
        "generation_logprobs": generation_logprobs_cpu,
        "sample_rows": sample_rows,
        "sample_prompt_token_ids_list": prompt_token_ids_list,
        "sample_response_token_ids_list": response_token_ids_list,
        "sample_full_token_ids_list": full_token_ids_list,
        "offline_prefix_token_ids_list": offline_prefix_token_ids_list,
        "offline_target_token_ids": offline_target_token_ids,
        "offline_generation_logprobs": offline_generation_logprobs,
        "offline_metadata": offline_metadata,
        # Default key consumed by the offline forward scripts.
        "token_ids_list": offline_prefix_token_ids_list,
        "prompts": [
            (
                f"sample={item['sample_idx']} pos={item['position']} "
                f"target={item['target_token_id']}"
            )
            for item in offline_metadata
        ],
    }
    torch.save(payload, output_path)
    print(
        "[token-dump] first-batch token ids saved: "
        f"{output_path} samples={len(sample_rows)} "
        f"offline_prefixes={len(offline_prefix_token_ids_list)}",
        flush=True,
    )


def _build_vllm_prefill_check_data(
    logprob_data: BatchedDataDict[ClippedPGLossDataDict],
) -> BatchedDataDict[dict[str, torch.Tensor]]:
    return BatchedDataDict(
        {
            "input_ids": logprob_data["input_ids"],
            "input_lengths": logprob_data["input_lengths"],
        }
    )


def _run_vllm_prefill_logprob_check(
    *,
    policy_generation: GenerationInterface,
    logprob_data: BatchedDataDict[ClippedPGLossDataDict],
    generation_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    score_prompt_logprobs = getattr(policy_generation, "score_prompt_logprobs", None)
    if score_prompt_logprobs is None:
        raise RuntimeError(
            "policy_generation does not expose score_prompt_logprobs; "
            "the vLLM prefill check requires sync VllmGeneration."
        )

    prefill_logprobs = score_prompt_logprobs(
        _build_vllm_prefill_check_data(logprob_data)
    )["logprobs"]
    return _compare_logprobs(
        generation_logprobs=generation_logprobs,
        policy_logprobs=prefill_logprobs,
        token_mask=token_mask,
        sample_mask=logprob_data["sample_mask"],
        input_ids=input_ids,
        top_k=top_k,
    )


def _run_rollout(
    *,
    policy_generation: GenerationInterface,
    repeated_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: MasterConfig,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    if _should_use_nemo_gym(master_config):
        generation_config = master_config.policy["generation"]
        nemo_gym_rollout_result = run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=repeated_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=master_config.policy["max_total_sequence_length"],
            generation_config=generation_config,
            max_rollout_turns=None,
            greedy=False,
        )
        return (
            nemo_gym_rollout_result.final_batch,
            nemo_gym_rollout_result.rollout_metrics,
        )

    if _should_use_async_rollouts(master_config):
        return run_async_multi_turn_rollout(
            policy_generation=policy_generation,
            input_batch=repeated_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=master_config.policy["max_total_sequence_length"],
            max_rollout_turns=master_config.grpo["max_rollout_turns"],
            greedy=False,
        )

    return run_multi_turn_rollout(
        policy_generation=policy_generation,
        input_batch=repeated_batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=master_config.policy["max_total_sequence_length"],
        max_rollout_turns=master_config.grpo["max_rollout_turns"],
        greedy=False,
    )


def _enable_vllm_eager_for_tensor_dumps(
    config: MasterConfig, tensor_dump_dir: Optional[str]
) -> None:
    if tensor_dump_dir is None:
        return

    generation_config = config.policy.get("generation")
    if generation_config is None:
        return

    vllm_cfg = generation_config.get("vllm_cfg")
    if not isinstance(vllm_cfg, dict):
        return

    if not vllm_cfg.get("enforce_eager", False):
        vllm_cfg["enforce_eager"] = True
        print(
            "[tensor-dump] Enabled policy.generation.vllm_cfg.enforce_eager=true "
            "for vLLM tensor hooks.",
            flush=True,
        )


def run_logprob_comparison(
    *,
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    wrapped_dataloader: StatefulDataLoader | MultipleDataloaderWrapper,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    logger: Logger,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    top_k: int,
    tensor_dump_dir: Optional[str],
    tensor_dump_prefix: str,
    tensor_dump_max_steps: int,
    tensor_dump_max_calls_per_module: int,
    vllm_prefill_check_first_batch: bool,
    first_batch_token_dump: Optional[str],
    inspect_vllm_layernorm_impl: bool,
) -> dict[str, float | int]:
    timer = Timer()
    need_refit = True
    if policy_generation is None:
        policy_generation = policy  # type: ignore[assignment]
        need_refit = False

    assert policy_generation is not None
    policy_generation_stale = True
    kv_scales_cache = None
    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]

    current_step = grpo_save_state["current_step"]
    total_steps = grpo_save_state["total_steps"]
    current_epoch = grpo_save_state["current_epoch"]
    max_num_steps = master_config.grpo["max_num_steps"]
    max_num_epochs = master_config.grpo["max_num_epochs"]
    totals = LogprobComparisonTotals()
    tensor_dump_steps_done = 0
    vllm_prefill_check_done = False
    first_batch_token_dump_done = False

    _install_tensor_dump_hooks(
        policy=policy,
        policy_generation=policy_generation,
        tensor_dump_dir=tensor_dump_dir,
        tensor_dump_max_steps=tensor_dump_max_steps,
        tensor_dump_max_calls_per_module=tensor_dump_max_calls_per_module,
    )

    if inspect_vllm_layernorm_impl and policy_generation is not policy:
        inspect_method = getattr(policy_generation, "inspect_layernorm_impl", None)
        if inspect_method is None:
            print(
                "[vllm-layernorm-impl] generation object does not expose "
                "inspect_layernorm_impl",
                flush=True,
            )
        else:
            info = inspect_method()
            print(
                "[vllm-layernorm-impl] nemo_rl_generation\n"
                f"{pprint.pformat(info, sort_dicts=True)}",
                flush=True,
            )

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        for batch in wrapped_dataloader:
            if total_steps >= max_num_steps:
                break

            step = total_steps + 1
            dump_current_step = _should_dump_tensor_step(
                tensor_dump_dir,
                tensor_dump_max_steps,
                tensor_dump_steps_done,
            )
            print(f"\n{'=' * 25} Logprob comparison step {step} {'=' * 25}")

            repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                master_config.grpo["num_generations_per_prompt"]
            )

            print("▶ Preparing generation workers...", flush=True)
            with timer.time("prepare_for_generation/total"):
                if need_refit and policy_generation_stale:
                    if sync_kv_scales and kv_scales_cache is None:
                        print("▶ Computing KV cache scales...", flush=True)
                        policy.prepare_for_lp_inference()
                        calibration_data = _build_calibration_data(
                            repeated_batch, tokenizer, master_config
                        )
                        kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                            calibration_data, include_q=True
                        )["layers"]

                    refit_policy_generation(
                        policy,
                        policy_generation,
                        colocated_inference,
                        timer=timer,
                        kv_scales=kv_scales_cache if sync_kv_scales else None,
                    )
                    policy_generation_stale = False
                else:
                    if colocated_inference and need_refit:
                        policy.offload_after_refit()
                    policy_generation.prepare_for_generation()

            print(
                f"▶ Running rollout for batch of size {repeated_batch.size}...",
                flush=True,
            )
            with timer.time("generation"):
                if dump_current_step and policy_generation is not policy:
                    _call_debug_method(
                        policy_generation,
                        label="generation",
                        method_name="clear_debug_tensor_capture",
                    )
                with timer.time("rollout_generation"):
                    repeated_batch, rollout_metrics = _run_rollout(
                        policy_generation=policy_generation,
                        repeated_batch=repeated_batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        master_config=master_config,
                    )
                if vllm_prefill_check_first_batch and not vllm_prefill_check_done:
                    print(
                        "▶ Running vLLM whole-prefill logprob check...",
                        flush=True,
                    )
                    with timer.time("vllm_prefill_check"):
                        (
                            vllm_check_logprob_data,
                            vllm_check_generation_logprobs,
                            vllm_check_token_mask,
                            vllm_check_input_ids,
                        ) = _build_logprob_data(
                            repeated_batch, tokenizer, master_config
                        )
                        vllm_prefill_metrics = _run_vllm_prefill_logprob_check(
                            policy_generation=policy_generation,
                            logprob_data=vllm_check_logprob_data,
                            generation_logprobs=vllm_check_generation_logprobs,
                            token_mask=vllm_check_token_mask,
                            input_ids=vllm_check_input_ids,
                            top_k=top_k,
                        )
                    _print_vllm_prefill_check_summary(
                        step=step,
                        comparison_metrics=vllm_prefill_metrics,
                    )
                    logger.log_metrics(
                        _logger_metrics(vllm_prefill_metrics, {}),
                        step,
                        prefix="vllm_prefill_check",
                    )
                    vllm_prefill_check_done = True
                if dump_current_step and policy_generation is not policy:
                    assert tensor_dump_dir is not None
                    _save_tensor_dump(
                        target=policy_generation,
                        label="generation",
                        tensor_dump_dir=tensor_dump_dir,
                        tensor_dump_prefix=tensor_dump_prefix,
                        step=step,
                    )
                policy_generation.finish_generation(discard_weights=colocated_inference)
                if colocated_inference:
                    policy_generation_stale = True

            print("▶ Building policy logprob input...", flush=True)
            with timer.time("logprob_data_processing"):
                (
                    logprob_data,
                    generation_logprobs,
                    token_mask,
                    input_ids,
                ) = _build_logprob_data(repeated_batch, tokenizer, master_config)
            if first_batch_token_dump and not first_batch_token_dump_done:
                _save_first_batch_token_dump(
                    path=first_batch_token_dump,
                    step=step,
                    logprob_data=logprob_data,
                    generation_logprobs=generation_logprobs,
                    token_mask=token_mask,
                    input_ids=input_ids,
                    tokenizer=tokenizer,
                )
                first_batch_token_dump_done = True

            print("▶ Computing policy logprobs...", flush=True)
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            with timer.time("policy_logprobs"):
                if dump_current_step:
                    _call_debug_method(
                        policy,
                        label="policy",
                        method_name="clear_debug_tensor_capture",
                    )
                policy_logprobs = policy.get_logprobs(logprob_data, timer=timer)[
                    "logprobs"
                ]
                if dump_current_step:
                    assert tensor_dump_dir is not None
                    _save_tensor_dump(
                        target=policy,
                        label="policy",
                        tensor_dump_dir=tensor_dump_dir,
                        tensor_dump_prefix=tensor_dump_prefix,
                        step=step,
                    )

            comparison_metrics = _compare_logprobs(
                generation_logprobs=generation_logprobs,
                policy_logprobs=policy_logprobs,
                token_mask=token_mask,
                sample_mask=logprob_data["sample_mask"],
                input_ids=input_ids,
                top_k=top_k,
            )
            totals.update(comparison_metrics)
            logger.log_metrics(
                _logger_metrics(comparison_metrics, rollout_metrics),
                step,
                prefix="logprob_comparison",
            )
            _print_step_summary(
                step=step,
                max_num_steps=max_num_steps,
                comparison_metrics=comparison_metrics,
                rollout_metrics=rollout_metrics,
            )
            _print_timing_summary(
                step=step,
                timer=timer,
                rollout_metrics=rollout_metrics,
                batch_size=repeated_batch.size,
                master_config=master_config,
            )
            if dump_current_step:
                assert tensor_dump_dir is not None
                _save_comparison_tensors(
                    tensor_dump_dir=tensor_dump_dir,
                    tensor_dump_prefix=tensor_dump_prefix,
                    step=step,
                    logprob_data=logprob_data,
                    generation_logprobs=generation_logprobs,
                    policy_logprobs=policy_logprobs,
                    token_mask=token_mask,
                    input_ids=input_ids,
                    comparison_metrics=comparison_metrics,
                )
                tensor_dump_steps_done += 1

            current_step += 1
            total_steps += 1

        current_epoch += 1
        current_step = 0

    summary = totals.as_metrics()
    print("\n" + "=" * 60)
    print("Logprob comparison complete")
    print(
        f"Compared tokens: {summary['num_compared_tokens']}\n"
        f"Mean abs diff: {summary['mean_abs_diff']:.6e}\n"
        f"Max abs diff: {summary['max_abs_diff']:.6e}\n"
        f"Mean rel diff: {summary['mean_rel_diff']:.6e}\n"
        f"Max rel diff: {summary['max_rel_diff']:.6e}\n"
        f"Mean signed diff: {summary['mean_signed_diff']:.6e}",
        flush=True,
    )
    print("=" * 60 + "\n", flush=True)
    logger.log_metrics(summary, total_steps, prefix="logprob_comparison/summary")
    return summary


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if args.config is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.config = str(repo_root / "examples" / "configs" / "grpo_math_1B.yaml")

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    _enable_vllm_eager_for_tensor_dumps(config, args.tensor_dump_dir)
    print("Applied CLI overrides")
    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(f"Using checkpoint directory: {config.checkpointing['checkpoint_dir']}")

    init_ray()

    policy = None
    policy_generation = None
    try:
        tokenizer = get_tokenizer(config.policy["tokenizer"])
        assert config.policy["generation"] is not None, (
            "A generation config is required for GRPO logprob comparison"
        )
        has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
        config.policy["generation"] = configure_generation_config(
            config.policy["generation"],
            tokenizer,
            has_refit_draft_weights=has_refit_draft_weights,
        )

        dataset, val_dataset, task_to_env, _val_task_to_env = setup_response_data(
            tokenizer, config.data, config.env
        )
        (
            policy,
            policy_generation,
            _cluster,
            dataloader,
            _val_dataloader,
            _loss_fn,
            logger,
            _checkpointer,
            grpo_state,
            master_config,
        ) = setup(config, tokenizer, dataset, val_dataset)

        run_logprob_comparison(
            policy=policy,
            policy_generation=policy_generation,
            wrapped_dataloader=dataloader,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            logger=logger,
            grpo_save_state=grpo_state,
            master_config=master_config,
            top_k=args.top_k,
            tensor_dump_dir=args.tensor_dump_dir,
            tensor_dump_prefix=args.tensor_dump_prefix,
            tensor_dump_max_steps=args.tensor_dump_max_steps,
            tensor_dump_max_calls_per_module=args.tensor_dump_max_calls_per_module,
            vllm_prefill_check_first_batch=args.vllm_prefill_check_first_batch,
            first_batch_token_dump=args.first_batch_token_dump,
            inspect_vllm_layernorm_impl=args.inspect_vllm_layernorm_impl,
        )
    finally:
        if policy_generation is not None and policy_generation is not policy:
            policy_generation.shutdown()
        if policy is not None:
            policy.shutdown()
        if ray.is_initialized():
            ray.shutdown()
        os.environ.pop("RAY_ADDRESS", None)


if __name__ == "__main__":
    main()
