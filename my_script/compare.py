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

"""Compare vLLM vs Megatron last-token logits captured by ``vllm_forward.py``.

Captures are produced by ``vllm_forward.py`` and ``megatron_forward.py`` over a
batch of real prompts.

Prints per-prompt summary stats and writes a scatter plot of every logit
value (megatron on the x-axis, vllm on the y-axis) to ``--plot``.

Run with either env:
    uv run --extra vllm  python my_script/compare.py
    uv run --extra mcore python my_script/compare.py
"""

import argparse
import os

import torch

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NEMOTRON3_NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_ALIASES = {
    "llama": DEFAULT_MODEL,
    "llama3.1-8b": DEFAULT_MODEL,
    "llama-3.1-8b": DEFAULT_MODEL,
    "nemotron3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron-3-nano": NEMOTRON3_NANO_MODEL,
    "nemotron3-nano-30b-a3b": NEMOTRON3_NANO_MODEL,
}


def resolve_model_ref(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def is_nemotron3_nano_ref(value: str | None) -> bool:
    if value is None:
        return False
    value = resolve_model_ref(value)
    normalised = value.lower().replace("_", "-")
    return (
        value == NEMOTRON3_NANO_MODEL or "nvidia-nemotron-3-nano-30b-a3b" in normalised
    )


def model_output_tag(model: str | None) -> str:
    if is_nemotron3_nano_ref(model):
        return "_nemotron3_nano"
    return ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model alias/id used to choose default capture paths. "
        "Use 'nemotron3-nano' for "
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.",
    )
    p.add_argument("--vllm", default=None)
    p.add_argument(
        "--megatron",
        default=None,
    )
    p.add_argument(
        "--batch-invariant",
        action="store_true",
        help="Use *_bi.pt files for both sides",
    )
    p.add_argument(
        "--mxfp8", action="store_true", help="Use *_mxfp8*.pt files for both sides"
    )
    p.add_argument(
        "--plot",
        default=None,
        help="Output path for the scatter PNG "
        "(default: my_script/compare_logits_scatter.png).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Subsample logit pairs to at most this many points "
        "in the scatter plot (default: 200000). Set to 0 for all.",
    )
    p.add_argument(
        "--compare-modules",
        action="store_true",
        help="Print debug tensor diffs when the capture payloads "
        "contain module_inputs_by_layer/module_outputs_by_layer.",
    )
    p.add_argument(
        "--log-softmax-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used to normalize Megatron logits before "
        "comparing against vLLM logprobs.",
    )
    p.add_argument(
        "--target-token-ids-file",
        default=None,
        help=(
            "Optional torch .pt token dump containing the target token id for "
            "each compared prompt prefix."
        ),
    )
    p.add_argument(
        "--target-token-ids-key",
        default="offline_target_token_ids",
        help=(
            "Payload key to read from --target-token-ids-file "
            "(default: offline_target_token_ids)."
        ),
    )
    args = p.parse_args()
    args.model = resolve_model_ref(args.model)
    suffix = model_output_tag(args.model)
    if args.mxfp8:
        suffix += "_mxfp8"
    if args.batch_invariant:
        suffix += "_bi"
    if args.vllm is None:
        args.vllm = os.path.join(DEFAULT_DIR, f"vllm_capture{suffix}.pt")
    if args.megatron is None:
        args.megatron = os.path.join(DEFAULT_DIR, f"megatron_capture{suffix}.pt")
    if args.plot is None:
        args.plot = os.path.join(DEFAULT_DIR, f"compare_logits_scatter{suffix}.png")
    return args


def load_target_token_ids(path, key):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        if key not in payload:
            raise KeyError(f"{path} does not contain target token key {key!r}")
        value = payload[key]
        metadata = payload.get("offline_metadata", [])
    else:
        value = payload
        metadata = []

    if isinstance(value, torch.Tensor):
        target_token_ids = [int(token_id) for token_id in value.flatten().tolist()]
    else:
        target_token_ids = [int(token_id) for token_id in value]
    return target_token_ids, metadata


def diff_stats(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    diff = (a - b).abs()
    return {
        "n": n,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "a_norm": float(a.norm()),
        "b_norm": float(b.norm()),
        "cos_sim": float(
            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        ),
    }


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for key in ("hidden_states", "x", "input", "residual"):
            if key in value:
                found = first_tensor(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def get_by_selector(entry, selector):
    if "+" in selector:
        terms = []
        for part in selector.split("+"):
            term = get_by_selector(entry, part)
            if not isinstance(term, torch.Tensor):
                return None
            terms.append(term)
        result = terms[0]
        for term in terms[1:]:
            if term.shape != result.shape:
                return None
            result = result + term
        return result
    if selector == "first":
        return first_tensor(entry)
    if isinstance(entry, dict) and "args" in entry and selector.startswith("arg"):
        idx = int(selector[3:])
        args = entry.get("args", ())
        if idx < len(args):
            return first_tensor(args[idx])
        return None
    if isinstance(entry, dict) and "kwargs" in entry and selector.startswith("kw."):
        key = selector[3:]
        return first_tensor(entry.get("kwargs", {}).get(key))
    if isinstance(entry, (list, tuple)) and selector.startswith("item"):
        idx = int(selector[4:])
        if idx < len(entry):
            return first_tensor(entry[idx])
        return None
    if selector == "output":
        return first_tensor(entry)
    return None


def select_tensor(entry, selectors):
    if isinstance(entry, dict) and "__calls__" in entry:
        calls = entry["__calls__"]
        for selector in selectors:
            tensors = []
            for call in calls:
                tensor = get_by_selector(call, selector)
                if not isinstance(tensor, torch.Tensor):
                    tensors = []
                    break
                tensors.append(tensor)
            tensor = concatenate_call_tensors(tensors)
            if tensor is not None:
                return tensor, f"{selector}x{len(tensors)}"
        return None, None

    for selector in selectors:
        tensor = get_by_selector(entry, selector)
        if isinstance(tensor, torch.Tensor):
            return tensor, selector
    return None, None


def concatenate_call_tensors(tensors):
    if not tensors:
        return None

    first = tensors[0]
    if first.dim() == 0:
        return torch.stack(tensors, dim=0)
    for tensor in tensors:
        if tensor.dim() != first.dim() or tensor.shape[1:] != first.shape[1:]:
            return None
    return torch.cat(tensors, dim=0)


def normalize_token_layout(tensor, seq_lens):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dim() < 2 or not seq_lens:
        return tensor

    total_tokens = sum(seq_lens)
    batch = len(seq_lens)
    max_seq = max(seq_lens)

    if tensor.shape[0] == total_tokens:
        return tensor
    if tensor.shape[0] == batch and tensor.shape[1] >= max_seq:
        return torch.cat(
            [tensor[i, :seq_len] for i, seq_len in enumerate(seq_lens)],
            dim=0,
        )
    if tensor.shape[1] == batch and tensor.shape[0] >= max_seq:
        return torch.cat(
            [tensor[:seq_len, i] for i, seq_len in enumerate(seq_lens)],
            dim=0,
        )
    return tensor


def normalize_router_layout(tensor, seq_lens):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dim() < 2 or not seq_lens:
        return tensor

    batch = len(seq_lens)
    max_seq = max(seq_lens)
    if batch == 1 and tensor.dim() >= 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    if tensor.shape[0] == batch * max_seq:
        tensor = tensor.reshape(max_seq, batch, *tensor.shape[1:])
    tensor = normalize_token_layout(tensor, seq_lens)
    if batch == 1 and tensor.dim() >= 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor


def squeeze_singleton_router_batch(tensor, seq_lens):
    if (
        isinstance(tensor, torch.Tensor)
        and len(seq_lens) == 1
        and tensor.dim() >= 3
        and tensor.shape[1] == 1
    ):
        return tensor.squeeze(1)
    return tensor


def padded_router_to_actual_indices(seq_lens):
    if not seq_lens:
        return None
    batch = len(seq_lens)
    max_seq = max(seq_lens)
    mapping = torch.full((batch * max_seq,), -1, dtype=torch.long)
    actual_idx = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            mapping[pos * batch + batch_idx] = actual_idx
            actual_idx += 1
    return mapping


def compare_selected_tensor(
    label,
    v_entry,
    m_entry,
    v_selectors,
    m_selectors,
    seq_lens,
):
    v_t, v_selector = select_tensor(v_entry, v_selectors)
    m_t, m_selector = select_tensor(m_entry, m_selectors)
    if v_t is None or m_t is None:
        print(f"  {label:<68s} {'MISSING':>11s}")
        return

    v_t = normalize_token_layout(v_t, seq_lens)
    m_t = normalize_token_layout(m_t, seq_lens)
    stats = diff_stats(v_t, m_t)
    print(
        f"  {label:<68s} {stats['max_abs_diff']:>11.6f} "
        f"{stats['mean_abs_diff']:>11.4e} {stats['cos_sim']:>10.6f} "
        f"{str(tuple(v_t.shape)):>22s} {str(tuple(m_t.shape)):>22s} "
        f"{v_selector:>10s} {m_selector:>10s}"
    )


def group_limited_topk(scores, topk, num_groups, group_topk):
    num_tokens, num_experts = scores.shape
    if num_groups is None or group_topk is None:
        return torch.topk(scores, k=topk, dim=1)
    if num_groups <= 0 or group_topk <= 0:
        raise ValueError("num_groups and group_topk must be positive")
    if num_experts % num_groups != 0:
        raise ValueError(
            f"num_experts={num_experts} is not divisible by num_groups={num_groups}"
        )
    group_score_topk = topk // group_topk
    if group_score_topk <= 0:
        raise ValueError(f"topk={topk} is too small for group_topk={group_topk}")

    group_scores = (
        scores.view(num_tokens, num_groups, -1)
        .topk(group_score_topk, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
        .bool()
    )
    masked_scores = scores.masked_fill(~score_mask, float("-inf"))
    return torch.topk(masked_scores, k=topk, dim=1)


def _prepare_expert_bias(expert_bias, scores):
    if not isinstance(expert_bias, torch.Tensor):
        return None
    expert_bias = expert_bias.float().reshape(1, -1)
    if expert_bias.shape[1] != scores.shape[1]:
        return None
    return expert_bias.to(device=scores.device)


def router_candidate_specs(num_experts, topk):
    yield ("sigmoid_raw", "sigmoid_raw", None, None)
    yield ("sigmoid_norm", "sigmoid", None, None)

    divisors = [
        value
        for value in (2, 4, 8, 16, 32, 64, 128)
        if value <= num_experts and num_experts % value == 0
    ]
    group_topks = [value for value in (1, 2, 3, 6) if value <= topk]
    for num_groups in divisors:
        experts_per_group = num_experts // num_groups
        for group_topk in group_topks:
            if group_topk > num_groups:
                continue
            group_score_topk = topk // group_topk
            if group_score_topk <= 0 or group_score_topk > experts_per_group:
                continue
            label = f"sigm_g{num_groups}k{group_topk}"
            yield (label, "sigmoid", num_groups, group_topk)

    yield ("softmax_topk", "softmax_topk", None, None)
    yield ("softmax_pre", "softmax_pre", None, None)


def sparse_routing_from_logits(
    logits,
    topk,
    scaling_factor,
    score_function,
    num_groups=None,
    group_topk=None,
    expert_bias=None,
):
    logits = logits.float()
    if score_function in ("sigmoid", "sigmoid_raw"):
        scores = torch.sigmoid(logits)
        expert_bias = _prepare_expert_bias(expert_bias, scores)
        scores_for_routing = scores + expert_bias if expert_bias is not None else scores
        _, top_indices = group_limited_topk(
            scores_for_routing, topk, num_groups, group_topk
        )
        top_scores = torch.gather(scores, dim=1, index=top_indices)
        if score_function == "sigmoid_raw":
            probs = top_scores
        elif topk > 1:
            probs = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            probs = top_scores
    elif score_function == "softmax_topk":
        top_scores, top_indices = group_limited_topk(
            logits, topk, num_groups, group_topk
        )
        probs = torch.softmax(top_scores, dim=-1)
    elif score_function == "softmax_pre":
        scores = torch.softmax(logits, dim=-1)
        probs, top_indices = group_limited_topk(scores, topk, num_groups, group_topk)
    else:
        raise ValueError(f"unknown score function candidate: {score_function}")

    if scaling_factor is not None:
        probs = probs * scaling_factor

    sparse_probs = torch.zeros_like(logits, dtype=probs.dtype)
    sparse_probs.scatter_(1, top_indices, probs)
    routing_map = torch.zeros_like(logits, dtype=torch.bool)
    routing_map.scatter_(1, top_indices, True)
    return sparse_probs, routing_map


def sparse_routing_from_topk(topk_weights, topk_ids, num_experts):
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.long()
    sparse_probs = torch.zeros(
        topk_weights.shape[0],
        num_experts,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    routing_map = torch.zeros_like(sparse_probs, dtype=torch.bool)
    valid = (topk_ids >= 0) & (topk_ids < num_experts)
    safe_ids = topk_ids.masked_fill(~valid, 0)
    sparse_probs.scatter_(1, safe_ids, topk_weights.masked_fill(~valid, 0.0))
    routing_map.scatter_(1, safe_ids, valid)
    return sparse_probs, routing_map


def router_candidate_results(
    v_logits,
    m_probs,
    m_map,
    topk,
    scaling_factor,
    expert_bias=None,
):
    results = []
    for label, score_function, num_groups, group_topk in router_candidate_specs(
        v_logits.shape[1], topk
    ):
        v_probs, v_map = sparse_routing_from_logits(
            v_logits,
            topk=topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            num_groups=num_groups,
            group_topk=group_topk,
            expert_bias=expert_bias,
        )
        row_matches = (v_map == m_map).all(dim=1)
        entry_matches = v_map == m_map
        prob_stats = diff_stats(v_probs, m_probs)
        results.append(
            {
                "label": label,
                "probs": v_probs,
                "map": v_map,
                "row_match": float(row_matches.float().mean()),
                "entry_match": float(entry_matches.float().mean()),
                "prob_stats": prob_stats,
            }
        )
    return results


def best_router_candidate(
    v_logits,
    m_probs,
    m_map,
    topk,
    scaling_factor,
    expert_bias=None,
):
    results = router_candidate_results(
        v_logits,
        m_probs,
        m_map,
        topk,
        scaling_factor,
        expert_bias=expert_bias,
    )
    return max(
        results,
        key=lambda item: (
            item["row_match"],
            item["entry_match"],
            -item["prob_stats"]["mean_abs_diff"],
        ),
    )


def get_router_config(payload, layer_idx, module_name="mlp.router.routing"):
    configs = payload.get("router_config_by_layer", {})
    return layer_map(configs, layer_idx).get(module_name, {})


def get_router_expert_bias(payload, layer_idx):
    config = get_router_config(payload, layer_idx)
    if not isinstance(config, dict):
        return None
    return config.get("expert_bias")


def format_router_config(config):
    if not isinstance(config, dict) or not config:
        return ""
    keys = (
        "topk",
        "score_function",
        "routing_type",
        "moe_router_pre_softmax",
        "moe_router_num_groups",
        "moe_router_group_topk",
        "moe_router_topk_scaling_factor",
        "moe_router_fusion",
        "enable_expert_bias",
        "moe_expert_capacity_factor",
        "moe_router_force_load_balancing",
        "moe_router_force_biased",
    )
    parts = []
    for key in keys:
        value = config.get(key)
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = f"tensor{tuple(value.shape)}"
        parts.append(f"{key}={value}")
    return " ".join(parts)


def print_router_internal_report(m, seq_lens, layer_indices):
    print("\n" + "=" * 110)
    print("Megatron router internal capture")
    print("=" * 110)
    print(
        f"  {'layer':>5s} {'check':<24s} {'map_rows':>10s} "
        f"{'map_entries':>11s} {'prob_max':>11s} {'prob_mean':>11s} "
        f"{'prob_cos':>10s}"
    )

    for layer_idx in layer_indices:
        m_gating = get_layer_entry(m, layer_idx, "mlp.router.gating", outputs=True)
        m_router = get_layer_entry(m, layer_idx, "mlp.router", outputs=True)
        m_routing_in = get_layer_entry(m, layer_idx, "mlp.router.routing")
        m_routing_out = get_layer_entry(
            m,
            layer_idx,
            "mlp.router.routing",
            outputs=True,
        )
        routing_logits, _ = select_tensor(m_routing_in, ("arg0", "first"))
        routing_probs, _ = select_tensor(m_routing_out, ("item0", "output", "first"))
        routing_map, _ = select_tensor(m_routing_out, ("item1",))
        router_probs, _ = select_tensor(m_router, ("item0", "output", "first"))
        router_map, _ = select_tensor(m_router, ("item1",))
        if routing_logits is None or routing_probs is None or routing_map is None:
            print(f"  {layer_idx:>5d} {'MISSING routing':<24s}")
            continue

        routing_logits = normalize_router_layout(routing_logits, seq_lens)
        routing_probs = normalize_router_layout(routing_probs, seq_lens).float()
        routing_map = normalize_router_layout(routing_map, seq_lens).bool()
        if router_probs is not None:
            router_probs = normalize_router_layout(router_probs, seq_lens).float()
        if router_map is not None:
            router_map = normalize_router_layout(router_map, seq_lens).bool()

        config = get_router_config(m, layer_idx)
        config_text = format_router_config(config)
        if config_text:
            print(f"        config layer={layer_idx}: {config_text}")

        gating_logits, _ = select_tensor(m_gating, ("output", "item0", "first"))
        if gating_logits is not None:
            gating_logits = normalize_router_layout(gating_logits, seq_lens)
            if gating_logits.shape == routing_logits.shape:
                stats = diff_stats(gating_logits, routing_logits)
                print(
                    f"        gating_vs_routing_logits layer={layer_idx}: "
                    f"max={stats['max_abs_diff']:.6g} "
                    f"mean={stats['mean_abs_diff']:.6g} "
                    f"cos={stats['cos_sim']:.6f}"
                )

        if (
            router_probs is not None
            and router_map is not None
            and router_probs.shape == routing_probs.shape
            and router_map.shape == routing_map.shape
        ):
            row_match = (router_map == routing_map).all(dim=1).float().mean()
            entry_match = (router_map == routing_map).float().mean()
            stats = diff_stats(router_probs, routing_probs)
            print(
                f"  {layer_idx:>5d} {'routing_vs_router':<24s} "
                f"{float(row_match):>10.6f} {float(entry_match):>11.6f} "
                f"{stats['max_abs_diff']:>11.6f} "
                f"{stats['mean_abs_diff']:>11.4e} {stats['cos_sim']:>10.6f}"
            )

        valid_rows = routing_map.any(dim=1)
        if not bool(valid_rows.any()):
            continue
        topk = int(torch.median(routing_map[valid_rows].sum(dim=1).float()).item())
        scale = float(torch.median(routing_probs[valid_rows].sum(dim=1)).item())
        candidate = best_router_candidate(
            routing_logits,
            routing_probs,
            routing_map,
            topk,
            scale,
            expert_bias=get_router_expert_bias(m, layer_idx),
        )
        stats = candidate["prob_stats"]
        print(
            f"  {layer_idx:>5d} {candidate['label']:<24s} "
            f"{candidate['row_match']:>10.6f} "
            f"{candidate['entry_match']:>11.6f} "
            f"{stats['max_abs_diff']:>11.6f} "
            f"{stats['mean_abs_diff']:>11.4e} {stats['cos_sim']:>10.6f}"
        )


def print_router_decision_report(v, m, seq_lens, layer_indices):
    print("\n" + "=" * 110)
    print("NemotronH router top-k decision diffs")
    print("=" * 110)
    print(
        f"  {'layer':>5s} {'candidate':<16s} {'topk':>5s} {'scale':>9s} "
        f"{'map_rows':>10s} {'map_entries':>11s} {'prob_max':>11s} "
        f"{'prob_mean':>11s} {'prob_cos':>10s}"
    )

    for layer_idx in layer_indices:
        v_gate = get_layer_entry(v, layer_idx, "mixer.gate", outputs=True)
        v_select = get_layer_entry(
            v,
            layer_idx,
            "mixer.experts.router.select_experts",
            outputs=True,
        )
        m_router = get_layer_entry(m, layer_idx, "mlp.router", outputs=True)
        m_gating = get_layer_entry(m, layer_idx, "mlp.router.gating", outputs=True)
        v_logits, _ = select_tensor(v_gate, ("output", "item0", "first"))
        v_topk_weights, _ = select_tensor(v_select, ("item0", "output", "first"))
        v_topk_ids, _ = select_tensor(v_select, ("item1",))
        m_logits, _ = select_tensor(m_gating, ("output", "item0", "first"))
        m_probs, _ = select_tensor(m_router, ("item0", "output", "first"))
        m_map, _ = select_tensor(m_router, ("item1",))
        if v_logits is None or m_probs is None:
            print(f"  {layer_idx:>5d} {'MISSING':<14s}")
            continue

        v_logits = normalize_router_layout(v_logits, seq_lens)
        m_probs = normalize_router_layout(m_probs, seq_lens).float()
        if m_map is None:
            m_map = m_probs != 0
        else:
            m_map = normalize_router_layout(m_map, seq_lens).bool()

        if v_logits.shape != m_probs.shape or m_probs.shape != m_map.shape:
            print(
                f"  {layer_idx:>5d} {'shape mismatch':<14s} "
                f"v={tuple(v_logits.shape)} m_probs={tuple(m_probs.shape)} "
                f"m_map={tuple(m_map.shape)}"
            )
            continue

        valid_rows = m_map.any(dim=1)
        if not bool(valid_rows.any()):
            print(f"  {layer_idx:>5d} {'empty map':<14s}")
            continue
        topk_values = m_map[valid_rows].sum(dim=1).float()
        topk = int(torch.median(topk_values).item())
        scaling_factor = float(torch.median(m_probs[valid_rows].sum(dim=1)).item())
        expert_bias = get_router_expert_bias(m, layer_idx)

        if v_topk_weights is not None and v_topk_ids is not None:
            v_topk_weights = normalize_router_layout(v_topk_weights, seq_lens)
            v_topk_ids = normalize_router_layout(v_topk_ids, seq_lens)
            if (
                v_topk_weights.dim() == 2
                and v_topk_ids.shape == v_topk_weights.shape
                and v_topk_weights.shape[0] == m_probs.shape[0]
            ):
                v_actual_probs, v_actual_map = sparse_routing_from_topk(
                    v_topk_weights,
                    v_topk_ids,
                    m_probs.shape[1],
                )
                row_match = (v_actual_map == m_map).all(dim=1).float().mean()
                entry_match = (v_actual_map == m_map).float().mean()
                raw_stats = diff_stats(v_actual_probs, m_probs)
                actual_sum = float(
                    torch.median(v_actual_probs[valid_rows].sum(dim=1)).item()
                )
                rescale = scaling_factor / actual_sum if actual_sum != 0.0 else 1.0
                rescaled_stats = diff_stats(v_actual_probs * rescale, m_probs)
                print(
                    f"        actual_vllm_select layer={layer_idx}: "
                    f"map_rows={float(row_match):.6f} "
                    f"map_entries={float(entry_match):.6f} "
                    f"v_sum={actual_sum:.4f} rescale={rescale:.4f} "
                    f"raw_prob_max={raw_stats['max_abs_diff']:.6f} "
                    f"raw_prob_mean={raw_stats['mean_abs_diff']:.4e} "
                    f"rescaled_prob_max={rescaled_stats['max_abs_diff']:.6f} "
                    f"rescaled_prob_mean={rescaled_stats['mean_abs_diff']:.4e} "
                    f"rescaled_prob_cos={rescaled_stats['cos_sim']:.6f}"
                )
            else:
                print(
                    f"        actual_vllm_select layer={layer_idx}: "
                    f"shape mismatch weights={tuple(v_topk_weights.shape)} "
                    f"ids={tuple(v_topk_ids.shape)} m={tuple(m_probs.shape)}"
                )

        candidate_results = router_candidate_results(
            v_logits,
            m_probs,
            m_map,
            topk,
            scaling_factor,
            expert_bias=expert_bias,
        )
        candidate_results.sort(
            key=lambda item: (
                item["row_match"],
                item["entry_match"],
                -item["prob_stats"]["mean_abs_diff"],
            ),
            reverse=True,
        )
        keep_labels = {"sigmoid_raw", "sigmoid_norm", "softmax_topk", "softmax_pre"}
        rows_to_print = [
            item
            for index, item in enumerate(candidate_results)
            if index < 8 or item["label"] in keep_labels
        ]
        printed_labels = set()
        for item in rows_to_print:
            if item["label"] in printed_labels:
                continue
            printed_labels.add(item["label"])
            prob_stats = item["prob_stats"]
            print(
                f"  {layer_idx:>5d} {item['label']:<16s} {topk:>5d} "
                f"{scaling_factor:>9.4f} "
                f"{item['row_match']:>10.6f} "
                f"{item['entry_match']:>11.6f} "
                f"{prob_stats['max_abs_diff']:>11.6f} "
                f"{prob_stats['mean_abs_diff']:>11.4e} "
                f"{prob_stats['cos_sim']:>10.6f}"
            )

        if m_logits is None:
            continue
        m_logits = normalize_router_layout(m_logits, seq_lens)
        if m_logits.shape != v_logits.shape:
            continue
        print_router_margin_report(
            layer_idx,
            v_logits=v_logits,
            m_logits=m_logits,
            m_map=m_map,
            topk=topk,
            expert_bias=expert_bias,
        )


def topk_boundary_margin(scores, topk):
    top_values = torch.topk(scores, k=topk + 1, dim=1).values
    return top_values[:, topk - 1] - top_values[:, topk]


def print_router_margin_report(
    layer_idx,
    *,
    v_logits,
    m_logits,
    m_map,
    topk,
    expert_bias=None,
):
    v_scores = torch.sigmoid(v_logits.float())
    m_scores = torch.sigmoid(m_logits.float())
    expert_bias = _prepare_expert_bias(expert_bias, m_scores)
    v_route_scores = v_scores + expert_bias if expert_bias is not None else v_scores
    m_route_scores = m_scores + expert_bias if expert_bias is not None else m_scores
    _, v_indices = torch.topk(v_route_scores, k=topk, dim=1)
    _, m_indices_from_logits = torch.topk(m_route_scores, k=topk, dim=1)
    v_map = torch.zeros_like(m_map, dtype=torch.bool)
    v_map.scatter_(1, v_indices, True)
    m_logit_map = torch.zeros_like(m_map, dtype=torch.bool)
    m_logit_map.scatter_(1, m_indices_from_logits, True)

    mismatch = (v_map != m_map).any(dim=1)
    m_logits_match_router = (m_logit_map == m_map).all(dim=1).float().mean()
    if not bool(mismatch.any()):
        print(
            f"        margin layer={layer_idx}: all sigmoid top-k rows match; "
            f"m_logits_vs_router_rows={float(m_logits_match_router):.6f}"
        )
        return

    v_margin = topk_boundary_margin(v_route_scores, topk)
    m_margin = topk_boundary_margin(m_route_scores, topk)
    row_logit_diff = (v_logits.float() - m_logits.float()).abs().max(dim=1).values
    mismatch_count = int(mismatch.sum().item())
    total_count = int(mismatch.numel())
    print(
        f"        margin layer={layer_idx}: mismatched_rows={mismatch_count}/{total_count} "
        f"m_logits_vs_router_rows={float(m_logits_match_router):.6f} "
        f"v_margin_median={float(v_margin[mismatch].median()):.6g} "
        f"m_margin_median={float(m_margin[mismatch].median()):.6g} "
        f"row_logit_diff_median={float(row_logit_diff[mismatch].median()):.6g} "
        f"row_logit_diff_max={float(row_logit_diff[mismatch].max()):.6g}"
    )


def print_expert_combine_report(v, m, seq_lens, layer_indices):
    print("\n" + "=" * 110)
    print("NemotronH routed expert combined-output diffs")
    print("=" * 110)
    print(
        f"  {'layer':>5s} {'mode':>9s} {'rows':>8s} {'matched':>8s} "
        f"{'all_max':>11s} {'all_mean':>11s} {'all_cos':>10s} "
        f"{'match_max':>11s} {'match_mean':>11s} {'match_cos':>10s} "
        f"{'pack_prob_max':>13s}"
    )

    for layer_idx in layer_indices:
        v_expert = get_layer_entry(v, layer_idx, "mixer.experts", outputs=True)
        v_shared = get_layer_entry(
            v, layer_idx, "mixer.shared_experts.down_proj", outputs=True
        )
        v_gate = get_layer_entry(v, layer_idx, "mixer.gate", outputs=True)
        m_router = get_layer_entry(m, layer_idx, "mlp.router", outputs=True)
        m_expert_in = get_layer_entry(m, layer_idx, "mlp.experts")
        m_expert_out = get_layer_entry(m, layer_idx, "mlp.experts", outputs=True)

        v_expert_t, _ = select_tensor(v_expert, ("output", "item0", "first"))
        v_shared_t, _ = select_tensor(v_shared, ("output", "item0", "first"))
        v_logits, _ = select_tensor(v_gate, ("output", "item0", "first"))
        m_router_probs, _ = select_tensor(m_router, ("item0", "output", "first"))
        m_router_map, _ = select_tensor(m_router, ("item1",))
        m_packed_probs, _ = select_tensor(m_expert_in, ("arg2",))
        m_packed_output, _ = select_tensor(m_expert_out, ("output", "item0", "first"))
        if (
            v_expert_t is None
            or v_logits is None
            or m_router_probs is None
            or m_router_map is None
            or m_packed_probs is None
            or m_packed_output is None
        ):
            print(f"  {layer_idx:>5d} {'MISSING':>8s}")
            continue

        m_router_probs = squeeze_singleton_router_batch(m_router_probs, seq_lens)
        m_router_map = squeeze_singleton_router_batch(m_router_map, seq_lens).bool()
        num_padded_tokens = m_router_map.shape[0]
        flat_indices = m_router_map.T.contiguous().reshape(-1).nonzero().flatten()
        token_indices = flat_indices % num_padded_tokens
        expert_indices = flat_indices // num_padded_tokens
        if m_packed_output.shape[0] != flat_indices.numel():
            print(
                f"  {layer_idx:>5d} {'shape':>8s} packed={tuple(m_packed_output.shape)} "
                f"routes={int(flat_indices.numel())}"
            )
            continue

        packed_prob_ref = m_router_probs[token_indices, expert_indices].float()
        pack_prob_diff = (packed_prob_ref - m_packed_probs.float()).abs()
        v_expert_t = normalize_router_layout(v_expert_t, seq_lens).float()
        if v_expert_t.shape[-1] != m_packed_output.shape[-1]:
            print(
                f"  {layer_idx:>5d} {'shape':>9s} "
                f"v={tuple(v_expert_t.shape)} m={tuple(m_packed_output.shape)}"
            )
            continue

        m_map = normalize_router_layout(m_router_map, seq_lens)
        v_logits = normalize_router_layout(v_logits, seq_lens)
        if v_shared_t is not None:
            v_shared_t = normalize_router_layout(v_shared_t, seq_lens).float()
            if v_shared_t.shape != v_expert_t.shape:
                v_shared_t = None
        v_targets = [("with_shared", v_expert_t)]
        if v_shared_t is not None:
            v_targets.append(("routed_only", v_expert_t - v_shared_t))
        topk = int(torch.median(m_map.sum(dim=1).float()).item())
        scale = float(
            torch.median(
                normalize_router_layout(m_router_probs, seq_lens).sum(dim=1)
            ).item()
        )
        best_candidate = best_router_candidate(
            v_logits,
            m_router_probs,
            m_map,
            topk,
            scale,
            expert_bias=get_router_expert_bias(m, layer_idx),
        )
        v_probs = best_candidate["probs"]
        v_map = best_candidate["map"]
        matched_rows = (v_map == m_map).all(dim=1)

        actual_indices = padded_router_to_actual_indices(seq_lens)
        if actual_indices is not None:
            actual_indices = actual_indices[token_indices.cpu()]
            valid_actual = actual_indices >= 0
            v_packed_probs = torch.zeros_like(m_packed_probs.float())
            v_packed_probs[valid_actual] = v_probs[
                actual_indices[valid_actual], expert_indices.cpu()[valid_actual]
            ].float()
            m_unweighted_output = (
                m_packed_output.float()
                / m_packed_probs.float().clamp_min(1.0e-12).unsqueeze(-1)
            )
            vprob_values = m_unweighted_output * v_packed_probs.unsqueeze(-1)
        else:
            vprob_values = m_packed_output.float()

        combine_modes = (
            (
                "prob_mul",
                m_packed_output.float() * m_packed_probs.float().unsqueeze(-1),
            ),
            ("as_is", m_packed_output.float()),
            ("vprob", vprob_values),
        )
        for mode_name, packed_values in combine_modes:
            combined_padded = torch.zeros(
                num_padded_tokens,
                m_packed_output.shape[-1],
                dtype=torch.float32,
            )
            combined_padded.index_add_(0, token_indices.cpu(), packed_values)
            combined = normalize_router_layout(combined_padded, seq_lens).float()
            for target_name, v_target in v_targets:
                label = mode_name if target_name == "with_shared" else f"{mode_name}-r"
                if v_target.shape != combined.shape:
                    print(
                        f"  {layer_idx:>5d} {label:>9s} {'shape':>8s} "
                        f"v={tuple(v_target.shape)} m={tuple(combined.shape)}"
                    )
                    continue

                all_stats = diff_stats(v_target, combined)
                if bool(matched_rows.any()):
                    matched_stats = diff_stats(
                        v_target[matched_rows], combined[matched_rows]
                    )
                    match_max = matched_stats["max_abs_diff"]
                    match_mean = matched_stats["mean_abs_diff"]
                    match_cos = matched_stats["cos_sim"]
                else:
                    match_max = float("nan")
                    match_mean = float("nan")
                    match_cos = float("nan")

                print(
                    f"  {layer_idx:>5d} {label:>9s} {v_target.shape[0]:>8d} "
                    f"{int(matched_rows.sum().item()):>8d} "
                    f"{all_stats['max_abs_diff']:>11.6f} "
                    f"{all_stats['mean_abs_diff']:>11.4e} "
                    f"{all_stats['cos_sim']:>10.6f} "
                    f"{match_max:>11.6f} {match_mean:>11.4e} {match_cos:>10.6f} "
                    f"{float(pack_prob_diff.max()):>13.6f}"
                )


def module_inputs(payload):
    return payload.get("module_inputs_by_layer", {})


def module_outputs(payload):
    return payload.get("module_outputs_by_layer", {})


def module_input_calls(payload):
    return payload.get("module_input_calls_by_layer", {})


def module_output_calls(payload):
    return payload.get("module_output_calls_by_layer", {})


def layer_map(source, layer_idx):
    layer = source.get(layer_idx)
    if layer is None:
        layer = source.get(str(layer_idx), {})
    return layer or {}


def get_layer_entry(payload, layer_idx, module_name, outputs=False):
    call_source = (
        module_output_calls(payload) if outputs else module_input_calls(payload)
    )
    calls = layer_map(call_source, layer_idx).get(module_name)
    if calls:
        return {"__calls__": calls}

    source = module_outputs(payload) if outputs else module_inputs(payload)
    return layer_map(source, layer_idx).get(module_name, {})


def captured_module_layers(payload):
    layers = payload.get("captured_module_layers")
    if isinstance(layers, torch.Tensor):
        layers = layers.flatten().tolist()
    if isinstance(layers, (list, tuple)):
        return sorted({int(layer_idx) for layer_idx in layers})

    # Older captures only fully hooked layer 0. Other layers usually contain
    # only the synthetic <layer> entrypoint hook.
    detail_layers = []
    for key, layer_entries in module_inputs(payload).items():
        if (
            isinstance(layer_entries, dict)
            and len(set(layer_entries) - {"<layer>"}) > 0
        ):
            detail_layers.append(int(key))
    return sorted(set(detail_layers or [0]))


def print_common_exact_name_report(v, m, seq_lens, layer_indices, outputs=False):
    direction = "output" if outputs else "input"
    print("\n" + "=" * 110)
    print(f"Common exact-name {direction} diffs for captured module-detail layers")
    print("=" * 110)
    v_entries = module_outputs(v) if outputs else module_inputs(v)
    m_entries = module_outputs(m) if outputs else module_inputs(m)
    if not outputs:
        v_selectors = (
            "kw.hidden_states+kw.residual",
            "arg1+arg2",
            "kw.hidden_states",
            "arg1",
            "arg0",
            "first",
        )
        m_selectors = ("arg0", "kw.hidden_states", "first")
    for layer_idx in layer_indices:
        v_layer = v_entries.get(layer_idx) or v_entries.get(str(layer_idx)) or {}
        m_layer = m_entries.get(layer_idx) or m_entries.get(str(layer_idx)) or {}
        common = sorted(set(v_layer) & set(m_layer))
        if not common:
            print(f"\n  layer {layer_idx}: no common captured module names")
            continue
        print(f"\n  layer {layer_idx}:")
        for name in common:
            if outputs and name == "<layer>":
                v_selectors = ("item0+item1", "output", "first")
                m_selectors = ("item0+item1", "output", "first")
            elif outputs:
                v_selectors = ("output", "item0", "first")
                m_selectors = ("output", "item0", "first")
            compare_selected_tensor(
                name,
                get_layer_entry(v, layer_idx, name, outputs=outputs),
                get_layer_entry(m, layer_idx, name, outputs=outputs),
                v_selectors,
                m_selectors,
                seq_lens,
            )
        print(f"    vLLM-only names: {sorted(set(v_layer) - set(m_layer))}")
        print(f"    Megatron-only names: {sorted(set(m_layer) - set(v_layer))}")


def layer_entry_stream(payload, layer_idx, seq_lens, is_vllm):
    entry = get_layer_entry(payload, layer_idx, "<layer>")
    if is_vllm:
        selectors = (
            "kw.hidden_states+kw.residual",
            "arg1+arg2",
            "kw.hidden_states",
            "arg1",
            "arg0",
            "first",
        )
    else:
        selectors = ("kw.hidden_states", "arg0", "first")
    tensor, selector = select_tensor(entry, selectors)
    if tensor is None:
        return None, selector
    return normalize_router_layout(tensor, seq_lens), selector


def last_token_offsets(seq_lens):
    offsets = []
    total = 0
    for seq_len in seq_lens:
        total += seq_len
        offsets.append(total - 1)
    return torch.tensor(offsets, dtype=torch.long)


def print_layer_entry_last_token_report(v, m, seq_lens, num_layers):
    if not seq_lens:
        return

    offsets = last_token_offsets(seq_lens)
    print("\n" + "=" * 110)
    print("Layer-entry last-token residual-stream diffs")
    print("=" * 110)
    print(f"  {'layer':>5s} {'top prompt max/mean hidden diffs':<90s}")
    for layer_idx in range(num_layers):
        v_t, _ = layer_entry_stream(v, layer_idx, seq_lens, is_vllm=True)
        m_t, _ = layer_entry_stream(m, layer_idx, seq_lens, is_vllm=False)
        if v_t is None or m_t is None:
            print(f"  {layer_idx:>5d} MISSING")
            continue
        if v_t.shape != m_t.shape or v_t.size(0) <= int(offsets[-1]):
            print(
                f"  {layer_idx:>5d} shape mismatch v={tuple(v_t.shape)} "
                f"m={tuple(m_t.shape)}"
            )
            continue

        diff = (v_t[offsets] - m_t[offsets]).abs().float()
        max_by_prompt = diff.max(dim=1).values
        mean_by_prompt = diff.mean(dim=1)
        top = torch.topk(max_by_prompt, k=min(5, len(seq_lens)))
        parts = [
            f"p{int(prompt_idx)}:{float(max_by_prompt[prompt_idx]):.6f}/"
            f"{float(mean_by_prompt[prompt_idx]):.2e}"
            for prompt_idx in top.indices
        ]
        print(f"  {layer_idx:>5d} {' '.join(parts):<90s}")


def print_module_capture_report(v, m):
    v_inputs = module_inputs(v)
    m_inputs = module_inputs(m)
    if not v_inputs or not m_inputs:
        return
    v_outputs = module_outputs(v)
    m_outputs = module_outputs(m)

    seq_lens = v.get("seq_lens") or m.get("seq_lens") or []
    num_layers = min(
        int(v.get("num_layers", len(v_inputs))),
        int(m.get("num_layers", len(m_inputs))),
    )

    print("\n" + "=" * 110)
    print("Layer-entry hidden-state input diffs")
    print("=" * 110)
    print(
        f"  {'layer':>5s} {'max_abs':>11s} {'mean_abs':>11s} {'cos_sim':>10s} "
        f"{'v_shape':>22s} {'m_shape':>22s} {'v_sel':>10s} {'m_sel':>10s}"
    )
    for layer_idx in range(num_layers):
        v_entry = get_layer_entry(v, layer_idx, "<layer>")
        m_entry = get_layer_entry(m, layer_idx, "<layer>")
        compare_selected_tensor(
            str(layer_idx),
            v_entry,
            m_entry,
            (
                "kw.hidden_states+kw.residual",
                "arg1+arg2",
                "kw.hidden_states",
                "arg1",
                "arg0",
                "first",
            ),
            ("kw.hidden_states", "arg0", "first"),
            seq_lens,
        )

    module_layers = sorted(
        set(captured_module_layers(v)) & set(captured_module_layers(m))
    )
    if not module_layers:
        module_layers = [0]

    print("\n" + "=" * 110)
    print("Semantic module input diffs for captured module-detail layers")
    print("=" * 110)
    print(
        f"  {'pair':<68s} {'max_abs':>11s} {'mean_abs':>11s} {'cos_sim':>10s} "
        f"{'v_shape':>22s} {'m_shape':>22s} {'v_sel':>10s} {'m_sel':>10s}"
    )
    pairs = [
        (
            "layer residual stream",
            "<layer>",
            "<layer>",
            ("kw.hidden_states+kw.residual", "arg1+arg2", "kw.hidden_states", "arg1"),
            ("kw.hidden_states", "arg0"),
        ),
        (
            "NemotronH Mamba pre-mixer norm input",
            "norm",
            "mixer.in_proj.norm",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH Mamba post-norm -> in_proj",
            "mixer.in_proj",
            "mixer.in_proj.linear",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH Mamba gated norm y input",
            "mixer.norm",
            "mixer.norm",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH Mamba gated norm z input",
            "mixer.norm",
            "mixer.norm",
            ("arg1",),
            ("arg1",),
        ),
        (
            "NemotronH Mamba out_proj input",
            "mixer.out_proj",
            "mixer.out_proj",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "input_layernorm input",
            "input_layernorm",
            "input_layernorm",
            ("arg0", "kw.hidden_states", "first"),
            ("arg0", "kw.hidden_states", "first"),
        ),
        (
            "post-norm hidden -> qkv",
            "self_attn.qkv_proj",
            "self_attention.linear_qkv.linear",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "SDPA q",
            "self_attn.attn",
            "self_attention.core_attention",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "SDPA k",
            "self_attn.attn",
            "self_attention.core_attention",
            ("arg1",),
            ("arg1",),
        ),
        (
            "SDPA v",
            "self_attn.attn",
            "self_attention.core_attention",
            ("arg2",),
            ("arg2",),
        ),
        (
            "attention output -> o_proj",
            "self_attn.o_proj",
            "self_attention.linear_proj",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "post-norm hidden -> fc1",
            "mlp.gate_up_proj",
            "mlp.linear_fc1.linear",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "post-activation -> down/fc2",
            "mlp.down_proj",
            "mlp.linear_fc2",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH norm -> mixer/mlp",
            "norm",
            "pre_mlp_layernorm",
            ("output", "first"),
            ("output", "first"),
        ),
        (
            "NemotronH mixer/mlp input",
            "mixer",
            "mlp",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH router hidden input",
            "mixer.gate",
            "mlp.router",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH raw router hidden input",
            "mixer.gate",
            "mlp.router.gating",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH shared expert fc1/up input",
            "mixer.shared_experts.up_proj",
            "mlp.shared_experts.linear_fc1",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH shared expert fc2/down input",
            "mixer.shared_experts.down_proj",
            "mlp.shared_experts.linear_fc2",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
        (
            "NemotronH experts module input",
            "mixer.experts",
            "mlp.experts",
            ("arg0", "first"),
            ("arg0", "first"),
        ),
    ]
    for layer_idx in module_layers:
        print(f"\n  layer {layer_idx}:")
        for label, v_name, m_name, v_selectors, m_selectors in pairs:
            v_entry = get_layer_entry(v, layer_idx, v_name)
            m_entry = get_layer_entry(m, layer_idx, m_name)
            compare_selected_tensor(
                label, v_entry, m_entry, v_selectors, m_selectors, seq_lens
            )

    if v_outputs and m_outputs:
        print("\n" + "=" * 110)
        print("Semantic module output diffs for captured module-detail layers")
        print("=" * 110)
        print(
            f"  {'pair':<68s} {'max_abs':>11s} {'mean_abs':>11s} "
            f"{'cos_sim':>10s} {'v_shape':>22s} {'m_shape':>22s} "
            f"{'v_sel':>10s} {'m_sel':>10s}"
        )
        output_pairs = [
            (
                "layer residual stream output",
                "<layer>",
                "<layer>",
                ("item0+item1", "output", "first"),
                ("item0+item1", "output", "first"),
            ),
            (
                "NemotronH Mamba pre-mixer norm output",
                "norm",
                "mixer.in_proj.norm",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH Mamba in_proj linear output",
                "mixer.in_proj",
                "mixer.in_proj.linear",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH Mamba gated norm output",
                "mixer.norm",
                "mixer.norm",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH Mamba out_proj output",
                "mixer.out_proj",
                "mixer.out_proj",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "input_layernorm output",
                "input_layernorm",
                "input_layernorm",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "qkv projection output",
                "self_attn.qkv_proj",
                "self_attention.linear_qkv.linear",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "attention kernel output",
                "self_attn.attn",
                "self_attention.core_attention",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "attention projection output",
                "self_attn.o_proj",
                "self_attention.linear_proj",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "mlp fc1/gate_up output",
                "mlp.gate_up_proj",
                "mlp.linear_fc1.linear",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "mlp down/fc2 output",
                "mlp.down_proj",
                "mlp.linear_fc2",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH norm -> mixer/mlp output",
                "norm",
                "pre_mlp_layernorm",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH mixer/mlp output",
                "mixer",
                "mlp",
                ("output", "item0", "first"),
                ("item0", "output", "first"),
            ),
            (
                "NemotronH router probs/map output (not raw logits)",
                "mixer.gate",
                "mlp.router",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH raw router logits",
                "mixer.gate",
                "mlp.router.gating",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH shared expert fc1/up output",
                "mixer.shared_experts.up_proj",
                "mlp.shared_experts.linear_fc1",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH shared expert fc2/down output",
                "mixer.shared_experts.down_proj",
                "mlp.shared_experts.linear_fc2",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
            (
                "NemotronH experts output",
                "mixer.experts",
                "mlp.experts",
                ("output", "item0", "first"),
                ("output", "item0", "first"),
            ),
        ]
        for layer_idx in module_layers:
            print(f"\n  layer {layer_idx}:")
            for label, v_name, m_name, v_selectors, m_selectors in output_pairs:
                v_entry = get_layer_entry(v, layer_idx, v_name, outputs=True)
                m_entry = get_layer_entry(m, layer_idx, m_name, outputs=True)
                compare_selected_tensor(
                    label, v_entry, m_entry, v_selectors, m_selectors, seq_lens
                )

    print_common_exact_name_report(v, m, seq_lens, module_layers)
    print_common_exact_name_report(v, m, seq_lens, module_layers, outputs=True)
    print_router_internal_report(m, seq_lens, module_layers)
    print_router_decision_report(v, m, seq_lens, module_layers)
    print_expert_combine_report(v, m, seq_lens, module_layers)
    print_layer_entry_last_token_report(v, m, seq_lens, num_layers)


def plot_scatter(v_logits, m_logits, out_path, max_points):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_flat = v_logits.reshape(-1).numpy()
    m_flat = m_logits.reshape(-1).numpy()
    assert v_flat.shape == m_flat.shape, (
        f"shape mismatch: vllm {v_flat.shape} vs megatron {m_flat.shape}"
    )

    # Compute similarity stats on the full tensors before any sub-sampling.
    full_stats = diff_stats(v_logits, m_logits)
    v_t = v_logits.reshape(-1).float()
    m_t = m_logits.reshape(-1).float()
    pearson = float(torch.corrcoef(torch.stack([v_t, m_t]))[0, 1].item())

    n_total = v_flat.size
    if max_points and n_total > max_points:
        rng = torch.Generator().manual_seed(0)
        sel = torch.randperm(n_total, generator=rng)[:max_points].numpy()
        v_pts = v_flat[sel]
        m_pts = m_flat[sel]
        sub_note = f" (sub-sampled {max_points}/{n_total})"
    else:
        v_pts = v_flat
        m_pts = m_flat
        sub_note = f" ({n_total} pts)"

    lo = float(min(v_pts.min(), m_pts.min()))
    hi = float(max(v_pts.max(), m_pts.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m_pts, v_pts, s=1, alpha=0.2, rasterized=True)
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=0.5, label="y = x")
    ax.set_xlabel("Megatron next-token logprobs (log_softmax of logits)")
    ax.set_ylabel("vLLM generation logprobs")
    ax.set_title(f"Next-token logprobs: vLLM vs Megatron{sub_note}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left")

    sim_text = (
        f"cos_sim   = {full_stats['cos_sim']:.6f}\n"
        f"pearson r = {pearson:.6f}\n"
        f"max |Δ|   = {full_stats['max_abs_diff']:.4g}\n"
        f"mean |Δ|  = {full_stats['mean_abs_diff']:.4g}\n"
        f"|vllm|    = {full_stats['a_norm']:.4g}\n"
        f"|mcore|   = {full_stats['b_norm']:.4g}"
    )
    ax.text(
        0.98,
        0.02,
        sim_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)
    m = torch.load(args.megatron, map_location="cpu", weights_only=False)

    v_logprobs = v.get("next_token_logprobs")
    m_logits = m.get("last_token_logits")
    if v_logprobs is None or m_logits is None:
        raise RuntimeError(
            "expected 'next_token_logprobs' in vllm capture and "
            "'last_token_logits' in megatron capture; re-run both scripts."
        )
    v_logprobs = v_logprobs.float()
    m_logits = m_logits.float()

    # Megatron's logits dim may pad past vLLM's tokenizer length (e.g.,
    # 128256 vs 128256 — usually equal, but truncate to the common range
    # just in case).
    v_vocab = v_logprobs.shape[-1]
    m_vocab = m_logits.shape[-1]
    vocab = min(v_vocab, m_vocab)
    if v_vocab != m_vocab:
        print(
            f"[warn] vocab mismatch: vllm={v_vocab} megatron={m_vocab}; "
            f"truncating to {vocab}"
        )
    v_logprobs = v_logprobs[..., :vocab]
    m_logits = m_logits[..., :vocab]

    # Convert Megatron's raw logits to logprobs so both sides are directly
    # comparable. vLLM normalizes on GPU, so CUDA is useful for separating
    # transformer-body drift from log-softmax implementation differences.
    if args.log_softmax_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--log-softmax-device cuda requires CUDA")
        m_logprobs = torch.log_softmax(m_logits.cuda(), dim=-1).cpu()
    else:
        m_logprobs = torch.log_softmax(m_logits, dim=-1)

    v_prompts = v.get("prompts", [])
    m_prompts = m.get("prompts", [])

    print("=" * 70)
    print(f"vllm     capture : {args.vllm}")
    print(f"megatron capture : {args.megatron}")
    print(f"num prompts      : vllm={len(v_prompts)} megatron={len(m_prompts)}")
    print(
        f"logprob shape    : vllm={tuple(v_logprobs.shape)} "
        f"megatron(logits)={tuple(m_logits.shape)}"
    )
    print(f"log_softmax dev  : {args.log_softmax_device}")
    print(
        f"model family     : vllm={v.get('model_family', '?')} "
        f"megatron={m.get('model_family', '?')}"
    )
    print(
        f"module layers    : vllm={v.get('captured_module_layers', '?')} "
        f"megatron={m.get('captured_module_layers', '?')}"
    )

    n = min(v_logprobs.shape[0], m_logprobs.shape[0])
    v_logprobs = v_logprobs[:n]
    m_logprobs = m_logprobs[:n]

    if v_prompts != m_prompts:
        print(
            "[warn] prompt lists differ between captures — comparing by index "
            "anyway, but results may be meaningless"
        )

    print("\nPer-prompt next-token logprobs:")
    print(
        f"  {'idx':>3} | {'max_abs':>10} | {'mean_abs':>10} | {'cos_sim':>9} | "
        f"{'|vllm|':>10} | {'|mcore|':>10}"
    )
    for i in range(n):
        stats = diff_stats(v_logprobs[i], m_logprobs[i])
        print(
            f"  {i:>3} | {stats['max_abs_diff']:>10.6f} | "
            f"{stats['mean_abs_diff']:>10.4e} | {stats['cos_sim']:>9.6f} | "
            f"{stats['a_norm']:>10.4f} | {stats['b_norm']:>10.4f}"
        )

    print("\nAggregate (all prompts, all vocab):")
    for k, val in diff_stats(v_logprobs, m_logprobs).items():
        print(f"  {k}: {val}")

    if args.target_token_ids_file:
        target_token_ids, target_metadata = load_target_token_ids(
            args.target_token_ids_file,
            args.target_token_ids_key,
        )
        target_count = min(n, len(target_token_ids))
        if target_count < n:
            print(
                "[warn] fewer target token ids than compared rows: "
                f"targets={len(target_token_ids)} rows={n}; using {target_count}"
            )
        selected_rows = torch.arange(target_count)
        selected_targets = torch.tensor(
            target_token_ids[:target_count], dtype=torch.long
        )
        v_selected = v_logprobs[selected_rows, selected_targets]
        m_selected = m_logprobs[selected_rows, selected_targets]
        selected_diff = m_selected - v_selected
        selected_abs = selected_diff.abs()
        selected_rel = selected_abs / torch.maximum(
            torch.maximum(v_selected.abs(), m_selected.abs()),
            torch.full_like(selected_abs, 1e-12),
        )

        print("\nSelected target-token logprobs:")
        print(f"  target file: {args.target_token_ids_file}")
        print(f"  n: {target_count}")
        print(f"  mean_abs_diff: {float(selected_abs.mean()):.8e}")
        print(f"  max_abs_diff: {float(selected_abs.max()):.8e}")
        print(f"  mean_rel_diff: {float(selected_rel.mean()):.8e}")
        print(f"  max_rel_diff: {float(selected_rel.max()):.8e}")
        print(f"  mean_signed_diff: {float(selected_diff.mean()):.8e}")
        print("  worst selected-token diffs:")
        worst_count = min(10, target_count)
        top_abs, top_indices = torch.topk(selected_abs, worst_count)
        for rank, (abs_value, index) in enumerate(
            zip(top_abs.tolist(), top_indices.tolist()),
            start=1,
        ):
            meta = (
                target_metadata[index]
                if isinstance(target_metadata, list) and index < len(target_metadata)
                else {}
            )
            print(
                "    "
                f"#{rank} row={index} "
                f"sample={meta.get('sample_idx', '?')} "
                f"pos={meta.get('position', '?')} "
                f"token={int(selected_targets[index].item())} "
                f"vllm={float(v_selected[index]):.8e} "
                f"megatron={float(m_selected[index]):.8e} "
                f"abs={float(abs_value):.8e} "
                f"rel={float(selected_rel[index]):.8e}"
            )

    plot_scatter(v_logprobs, m_logprobs, args.plot, args.max_points)
    print(f"\nscatter plot -> {args.plot}")

    if args.compare_modules or (
        "module_inputs_by_layer" in v and "module_inputs_by_layer" in m
    ):
        print_module_capture_report(v, m)


if __name__ == "__main__":
    main()
