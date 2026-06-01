"""Compare vLLM vs Megatron captures saved by vllm_forward.py and megatron_forward.py.

Run with either env:
    uv run --extra vllm python my_script/compare.py
    uv run --extra mcore python my_script/compare.py
"""

import argparse
import os

import torch


DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm", default=os.path.join(DEFAULT_DIR, "vllm_capture.pt"))
    p.add_argument("--megatron", default=os.path.join(DEFAULT_DIR, "megatron_capture.pt"))
    p.add_argument("--batch-invariant", action="store_true",
                   help="Use *_bi.pt files for both sides")
    p.add_argument("--split-fused", action="store_true",
                   help="Use the megatron *_split.pt capture (LN+Linear unfused on first layer)")
    args = p.parse_args()
    suffix = ""
    if args.split_fused:
        suffix += "_split"
    if args.batch_invariant:
        suffix += "_bi"
    if suffix:
        args.vllm = os.path.join(DEFAULT_DIR, f"vllm_capture{suffix.replace('_split', '')}.pt")
        args.megatron = os.path.join(DEFAULT_DIR, f"megatron_capture{suffix}.pt")
    return args


def fmt_shape(t):
    if isinstance(t, torch.Tensor):
        return f"Tensor{tuple(t.shape)} dtype={t.dtype}"
    return repr(t)[:80]


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
        "cos_sim": float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()),
    }


def main():
    args = parse_args()
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)
    m = torch.load(args.megatron, map_location="cpu", weights_only=False)

    print("=" * 70)
    print(f"vllm prompt    : {v['prompt']!r}")
    print(f"megatron prompt: {m['prompt']!r}")
    print(f"vllm tokens    : {v['token_ids']}")
    print(f"megatron tokens: {m['token_ids']}")
    print("=" * 70)

    print("\nFinal logits:")
    print(f"  vllm    : {fmt_shape(v['logits'])}")
    print(f"  megatron: {fmt_shape(m['logits'])}")
    if isinstance(v["logits"], torch.Tensor) and isinstance(m["logits"], torch.Tensor):
        # Try to align on the seq_len * vocab dims. vLLM may emit a smaller
        # tensor (e.g. only sampled positions) than Megatron's full (B,S,V).
        vl = v["logits"].reshape(-1)
        ml = m["logits"].reshape(-1)
        # Last-row comparison: take the last seq position from megatron logits.
        if m["logits"].dim() == 3 and v["logits"].dim() <= 2:
            m_last = m["logits"][0, -1].reshape(-1)
            v_last = v["logits"].reshape(-1)[: m_last.numel()]
            print("  comparing last-position logits (truncated to vllm shape):")
            for k, val in diff_stats(v_last, m_last).items():
                print(f"    {k}: {val}")
        else:
            print("  raw flat comparison:")
            for k, val in diff_stats(vl, ml).items():
                print(f"    {k}: {val}")

    print("\nPer-layer residual-stream output (post layer i):")
    v_layers = v.get("layer_outputs") or {}
    m_layers = m.get("layer_outputs") or {}
    common = sorted(set(v_layers.keys()) & set(m_layers.keys()))
    if not common:
        print("  no layer_outputs captured on at least one side")
    else:
        print(f"  {'layer':>5} | {'max_abs':>11} | {'mean_abs':>11} | {'cos_sim':>9} | {'|vllm|':>10} | {'|mcore|':>10}")
        for i in common:
            v_t = v_layers[i]
            m_t = m_layers[i]
            # vllm captured as (S,H); mcore captured as (S,B,H) — flatten both.
            stats = diff_stats(v_t, m_t)
            print(f"  {i:>5} | {stats['max_abs_diff']:>11.6f} | {stats['mean_abs_diff']:>11.6e} | "
                  f"{stats['cos_sim']:>9.6f} | {stats['a_norm']:>10.4f} | {stats['b_norm']:>10.4f}")

    # Pull per-layer module-input captures; fall back to legacy first_layer_inputs.
    v_by_layer = v.get("module_inputs_by_layer") or {0: v.get("first_layer_inputs", {})}
    m_by_layer = m.get("module_inputs_by_layer") or {0: m.get("first_layer_inputs", {})}
    common_layers = sorted(set(v_by_layer.keys()) & set(m_by_layer.keys()))

    for layer_idx in common_layers:
        v_modules = v_by_layer[layer_idx]
        m_modules = m_by_layer[layer_idx]
        print(f"\nLayer {layer_idx} module inputs:")
        print(f"  vllm modules    : {sorted(v_modules.keys())}")
        print(f"  megatron modules: {sorted(m_modules.keys())}")

        # Heuristic name alignment between the two stacks.
        # When the megatron capture was produced with --split-fused / --split-all-fused,
        # the post-norm tensor is captured under `*.linear_qkv.linear` /
        # `*.linear_fc1.linear`, so prefer those if present.
        qkv_mname = (
            "self_attention.linear_qkv.linear"
            if "self_attention.linear_qkv.linear" in m_modules
            else "self_attention.linear_qkv"
        )
        fc1_mname = (
            "mlp.linear_fc1.linear"
            if "mlp.linear_fc1.linear" in m_modules
            else "mlp.linear_fc1"
        )
        layer_root = "<first_layer>" if layer_idx == 0 else "<layer>"
        v_layer_root = "<first_layer>" if "<first_layer>" in v_modules else "<layer>"
        m_layer_root = "<first_layer>" if "<first_layer>" in m_modules else "<layer>"
        name_pairs = [
            (v_layer_root, m_layer_root),
            ("input_layernorm", "input_layernorm"),
            ("self_attn", "self_attention"),
            ("self_attn.qkv_proj", qkv_mname),
            ("self_attn.o_proj", "self_attention.linear_proj"),
            ("post_attention_layernorm", "pre_mlp_layernorm"),
            ("mlp", "mlp"),
            ("mlp.gate_up_proj", fc1_mname),
            ("mlp.down_proj", "mlp.linear_fc2"),
        ]
        print(f"\n  Paired module input comparison (layer {layer_idx}, first tensor arg only):")
        for v_name, m_name in name_pairs:
            v_in = v_modules.get(v_name)
            m_in = m_modules.get(m_name)
            if v_in is None or m_in is None:
                print(f"    {v_name} <-> {m_name}: missing on one side "
                      f"(vllm={'yes' if v_in else 'no'}, mcore={'yes' if m_in else 'no'})")
                continue
            v_t = next((x for x in v_in if isinstance(x, torch.Tensor)), None)
            m_t = next((x for x in m_in if isinstance(x, torch.Tensor)), None)
            if v_t is None or m_t is None:
                print(f"    {v_name} <-> {m_name}: no tensor arg captured")
                continue
            print(f"    {v_name} <-> {m_name}: vllm {fmt_shape(v_t)} | mcore {fmt_shape(m_t)}")
            try:
                for k, val in diff_stats(v_t, m_t).items():
                    print(f"      {k}: {val}")
            except Exception as e:
                print(f"      comparison failed: {e}")


if __name__ == "__main__":
    main()
