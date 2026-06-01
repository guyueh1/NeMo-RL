"""Side-by-side dump and diff of every captured first-layer module input across
the vllm and megatron captures, including arg index > 0 (so residual-stream
positional args show up). Uses the already-saved *.pt files."""

import argparse
import os

import torch


DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm", default=os.path.join(DEFAULT_DIR, "vllm_capture.pt"))
    p.add_argument("--megatron", default=os.path.join(DEFAULT_DIR, "megatron_capture_split.pt"))
    return p.parse_args()


def describe(args):
    if not args:
        return "(no args)"
    parts = []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            parts.append(f"arg{i}=Tensor{tuple(a.shape)}|nrm={a.float().norm():.4f}")
        else:
            parts.append(f"arg{i}={type(a).__name__}")
    return ", ".join(parts)


def diff(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return float(d.max()), float(d.mean()), cos


def main():
    args = parse_args()
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)
    m = torch.load(args.megatron, map_location="cpu", weights_only=False)

    print("=" * 100)
    print("vLLM first-layer module captures")
    print("=" * 100)
    for name in sorted(v["first_layer_inputs"].keys()):
        print(f"  {name:<35s} {describe(v['first_layer_inputs'][name])}")

    print("\n" + "=" * 100)
    print("Megatron first-layer module captures")
    print("=" * 100)
    for name in sorted(m["first_layer_inputs"].keys()):
        print(f"  {name:<45s} {describe(m['first_layer_inputs'][name])}")

    # Hand-picked pairs that *should* see the same tensor on both sides.
    # Each entry: (vllm_name, mcore_name, arg_idx_v, arg_idx_m, label).
    pairs = [
        # (the layer's own hidden_states arg)
        ("<first_layer>",            "<first_layer>",            1, 0, "layer hidden_states (arg)"),
        # pre-attention norm input
        ("input_layernorm",          "input_layernorm",          0, 0, "input_layernorm input"),
        # post-norm activation = qkv input
        ("self_attn.qkv_proj",       "self_attention.linear_qkv.linear", 0, 0, "post-norm hidden -> qkv"),
        # the qkv-pre-fused-norm input on megatron side (norm input):
        ("self_attn.qkv_proj",       "self_attention.linear_qkv.norm",   0, 0, "qkv (vllm post-norm) vs norm-input (mcore pre-norm) [intentional mismatch]"),
        # SDPA inputs: vllm passes (q, k, v) positionally; mcore inside core_attention.
        ("self_attn.attn",           "self_attention.core_attention", 0, 0, "SDPA arg0"),
        ("self_attn.attn",           "self_attention.core_attention", 1, 1, "SDPA arg1"),
        ("self_attn.attn",           "self_attention.core_attention", 2, 2, "SDPA arg2"),
        # attention output -> o_proj
        ("self_attn.o_proj",         "self_attention.linear_proj", 0, 0, "attn out -> o_proj"),
        # vllm fused norm+add takes (h, r); the FIRST arg is attn output (norm ~21).
        # The SECOND arg is incoming residual. The mcore pre_mlp_layernorm input is the SUM.
        ("post_attention_layernorm", "pre_mlp_layernorm",         0, 0, "post-attn-LN arg0 (attn_out vs h+r) [boundary mismatch]"),
        ("post_attention_layernorm", "pre_mlp_layernorm",         1, 0, "post-attn-LN arg1 (residual)  vs h+r [boundary mismatch]"),
        # mlp / fc1: vllm post-norm hidden -> gate_up; mcore linear_fc1.linear is post-norm
        ("mlp.gate_up_proj",         "mlp.linear_fc1.linear",    0, 0, "post-norm hidden -> fc1"),
        # mcore linear_fc1.norm input is the residual stream (pre-norm).
        ("mlp.gate_up_proj",         "mlp.linear_fc1.norm",      0, 0, "fc1 (vllm post-norm) vs norm-input (mcore pre-norm) [intentional mismatch]"),
        # SwiGLU input: vllm splits gate_up internally inside mlp.act_fn (arg shape 11x28672).
        # No direct mcore module for the SwiGLU input — skip.
        # Down projection / fc2: post-activation input
        ("mlp.down_proj",            "mlp.linear_fc2",           0, 0, "post-act -> down/fc2"),
    ]

    # Also try: sum(vllm.post_attention_layernorm arg0 + arg1) vs pre_mlp_layernorm input.
    print("\n" + "=" * 100)
    print("Paired comparisons (lower = better)")
    print("=" * 100)
    print(f"  {'pair':<55s} {'max_abs':>11s} {'mean_abs':>11s} {'cos_sim':>10s} {'|v|':>10s} {'|m|':>10s}")

    for v_name, m_name, vi, mi, label in pairs:
        v_args = v["first_layer_inputs"].get(v_name)
        m_args = m["first_layer_inputs"].get(m_name)
        if v_args is None or m_args is None:
            print(f"  {label:<55s} {'MISSING':>11s}  vllm={v_name in v['first_layer_inputs']} mcore={m_name in m['first_layer_inputs']}")
            continue
        if len(v_args) <= vi or len(m_args) <= mi:
            print(f"  {label:<55s} {'ARG OOB':>11s}  vllm_args={len(v_args)} mcore_args={len(m_args)}")
            continue
        v_t, m_t = v_args[vi], m_args[mi]
        if not isinstance(v_t, torch.Tensor) or not isinstance(m_t, torch.Tensor):
            print(f"  {label:<55s} {'NOT TENSOR':>11s}")
            continue
        mx, mn, cs = diff(v_t, m_t)
        v_n = float(v_t.float().norm())
        m_n = float(m_t.float().norm())
        print(f"  {label:<55s} {mx:>11.6f} {mn:>11.4e} {cs:>10.6f} {v_n:>10.4f} {m_n:>10.4f}")

    # Special: vllm post_attention_layernorm arg0 + arg1  vs  mcore pre_mlp_layernorm arg0
    v_pln = v["first_layer_inputs"].get("post_attention_layernorm")
    m_pln = m["first_layer_inputs"].get("pre_mlp_layernorm")
    if v_pln is not None and m_pln is not None and len(v_pln) >= 2 and len(m_pln) >= 1:
        v_t = v_pln[0] + v_pln[1]
        m_t = m_pln[0]
        mx, mn, cs = diff(v_t, m_t)
        print(f"  {'vllm.post_attn_ln (arg0+arg1) vs mcore.pre_mlp_ln':<55s} "
              f"{mx:>11.6f} {mn:>11.4e} {cs:>10.6f} {float(v_t.float().norm()):>10.4f} {float(m_t.float().norm()):>10.4f}")


if __name__ == "__main__":
    main()
