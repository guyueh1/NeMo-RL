"""Run vLLM's FA4 cute kernel on Megatron-saved Q/K/V tensors, then compare
the result against:
  (a) Megatron's actual SDPA output (the `linear_proj` arg0 from the capture)
  (b) vLLM's actual SDPA output (the `o_proj` arg0 from `vllm_capture_bi.pt`)

This decouples the kernel-choice difference from the input difference:
- diff(vLLM_FA4(mcore_qkv), mcore_actual_sdpa_out) == TE-vs-FA4 kernel gap
- diff(vLLM_FA4(mcore_qkv), vllm_actual_sdpa_out)  should be ~0 since
  vllm_qkv == mcore_qkv bit-identical, modulo the layout/reshape we apply
  here.

Run with: uv run --extra vllm python my_script/run_vllm_fa4_on_mcore_qkv.py
"""

import argparse
import math
import os

import torch


DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--megatron",
        default=os.path.join(DEFAULT_DIR, "megatron_capture_split_vllmrope_vllmswiglu_bi.pt"),
    )
    p.add_argument(
        "--vllm", default=os.path.join(DEFAULT_DIR, "vllm_capture_bi.pt")
    )
    p.add_argument(
        "--num-splits", type=int, default=1,
        help="num_splits for FA4 (1 = batch-invariant)",
    )
    return p.parse_args()


def diff_stats(a, b):
    a = a.float().reshape(-1).cpu()
    b = b.float().reshape(-1).cpu()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return float(d.max()), float(d.mean()), cos, float(a.norm()), float(b.norm())


def main():
    args = parse_args()

    m = torch.load(args.megatron, map_location="cpu", weights_only=False)
    v = torch.load(args.vllm, map_location="cpu", weights_only=False)

    # Pull Megatron Q/K/V (post-RoPE) — shape (s, b, n, d).
    mcore_attn_args = m["first_layer_inputs"]["self_attention.core_attention"]
    q_m = mcore_attn_args[0]   # (s, b, n_q, d) = (11, 1, 32, 128)
    k_m = mcore_attn_args[1]   # (s, b, n_kv, d) = (11, 1, 8, 128)
    v_m = mcore_attn_args[2]
    print(f"[load] mcore Q {tuple(q_m.shape)} norm={q_m.float().norm():.4f}")
    print(f"[load] mcore K {tuple(k_m.shape)} norm={k_m.float().norm():.4f}")
    print(f"[load] mcore V {tuple(v_m.shape)} norm={v_m.float().norm():.4f}")

    # Megatron's actual attention output (== linear_proj input).
    mcore_actual_out = m["first_layer_inputs"]["self_attention.linear_proj"][0]
    print(f"[load] mcore actual SDPA out: {tuple(mcore_actual_out.shape)} norm={mcore_actual_out.float().norm():.4f}")

    # vLLM Q/K/V — shape (num_tokens, n*d) for both, separately. Captured
    # at `self_attn.attn` as (Q, K, V).
    vllm_attn_args = v["first_layer_inputs"]["self_attn.attn"]
    q_v = vllm_attn_args[0]   # (11, 4096) = (s, n*d)
    k_v = vllm_attn_args[1]   # (11, 1024)
    v_v = vllm_attn_args[2]
    print(f"[load] vllm Q {tuple(q_v.shape)} norm={q_v.float().norm():.4f}")
    print(f"[load] vllm K {tuple(k_v.shape)} norm={k_v.float().norm():.4f}")
    print(f"[load] vllm V {tuple(v_v.shape)} norm={v_v.float().norm():.4f}")

    # vLLM's actual SDPA output (== o_proj input). Shape (s, n*d).
    vllm_actual_out = v["first_layer_inputs"]["self_attn.o_proj"][0]
    print(f"[load] vllm actual SDPA out: {tuple(vllm_actual_out.shape)} norm={vllm_actual_out.float().norm():.4f}")

    # Confirm Q/K/V match between engines (already verified earlier but recheck).
    q_m_for_compare = q_m.squeeze(1).reshape(q_m.shape[0], -1)   # (s, n*d)
    k_m_for_compare = k_m.squeeze(1).reshape(k_m.shape[0], -1)
    v_m_for_compare = v_m.squeeze(1).reshape(v_m.shape[0], -1)
    print("\n[sanity] mcore vs vllm Q/K/V (should be bit-identical):")
    for name, a, b in [("Q", q_m_for_compare, q_v), ("K", k_m_for_compare, k_v), ("V", v_m_for_compare, v_v)]:
        mx, mn, cs, na, nb = diff_stats(a, b)
        print(f"  {name}: max={mx:.6g} mean={mn:.6g} cos={cs:.7f}  |mcore|={na:.4f} |vllm|={nb:.4f}")

    # Move to GPU and bf16.
    device = "cuda"
    # vLLM cute expects (b, s, n, d) layout. Convert mcore (s, b, n, d) -> (b, s, n, d).
    q_in = q_m.to(device=device, dtype=torch.bfloat16).transpose(0, 1).contiguous()
    k_in = k_m.to(device=device, dtype=torch.bfloat16).transpose(0, 1).contiguous()
    v_in = v_m.to(device=device, dtype=torch.bfloat16).transpose(0, 1).contiguous()
    head_dim = q_in.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)
    print(f"\n[fa4] calling vllm FA4 with q={tuple(q_in.shape)} k={tuple(k_in.shape)} v={tuple(v_in.shape)} "
          f"softmax_scale={softmax_scale:.6f} num_splits={args.num_splits}")

    from vllm.vllm_flash_attn.cute.interface import flash_attn_func as fa4

    out = fa4(
        q_in, k_in, v_in,
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=args.num_splits,
        deterministic=False,
    )
    if isinstance(out, tuple):
        print(f"[fa4] returned tuple of len {len(out)}; using out[0]")
        out = out[0]
    print(f"[fa4] returned: {tuple(out.shape)} dtype={out.dtype}")

    # Reshape vLLM-style output. fa4 returns (b, s, n, d). Megatron's
    # linear_proj sees (s, b, n*d). vLLM's o_proj sees (s, n*d).
    out = out.float().cpu()                                # (b, s, n, d)
    b, s, n, d = out.shape
    out_mcore_layout = out.transpose(0, 1).reshape(s, b, n * d)   # (s, b, n*d)
    out_vllm_layout  = out.transpose(0, 1).reshape(s * b, n * d)  # (s, n*d) since b=1

    print("\n[compare A] vLLM-FA4(mcore_QKV) vs Megatron actual attention output:")
    mx, mn, cs, na, nb = diff_stats(out_mcore_layout, mcore_actual_out)
    print(f"  max={mx:.6g} mean={mn:.6g} cos={cs:.7f} |fa4|={na:.4f} |te|={nb:.4f}")
    print("  -> measures TE-DotProductAttention vs vLLM-FA4 kernel-choice gap")

    print("\n[compare B] vLLM-FA4(mcore_QKV) vs vLLM actual attention output:")
    mx, mn, cs, na, nb = diff_stats(out_vllm_layout, vllm_actual_out)
    print(f"  max={mx:.6g} mean={mn:.6g} cos={cs:.7f} |fa4|={na:.4f} |vllm|={nb:.4f}")
    print("  -> should be ~0 (same kernel, same QKV, just different invocation)")


if __name__ == "__main__":
    main()
