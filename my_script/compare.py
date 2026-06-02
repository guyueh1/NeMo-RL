"""Compare vLLM vs Megatron last-token logits captured by ``vllm_forward.py``
and ``megatron_forward.py`` over a batch of real prompts.

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm", default=os.path.join(DEFAULT_DIR, "vllm_capture.pt"))
    p.add_argument("--megatron", default=os.path.join(DEFAULT_DIR, "megatron_capture.pt"))
    p.add_argument("--batch-invariant", action="store_true",
                   help="Use *_bi.pt files for both sides")
    p.add_argument("--mxfp8", action="store_true",
                   help="Use *_mxfp8*.pt files for both sides")
    p.add_argument("--plot", default=None,
                   help="Output path for the scatter PNG "
                        "(default: my_script/compare_logits_scatter.png).")
    p.add_argument("--max-points", type=int, default=200_000,
                   help="Subsample logit pairs to at most this many points "
                        "in the scatter plot (default: 200000). Set to 0 for all.")
    args = p.parse_args()
    suffix = ""
    if args.mxfp8:
        suffix += "_mxfp8"
    if args.batch_invariant:
        suffix += "_bi"
    if suffix:
        args.vllm = os.path.join(DEFAULT_DIR, f"vllm_capture{suffix}.pt")
        args.megatron = os.path.join(DEFAULT_DIR, f"megatron_capture{suffix}.pt")
    if args.plot is None:
        args.plot = os.path.join(DEFAULT_DIR, f"compare_logits_scatter{suffix}.png")
    return args


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
            torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0)
            ).item()
        ),
    }


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
    pearson = float(
        torch.corrcoef(torch.stack([v_t, m_t]))[0, 1].item()
    )

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
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8,
                  edgecolor="gray"),
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
        print(f"[warn] vocab mismatch: vllm={v_vocab} megatron={m_vocab}; "
              f"truncating to {vocab}")
    v_logprobs = v_logprobs[..., :vocab]
    m_logits = m_logits[..., :vocab]

    # Convert Megatron's raw logits to logprobs so both sides are directly
    # comparable.
    m_logprobs = torch.log_softmax(m_logits, dim=-1)

    v_prompts = v.get("prompts", [])
    m_prompts = m.get("prompts", [])

    print("=" * 70)
    print(f"vllm     capture : {args.vllm}")
    print(f"megatron capture : {args.megatron}")
    print(f"num prompts      : vllm={len(v_prompts)} megatron={len(m_prompts)}")
    print(f"logprob shape    : vllm={tuple(v_logprobs.shape)} "
          f"megatron(logits)={tuple(m_logits.shape)}")

    n = min(v_logprobs.shape[0], m_logprobs.shape[0])
    v_logprobs = v_logprobs[:n]
    m_logprobs = m_logprobs[:n]

    if v_prompts != m_prompts:
        print("[warn] prompt lists differ between captures — comparing by index "
              "anyway, but results may be meaningless")

    print("\nPer-prompt next-token logprobs:")
    print(f"  {'idx':>3} | {'max_abs':>10} | {'mean_abs':>10} | {'cos_sim':>9} | "
          f"{'|vllm|':>10} | {'|mcore|':>10}")
    for i in range(n):
        stats = diff_stats(v_logprobs[i], m_logprobs[i])
        print(f"  {i:>3} | {stats['max_abs_diff']:>10.6f} | "
              f"{stats['mean_abs_diff']:>10.4e} | {stats['cos_sim']:>9.6f} | "
              f"{stats['a_norm']:>10.4f} | {stats['b_norm']:>10.4f}")

    print("\nAggregate (all prompts, all vocab):")
    for k, val in diff_stats(v_logprobs, m_logprobs).items():
        print(f"  {k}: {val}")

    plot_scatter(v_logprobs, m_logprobs, args.plot, args.max_points)
    print(f"\nscatter plot -> {args.plot}")


if __name__ == "__main__":
    main()
