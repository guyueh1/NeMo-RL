"""Compare MXFP8 quantization between vLLM and Megatron-TE on a real weight.

Runs in the mcore venv so both TE and vLLM imports are available. Loads a
BF16 weight tensor from the HF Llama ckpt, quantizes it three ways, and
compares the resulting (fp8_data, e8m0_scale) tuples plus the bf16
dequantized values.

Usage:
    uv run --extra mcore python my_script/compare_mxfp8_quant.py
"""

import argparse
import os
import sys

import torch
from safetensors.torch import safe_open
from transformers import AutoModelForCausalLM


DEFAULT_BF16_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MXFP8_PATH = (
    "/lustre/fsw/coreai_dlalgo_llm/users/guyueh/checkpoints/"
    "meta-llama--Llama-3.1-8B-Instruct.mxfp8"
)
DEFAULT_PARAM = "model.layers.0.self_attn.q_proj.weight"


def load_bf16_weight(bf16_id: str, key: str) -> torch.Tensor:
    print(f"[load] HF ckpt {bf16_id} -> {key}")
    model = AutoModelForCausalLM.from_pretrained(
        bf16_id, torch_dtype=torch.bfloat16
    )
    state = dict(model.named_parameters())
    return state[key].detach().cuda()


def load_vllm_mxfp8_ckpt_weight(
    mxfp8_path: str, key: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read fp8 data + uint8 E8M0 scale from the MXFP8 safetensors ckpt."""
    index_path = os.path.join(mxfp8_path, "model.safetensors.index.json")
    import json
    with open(index_path) as f:
        index = json.load(f)
    weight_shard = index["weight_map"][key]
    scale_shard = index["weight_map"][key + "_scale"]
    with safe_open(os.path.join(mxfp8_path, weight_shard), framework="pt") as f:
        data = f.get_tensor(key).cuda()
    if scale_shard == weight_shard:
        with safe_open(os.path.join(mxfp8_path, scale_shard), framework="pt") as f:
            scale = f.get_tensor(key + "_scale").cuda()
    else:
        with safe_open(os.path.join(mxfp8_path, scale_shard), framework="pt") as f:
            scale = f.get_tensor(key + "_scale").cuda()
    return data, scale


def te_quantize(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize via TE's MXFP8 quantizer (Megatron path)."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    q = MXFP8Quantizer(
        fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
    )
    q.optimize_for_gemm = False
    out = q.quantize_impl(w_bf16)
    return out._rowwise_data, out._rowwise_scale_inv


def vllm_quantize(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize via vLLM's mxfp8_e4m3_quantize (FlashInfer on Blackwell, torch fallback elsewhere)."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    return mxfp8_e4m3_quantize(w_bf16, is_sf_swizzled_layout=False)


def vllm_dequant(data: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        dequant_mxfp8_to_bf16,
    )

    return dequant_mxfp8_to_bf16(data, scale)


def fp8_byte_diff(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    # Compare fp8 tensors byte-wise via int8 reinterpret.
    a_u = a.view(torch.uint8)
    b_u = b.view(torch.uint8)
    same = (a_u == b_u).all().item()
    n_diff = (a_u != b_u).sum().item()
    pct = n_diff * 100.0 / a_u.numel()
    print(f"  [{label}] fp8 byte-identical: {same}  diff_count={n_diff} ({pct:.4f}%)")


def scale_byte_diff(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    # Scales are uint8; compare directly.
    a_flat = a.flatten()
    b_flat = b.flatten()
    n = min(a_flat.numel(), b_flat.numel())
    a_flat = a_flat[:n]
    b_flat = b_flat[:n]
    same = (a_flat == b_flat).all().item()
    diff = (a_flat.to(torch.int32) - b_flat.to(torch.int32)).abs()
    print(
        f"  [{label}] scale-byte-identical: {same}  max_diff={diff.max().item()}  "
        f"mean_diff={diff.float().mean().item():.4f}  diff_count={(diff != 0).sum().item()}"
    )


def bf16_diff(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    n = min(a32.numel(), b32.numel())
    a32 = a32[:n]
    b32 = b32[:n]
    d = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(
        a32.unsqueeze(0), b32.unsqueeze(0)
    ).item()
    print(
        f"  [{label}] max_abs={d.max().item():.6g}  mean_abs={d.mean().item():.4e}  cos={cos:.6f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16-id", default=DEFAULT_BF16_ID)
    p.add_argument("--mxfp8-path", default=DEFAULT_MXFP8_PATH)
    p.add_argument("--param", default=DEFAULT_PARAM,
                   help="HF parameter name (e.g. model.layers.0.self_attn.q_proj.weight)")
    args = p.parse_args()

    w_bf16 = load_bf16_weight(args.bf16_id, args.param)
    print(f"[bf16] shape={tuple(w_bf16.shape)} dtype={w_bf16.dtype} "
          f"norm={w_bf16.float().norm().item():.4f}")

    print()
    print("== Quantization paths ==")
    print()

    # 1. vLLM converter output (read from disk).
    print("[1] vLLM converter (from MXFP8 ckpt on disk):")
    conv_data, conv_scale = load_vllm_mxfp8_ckpt_weight(args.mxfp8_path, args.param)
    print(f"    data shape={tuple(conv_data.shape)} dtype={conv_data.dtype}")
    print(f"    scale shape={tuple(conv_scale.shape)} dtype={conv_scale.dtype}")
    print(f"    scale[0,:8]={conv_scale.flatten()[:8].tolist()}")

    # 2. TE's MXFP8 quantizer applied to the same bf16 weight.
    print()
    print("[2] TE quantize_impl (Megatron runtime path):")
    te_data, te_scale = te_quantize(w_bf16)
    print(f"    data shape={tuple(te_data.shape)} dtype={te_data.dtype}")
    print(f"    scale shape={tuple(te_scale.shape)} dtype={te_scale.dtype}")
    print(f"    scale[0,:8]={te_scale.flatten()[:8].tolist()}")

    # 3. vLLM's runtime quantizer (FlashInfer on Blackwell).
    print()
    print("[3] vLLM mxfp8_e4m3_quantize (runtime; FlashInfer on Blackwell):")
    vllm_rt_data, vllm_rt_scale = vllm_quantize(w_bf16)
    print(f"    data shape={tuple(vllm_rt_data.shape)} dtype={vllm_rt_data.dtype}")
    print(f"    scale shape={tuple(vllm_rt_scale.shape)} dtype={vllm_rt_scale.dtype}")
    print(f"    scale[0,:8]={vllm_rt_scale.flatten()[:8].tolist()}")

    # Align all to same scale shape — TE may store padded; converter is compact.
    # Truncate to compact shape: [N, K/32].
    N, K = w_bf16.shape
    expected_scale_shape = (N, K // 32)
    print(f"\n[expected compact scale shape] {expected_scale_shape}")

    def truncate_scale(s):
        if s.ndim == 1:
            return s.view(N, -1)[:, : K // 32].contiguous()
        if s.ndim == 2 and s.shape != expected_scale_shape:
            return s[: N, : K // 32].contiguous()
        return s

    conv_scale_t = truncate_scale(conv_scale)
    te_scale_t = truncate_scale(te_scale)
    vllm_rt_scale_t = truncate_scale(vllm_rt_scale)
    print(f"  conv_scale truncated: {tuple(conv_scale_t.shape)}")
    print(f"  te_scale truncated:   {tuple(te_scale_t.shape)}")
    print(f"  vllm_rt_scale truncated: {tuple(vllm_rt_scale_t.shape)}")

    print()
    print("== Pairwise comparisons (uint8 scale, fp8 data, bf16 dequanted) ==")

    print()
    print("(converter [1]) vs (TE [2]):")
    scale_byte_diff(conv_scale_t, te_scale_t, "scale")
    fp8_byte_diff(conv_data, te_data, "data")
    bf16_diff(vllm_dequant(conv_data, conv_scale_t),
              vllm_dequant(te_data, te_scale_t),
              "dequanted-bf16")

    print()
    print("(converter [1]) vs (vLLM runtime [3]):")
    scale_byte_diff(conv_scale_t, vllm_rt_scale_t, "scale")
    fp8_byte_diff(conv_data, vllm_rt_data, "data")
    bf16_diff(vllm_dequant(conv_data, conv_scale_t),
              vllm_dequant(vllm_rt_data, vllm_rt_scale_t),
              "dequanted-bf16")

    print()
    print("(TE [2]) vs (vLLM runtime [3]):")
    scale_byte_diff(te_scale_t, vllm_rt_scale_t, "scale")
    fp8_byte_diff(te_data, vllm_rt_data, "data")
    bf16_diff(vllm_dequant(te_data, te_scale_t),
              vllm_dequant(vllm_rt_data, vllm_rt_scale_t),
              "dequanted-bf16")


if __name__ == "__main__":
    main()
