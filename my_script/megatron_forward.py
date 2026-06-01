"""Run a single prompt through a Megatron Llama-3.1-8B model with forward-only,
capture inputs to every module on the requested decoder layers, and save the
final logits.

BF16 (default):
    uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py

MXFP8 (Megatron consumes the BF16 HF ckpt unchanged and quantizes activations
and GEMM weights on the fly via TE's MXFP8BlockScaling recipe):
    uv run --extra mcore torchrun --nproc_per_node=1 my_script/megatron_forward.py \\
        --mxfp8 [--fp8-format e4m3|hybrid]

Setting ``--mxfp8`` sets ``provider.fp8`` to the chosen FP8 element format and
``provider.fp8_recipe = "mxfp8"``. ``TransformerBlock.forward`` then triggers
``fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling(...))`` for every
decoder layer, so all linears run as MXFP8 GEMMs. ``fp8_param`` is left at
default (False) — it's only required for the ``inference_optimized`` impl,
which we are not using.
"""

import argparse
import os

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, print_rank_0
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.module import Float16Module
from transformers import AutoTokenizer


DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def default_output(
    batch_invariant: bool,
    split_fused: bool,
    no_rope_fusion: bool = False,
    vllm_rope: bool = False,
    vllm_swiglu: bool = False,
    vllm_sdpa: bool = False,
    vllm_rmsnorm: bool = False,
    split_all_fused: bool = False,
    mxfp8: bool = False,
) -> str:
    suffix = ""
    if mxfp8:
        suffix += "_mxfp8"
    if split_all_fused:
        suffix += "_splitall"
    elif split_fused:
        suffix += "_split"
    if no_rope_fusion:
        suffix += "_norope"
    if vllm_rope:
        suffix += "_vllmrope"
    if vllm_swiglu:
        suffix += "_vllmswiglu"
    if vllm_sdpa:
        suffix += "_vllmsdpa"
    if vllm_rmsnorm:
        suffix += "_vllmrmsnorm"
    if batch_invariant:
        suffix += "_bi"
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"megatron_capture{suffix}.pt",
    )


def install_mxfp8_passthrough_for_bi_gemm():
    """When ``--batch-invariant`` and ``--mxfp8`` are both on, Megatron's BI
    matmul patch (``_te_general_gemm_patched``) can't handle MXFP8 tensors —
    it calls ``B.is_cuda`` which fails on ``MXFP8TensorStorage``.

    Wrap the patch so any GEMM call whose inputs aren't regular fp32/fp16/bf16
    CUDA tensors (i.e. the MXFP8 quantized linears) routes to TE's original
    ``general_gemm`` instead. BF16 GEMMs keep going through the BI path.

    This mirrors what vLLM already does for MXFP8: its BI matmul patches only
    intercept ``aten::{mm, addmm, matmul, linear}`` for plain bf16 tensors;
    the MXFP8 GEMM call path goes through ModelOpt's kernel which doesn't hit
    ``aten::mm``, so vLLM's BI patches don't apply to it either.

    Must be called AFTER ``enable_batch_invariant_mode()``.
    """
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik_mod
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.linear as te_linear_mod
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import megatron.core.extensions.transformer_engine as meg_te

    orig_patched = bik_mod._te_general_gemm_patched
    orig_te = bik_mod._TE_GENERAL_GEMM_ORIG
    if orig_te is None:
        raise RuntimeError("enable_batch_invariant_mode() must run before "
                           "install_mxfp8_passthrough_for_bi_gemm()")
    extract = bik_mod._extract_te_gemm_args
    regular_dtypes = (torch.bfloat16, torch.float16, torch.float32)

    def _is_regular(t):
        if t is None:
            return True
        if isinstance(t, torch.Tensor):
            return t.dtype in regular_dtypes
        return False

    def _wrapper(*args, **kwargs):
        a, b, *_rest = extract(args, kwargs)
        if not _is_regular(a) or not _is_regular(b):
            return orig_te(*args, **kwargs)
        return orig_patched(*args, **kwargs)

    for mod, attr in (
        (te_cpp, "general_gemm"),
        (te_linear_mod, "general_gemm"),
        (te_layernorm_linear_mod, "general_gemm"),
        (meg_te, "general_gemm"),
    ):
        if hasattr(mod, attr):
            setattr(mod, attr, _wrapper)


def install_vllm_style_rmsnorm():
    """Monkey-patch Megatron's BI RMSNorm to call vLLM's `rms_norm_batch_invariant`
    Triton kernel directly, so both engines run the *exact same* CUDA kernel.

    Megatron's default `BatchInvariantRMSNormFn` is a PyTorch implementation
    (`mean_dim(x*x)` -> `torch.sqrt` -> multiply). vLLM's `rms_norm_batch_invariant`
    is a single fused Triton kernel (`_rms_norm_kernel`: load-cast-square-reduce-
    rsqrt-multiply-cast). Even though both nominally use `BLOCK_SIZE=1024`
    sequential outer reductions and `1.0/sqrt(...)`, they differ in:
      - `tl.sqrt` (Triton, may compile to `sqrt.approx.f32`) vs `torch.sqrt`
        (PyTorch, IEEE-compliant `__fsqrt_rn`) — diverges by 1 ULP on some inputs.
      - Single fused kernel vs separate `x*x` materialisation + `mean_dim` call —
        same fp32 values in principle, but the Triton compiler may pick a
        different reduction tree than `mean_kernel`.

    By dispatching Megatron's BI RMSNorm through vLLM's exact Triton kernel,
    every RMSNorm call on either engine resolves to byte-identical kernel
    invocations on byte-identical inputs, guaranteeing bit-identical outputs.

    Must be called AFTER `enable_batch_invariant_mode()` — that's the call
    that installs `_te_rmsnorm_forward_patched` onto TE's RMSNorm; this patch
    swaps the autograd function that `_te_rmsnorm_forward_patched` invokes by
    global-name lookup.
    """
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik_mod
    from vllm.model_executor.layers.batch_invariant import (
        rms_norm as vllm_rms_norm_triton,
    )

    class _VllmStyleBatchInvariantRMSNormFn(torch.autograd.Function):
        """RMSNorm autograd Fn that dispatches to vLLM's exact BI Triton kernel."""

        @staticmethod
        def forward(ctx, x, weight, eps, zero_centered_gamma):
            if not x.is_cuda:
                raise RuntimeError("Batch-invariant RMSNorm requires CUDA tensors.")
            w_eff = (weight + 1.0) if zero_centered_gamma else weight
            # vLLM's rms_norm wrapper reshapes (..., H) to (-1, H), runs the
            # Triton kernel, and reshapes back. We just need to pass the input
            # and weight in the original dtype.
            out = vllm_rms_norm_triton(x, w_eff, eps)

            ctx.eps = eps
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.save_for_backward(x, weight)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            raise NotImplementedError(
                "Backward not implemented for the vllm-style RMSNorm patch. "
                "This script is forward-only."
            )

    bik_mod.BatchInvariantRMSNormFn = _VllmStyleBatchInvariantRMSNormFn


def install_vllm_style_sdpa():
    """Monkey-patch TE's `DotProductAttention.forward` to dispatch to vLLM's
    FA2 (`vllm.vllm_flash_attn.flash_attn_varlen_func`) with `num_splits=1`
    and `fa_version=2` — the exact kernel vLLM uses under
    `VLLM_BATCH_INVARIANT=1` on Blackwell.

    Why FA2 and not FA4: vLLM rejects FA4 in BI mode because FA4 uses
    batch-shape-dependent scheduling heuristics on SM100+ (see
    `vllm/v1/attention/backends/fa_utils.py:137-142`). Under BI, vLLM falls
    back to FA2. So matching vLLM's BI output bit-for-bit requires Megatron
    to also call FA2 — not FA4. (An earlier version of this patch called the
    FA4 cute kernel; that no longer matches what vLLM actually runs in BI.)

    This patch:
      1. Reshapes Megatron/TE's (s, b, n, d) tensors into FA2 varlen's packed
         (total_tokens, n, d) layout with a `cu_seqlens` describing one
         contiguous sequence per batch element.
      2. Calls `flash_attn_varlen_func(..., fa_version=2, num_splits=1)`.
      3. Reshapes the (total, n, d) output back to TE's expected
         (s, b, n*d) (or (b, s, n*d) for bshd) layout.

    Inference-only; no bias / sliding window / fp8 / sinks / paged KV.
    """
    import math
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as dpa_mod,
    )
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    def _vllm_fa2_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask=None,
        qkv_format=None,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        attn_mask_type=None,
        window_size=None,
        bottom_right_diagonal=None,
        checkpoint_core_attention=False,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
        alibi_slopes=None,
        fast_zero_fill=True,
        inference_params=None,
        pad_between_seqs=None,
        fp8_output=False,
        num_splits=1,
    ):
        assert core_attention_bias is None, "bias not supported in vllm-fa2 patch"
        assert alibi_slopes is None, "alibi not supported in vllm-fa2 patch"
        assert inference_params is None, "kv-cache inference not supported in vllm-fa2 patch"
        assert fp8_output is False, "fp8 output not supported in vllm-fa2 patch"

        fmt = qkv_format or getattr(self, "qkv_format", "sbhd")
        if fmt == "sbhd":
            # (s, b, n, d) -> (b, s, n, d) -> (b*s, n, d)
            s_q, b_q, n_q, d = query_layer.shape
            s_kv, b_kv, n_kv, _ = key_layer.shape
            q = query_layer.transpose(0, 1).reshape(b_q * s_q, n_q, d).contiguous()
            k = key_layer.transpose(0, 1).reshape(b_kv * s_kv, n_kv, d).contiguous()
            v = value_layer.transpose(0, 1).reshape(b_kv * s_kv, n_kv, d).contiguous()
        elif fmt == "bshd":
            b_q, s_q, n_q, d = query_layer.shape
            b_kv, s_kv, n_kv, _ = key_layer.shape
            q = query_layer.reshape(b_q * s_q, n_q, d).contiguous()
            k = key_layer.reshape(b_kv * s_kv, n_kv, d).contiguous()
            v = value_layer.reshape(b_kv * s_kv, n_kv, d).contiguous()
        else:
            raise NotImplementedError(f"qkv_format={fmt!r} not supported")

        cu_q = torch.arange(0, (b_q + 1) * s_q, s_q, dtype=torch.int32, device=q.device)
        cu_k = torch.arange(0, (b_kv + 1) * s_kv, s_kv, dtype=torch.int32, device=q.device)
        softmax_scale = 1.0 / math.sqrt(d)
        mt = attn_mask_type or "causal"
        causal = mt.startswith("causal")

        out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=s_q,
            cu_seqlens_q=cu_q,
            max_seqlen_k=s_kv,
            cu_seqlens_k=cu_k,
            softmax_scale=softmax_scale,
            causal=causal,
            num_splits=1,           # batch-invariant K reduction
            fa_version=2,           # match vLLM BI path
            deterministic=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        # out: (b*s, n_q, d)
        out = out.reshape(b_q, s_q, n_q, d)
        if fmt == "sbhd":
            out = out.transpose(0, 1)  # -> (s, b, n_q, d)
            return out.reshape(s_q, b_q, n_q * d).contiguous()
        return out.reshape(b_q, s_q, n_q * d).contiguous()

    dpa_mod.DotProductAttention.forward = _vllm_fa2_forward


def install_vllm_style_swiglu():
    """Monkey-patch Megatron's SwiGLU to match vLLM's `silu_and_mul` CUDA kernel.

    vLLM's kernel (`csrc/activation_kernels.cu`) does:
      silu_bf16 = (bf16)((float)gate / (1 + exp(-(float)gate)))   # round #1
      out_bf16  = silu_bf16 * up_bf16                              # round #2

    Megatron's default uses `@jit_fuser` (= `torch.compile`) on:
      def swiglu(y):
          y_1, y_2 = torch.chunk(y, 2, -1)
          return F.silu(y_1) * y_2
    which TorchInductor fuses into one Triton kernel keeping the silu result
    in fp32 registers — only ONE bf16 round at the final store. That extra
    precision is what creates the ~1-bf16-ULP drift at `linear_fc2` input
    vs vLLM's `down_proj` input.

    This patch replaces `swiglu` with an eager-mode version that runs as two
    separate kernels (silu materialises bf16, then bf16*bf16 multiply), so
    Megatron matches vLLM's two-rounding-event behaviour bit-for-bit.

    NOTE: This is a *downgrade* on Megatron's side for the sake of cross-
    engine bit equality. The "correct" long-term fix is to make vLLM's CUDA
    kernel keep silu in fp32 until the multiply, matching Megatron's compiled
    path. See TODO in `my_docs/llama3_8b_numeric_mismatch.md`.
    """
    import torch.nn.functional as F
    import megatron.core.fusions.fused_bias_swiglu as swg_mod

    def _vllm_style_swiglu(y):
        y_1, y_2 = torch.chunk(y, 2, -1)
        silu_out = F.silu(y_1)        # eager: fp32 compute, materialise bf16
        return silu_out * y_2          # bf16 * bf16

    def _vllm_style_bias_swiglu(y, bias):
        return _vllm_style_swiglu(y + bias)

    swg_mod.swiglu = _vllm_style_swiglu
    swg_mod.bias_swiglu = _vllm_style_bias_swiglu


def install_vllm_style_rope():
    """Monkey-patch Megatron's `apply_rotary_pos_emb` to match vLLM's numerical
    behaviour (precision + multiply-add order).

    vLLM:
      1. Precompute cos/sin in fp32 from fp32 freqs.
      2. Cast cos/sin to bf16 at module init (lossy).
      3. Inside the C++ wrapper, upcast cos_sin_cache back to fp32
         (`pos_encoding_kernels.cu:171`).
      4. CUDA kernel: read bf16 q/k -> fp32 -> apply
         `out_first  = x1*cos - x2*sin` (fp32)
         `out_second = x2*cos + x1*sin` (fp32)
         single bf16 cast per output element.

    This patch replicates all four steps in PyTorch so Megatron's intermediate
    bf16 rounding events disappear and cos/sin precision matches vLLM exactly.
    """
    from megatron.core.models.common.embeddings import rope_utils

    def _vllm_style_apply_rope(
        t: torch.Tensor,
        freqs: torch.Tensor,
        config,
        cu_seqlens=None,
        mscale: float = 1.0,
        cp_group=None,
    ):
        assert cu_seqlens is None, "vllm-style RoPE patch only supports non-thd"
        rot_dim = freqs.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

        # Step 1+2+3: compute cos/sin, cast through bf16 to bf16 (precision loss),
        # then back to fp32 (matches vLLM's `cache.to(kFloat32)` in the C++ wrapper).
        target_dtype = t.dtype
        cos = (torch.cos(freqs) * mscale).to(target_dtype).to(torch.float32)
        sin = (torch.sin(freqs) * mscale).to(target_dtype).to(torch.float32)

        # Megatron passes freqs already duplicated via cat(f, f). The first half
        # carries the real frequencies; the second half is just a copy. vLLM's
        # cache stores only the half-dim cos/sin. Take the first half so we apply
        # the same per-element cos to x1 and x2.
        cos_h = cos[..., : rot_dim // 2]
        sin_h = sin[..., : rot_dim // 2]

        # Step 4: rotation in fp32, single bf16 cast per output element.
        t_rot_fp32 = t_rot.to(torch.float32)
        if not getattr(config, "rotary_interleaved", False):
            # NeoX layout: t_rot = cat(x1, x2)
            x1, x2 = torch.chunk(t_rot_fp32, 2, dim=-1)
            o1 = x1 * cos_h - x2 * sin_h
            o2 = x2 * cos_h + x1 * sin_h
            out_fp32 = torch.cat((o1, o2), dim=-1)
        else:
            # GPT-J / interleaved layout
            x1 = t_rot_fp32[..., 0::2]
            x2 = t_rot_fp32[..., 1::2]
            o1 = x1 * cos_h - x2 * sin_h
            o2 = x2 * cos_h + x1 * sin_h
            out_fp32 = torch.stack((o1, o2), dim=-1).flatten(-2)

        out = out_fp32.to(target_dtype)
        return torch.cat((out, t_pass), dim=-1) if t_pass.numel() > 0 else out

    rope_utils.apply_rotary_pos_emb = _vllm_style_apply_rope
    # Also patch the re-export in transformer.attention if present so the
    # `attention.py` import binding is updated, not just the rope_utils module.
    import importlib
    for mod_name in ("megatron.core.transformer.attention",):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "apply_rotary_pos_emb"):
                mod.apply_rotary_pos_emb = _vllm_style_apply_rope
        except ImportError:
            pass


class SplitNormLinear(torch.nn.Module):
    """Drop-in replacement for `TELayerNormColumnParallelLinear` that materialises
    the post-norm tensor (so a forward hook on `linear` captures vLLM-comparable
    `qkv_proj` / `gate_up_proj` input)."""

    def __init__(self, norm, linear):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x):
        return self.linear(self.norm(x))


def split_layer_fused(layer, config):
    """Unfuse `self_attention.linear_qkv` and `mlp.linear_fc1` on one decoder
    layer using the upstream `split_te_layernorm_column_parallel_linear`
    primitive, replacing each with a tiny `SplitNormLinear(norm, linear)`
    wrapper. This forces the post-norm activation to be materialised in bf16
    (an extra round-trip) so the layer's numerics match vLLM's standalone
    `RMSNorm` + `qkv_proj` / `gate_up_proj` path exactly.
    """
    from megatron.core.extensions.transformer_engine import (
        split_te_layernorm_column_parallel_linear,
    )

    for owner, attr in (
        (layer.self_attention, "linear_qkv"),
        (layer.mlp, "linear_fc1"),
    ):
        fused = getattr(owner, attr)
        norm, linear = split_te_layernorm_column_parallel_linear(fused, config)
        norm = norm.to(device=fused.weight.device, dtype=fused.weight.dtype)
        linear = linear.to(device=fused.weight.device, dtype=fused.weight.dtype)
        setattr(owner, attr, SplitNormLinear(norm, linear))


def split_first_layer_fused(first_layer, config):
    """Back-compat wrapper around `split_layer_fused` for layer 0 only."""
    split_layer_fused(first_layer, config)


def split_all_layers_fused(decoder, config):
    """Apply `split_layer_fused` to every decoder layer (not just layer 0)."""
    for i, layer in enumerate(decoder.layers):
        split_layer_fused(layer, config)
    return len(decoder.layers)


class SingleBatchIterator:
    def __init__(self, input_ids, position_ids):
        self.batch = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def forward_step(data_iterator, model, **kwargs):
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **_):
        return x

    return model(**forward_args), loss_func


def unwrap(m):
    if isinstance(m, Float16Module):
        m = m.module
    if hasattr(m, "language_model"):
        m = m.language_model
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", default=None)
    p.add_argument("--batch-invariant", action="store_true")
    p.add_argument("--split-fused", action="store_true",
                   help="Unfuse LN+Linear on first layer (linear_qkv, linear_fc1)")
    p.add_argument("--split-all-fused", action="store_true",
                   help="Unfuse LN+Linear on ALL decoder layers (implies "
                        "--split-fused for layer 0). Required for full-depth "
                        "bit-equality with vLLM's standalone-norm + linear path.")
    p.add_argument("--no-rope-fusion", action="store_true",
                   help="Disable TE fused RoPE (apply_rope_fusion=False) to "
                        "match vLLM's bf16-cast-of-cos/sin precision behavior.")
    p.add_argument("--vllm-rope", action="store_true",
                   help="Monkey-patch Megatron's apply_rotary_pos_emb with a "
                        "PyTorch RoPE that reproduces vLLM's exact precision "
                        "behaviour (bf16 cos/sin cache, fp32 rotation, single "
                        "bf16 cast at end).")
    p.add_argument("--vllm-swiglu", action="store_true",
                   help="Monkey-patch Megatron's swiglu to bypass torch.compile "
                        "fusion and match vLLM's two-rounding-event SwiGLU "
                        "(eager F.silu then bf16 multiply).")
    p.add_argument("--vllm-sdpa", action="store_true",
                   help="Monkey-patch TE DotProductAttention to call vLLM's "
                        "FA2 (flash_attn_varlen_func, fa_version=2, "
                        "num_splits=1) — the kernel vLLM actually uses under "
                        "VLLM_BATCH_INVARIANT=1 on Blackwell.")
    p.add_argument("--vllm-rmsnorm", action="store_true",
                   help="Monkey-patch Megatron's BI RMSNorm to use 1/sqrt(...) "
                        "instead of torch.rsqrt(...), matching vLLM's original "
                        "Triton BI RMSNorm kernel bit-for-bit. Only meaningful "
                        "in combination with --batch-invariant.")
    p.add_argument("--mxfp8", action="store_true",
                   help="Enable MXFP8 (Blackwell-only). Configures the model "
                        "provider with fp8=<format> and fp8_recipe='mxfp8' so "
                        "TE's fp8_autocast wraps every decoder layer.")
    p.add_argument("--fp8-format", default="e4m3", choices=["e4m3", "hybrid"],
                   help="FP8 element format used by the MXFP8 recipe (default: "
                        "e4m3). 'hybrid' uses e4m3 for fwd and e5m2 for bwd. "
                        "Only meaningful with --mxfp8.")
    p.add_argument("--capture-layers", default="0",
                   help="Comma-separated 0-indexed decoder layer numbers to capture "
                        "per-module input tensors for (default: 0).")
    args = p.parse_args()
    args.capture_layers = [int(x) for x in args.capture_layers.split(",") if x.strip()]
    if args.split_all_fused:
        args.split_fused = True   # imply layer-0 split when all-layers split
    if args.output is None:
        args.output = default_output(
            args.batch_invariant,
            args.split_fused,
            args.no_rope_fusion,
            args.vllm_rope,
            args.vllm_swiglu,
            args.vllm_sdpa,
            args.vllm_rmsnorm,
            args.split_all_fused,
            args.mxfp8,
        )
    return args


def main():
    args = parse_args()

    if args.batch_invariant:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            enable_batch_invariant_mode,
        )
        enable_batch_invariant_mode()
        print_rank_0("[megatron] batch_invariant_mode ENABLED")
        if args.mxfp8:
            install_mxfp8_passthrough_for_bi_gemm()
            print_rank_0("[megatron] BI GEMM patch wrapped: MXFP8 tensors fall "
                         "through to TE's original general_gemm")

    if args.vllm_rmsnorm:
        if not args.batch_invariant:
            raise SystemExit("--vllm-rmsnorm requires --batch-invariant "
                             "(patches the BI RMSNorm autograd function)")
        install_vllm_style_rmsnorm()
        print_rank_0("[megatron] vllm-style RMSNorm patched (1/sqrt instead of rsqrt)")

    if args.vllm_rope:
        # Force apply_rope_fusion off so the python apply_rotary_pos_emb path
        # is taken (which is what we monkey-patch).
        install_vllm_style_rope()
        print_rank_0("[megatron] vllm-style RoPE patched into apply_rotary_pos_emb")

    if args.vllm_swiglu:
        install_vllm_style_swiglu()
        print_rank_0("[megatron] vllm-style SwiGLU patched (eager F.silu + bf16 mul)")

    if args.vllm_sdpa:
        install_vllm_style_sdpa()
        print_rank_0("[megatron] TE DotProductAttention.forward patched -> vLLM FA2 (num_splits=1)")

    print_rank_0(f"[megatron] loading bridge for {args.model}")
    bridge = AutoBridge.from_hf_pretrained(
        args.model,
        trust_remote_code=is_safe_repo(trust_remote_code=False, hf_path=args.model),
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.expert_model_parallel_size = 1
    model_provider.expert_tensor_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    # Forward-only inference — disable grad-fusion knobs that need APEX.
    model_provider.gradient_accumulation_fusion = False
    if args.no_rope_fusion or args.vllm_rope:
        # Use the unfused PyTorch RoPE path so our monkey-patch is actually
        # called (TE fused path is a CUDA kernel, not patchable from Python).
        model_provider.apply_rope_fusion = False

    if args.mxfp8:
        # Setting `fp8` to a non-None format flips on TE's `fp8_autocast` inside
        # `TransformerBlock.forward`; `fp8_recipe="mxfp8"` selects
        # `MXFP8BlockScaling` (Blackwell-only). Weights stay BF16 on disk/host —
        # TE quantizes per-block (1x32, E8M0 scales) at the GEMM boundary.
        model_provider.fp8 = args.fp8_format
        model_provider.fp8_recipe = "mxfp8"
        print_rank_0(f"[megatron] MXFP8 enabled: fp8={args.fp8_format}, fp8_recipe=mxfp8")

    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    model_list = model_provider.provide_distributed_model(wrap_with_ddp=False)
    model_list = [m.cuda() for m in model_list]
    for m in model_list:
        m.eval()
        disable_mtp_for_inference(m)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids_list = tokenizer.encode(args.prompt, add_special_tokens=True)
    print_rank_0(f"[megatron] prompt: {args.prompt!r}")
    print_rank_0(f"[megatron] token ids ({len(token_ids_list)}): {token_ids_list}")

    # TE's MXFP8 quantizer requires the product of leading dims (= seq_len *
    # batch for our (s, b, h) layout) to be divisible by 32 (the MXFP8 block
    # size). Pad seq_len up to the next multiple of 32 with the EOS token;
    # the padded positions don't affect the captured tensors at the original
    # positions thanks to causal attention, and compare.py's min-length
    # truncation discards them anyway.
    real_seq_len = len(token_ids_list)
    if args.mxfp8 and real_seq_len % 32 != 0:
        padded_len = ((real_seq_len + 31) // 32) * 32
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        token_ids_list = token_ids_list + [pad_id] * (padded_len - real_seq_len)
        print_rank_0(f"[megatron] padded seq_len {real_seq_len} -> {padded_len} for MXFP8 "
                     f"(TE MXFP8 block size requires divisibility by 32)")

    input_ids = torch.tensor([token_ids_list], dtype=torch.long, device="cuda")
    position_ids = torch.arange(
        input_ids.size(1), dtype=torch.long, device=input_ids.device
    ).unsqueeze(0).expand_as(input_ids)

    # Register hooks on the (single, last-stage) model chunk.
    assert len(model_list) == 1, "expected one model chunk with pp=1"
    inner = unwrap(model_list[0])

    if args.split_all_fused:
        n_split = split_all_layers_fused(inner.decoder, inner.config)
        print_rank_0(f"[megatron] split fused LN+Linear on ALL {n_split} decoder layers "
                     "(linear_qkv -> SplitNormLinear, linear_fc1 -> SplitNormLinear)")
    elif args.split_fused:
        split_first_layer_fused(inner.decoder.layers[0], inner.config)
        print_rank_0("[megatron] split fused LN+Linear on first layer "
                     "(linear_qkv -> SplitNormLinear(norm, linear), "
                     "linear_fc1 -> SplitNormLinear(norm, linear))")

    captured_module_inputs = {idx: {} for idx in args.capture_layers}
    captured_layer_outputs = {}
    handles = []

    def make_hook(layer_idx, name):
        bucket = captured_module_inputs[layer_idx]
        def hook(module, args_, output_):
            saved = []
            for a in args_:
                if isinstance(a, torch.Tensor):
                    saved.append(a.detach().to(torch.float32).cpu().clone())
                else:
                    saved.append(a)
            if name not in bucket:
                bucket[name] = saved
        return hook

    def make_layer_output_hook(idx):
        def hook(module, args_, output_):
            # TransformerLayer may return a tensor or (hidden_states, context) tuple.
            t = output_
            if isinstance(t, tuple):
                t = t[0]
            if isinstance(t, torch.Tensor):
                captured_layer_outputs[idx] = t.detach().to(torch.float32).cpu().clone()
        return hook

    for layer_idx in args.capture_layers:
        layer = inner.decoder.layers[layer_idx]
        for name, sub in layer.named_modules():
            qual = name if name else "<layer>"
            handles.append(sub.register_forward_hook(make_hook(layer_idx, qual)))

    for i, layer in enumerate(inner.decoder.layers):
        handles.append(layer.register_forward_hook(make_layer_output_hook(i)))

    # Forward pass (forward-only).
    with torch.no_grad():
        fwd_bwd = get_forward_backward_func()
        iterator = SingleBatchIterator(input_ids, position_ids)
        output = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=iterator,
            model=model_list,
            num_microbatches=1,
            forward_only=True,
            seq_length=input_ids.size(1),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )

    for h in handles:
        h.remove()

    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    logits_cpu = None
    if parallel_state.is_pipeline_last_stage() and isinstance(output, torch.Tensor):
        # With tp=1 there's nothing to all-gather; just move to CPU.
        logits_cpu = output.detach().to(torch.float32).cpu()
        print_rank_0(f"[megatron] logits shape: {tuple(logits_cpu.shape)}")

    if dist.is_initialized() and dist.get_rank() == 0:
        payload = {
            "prompt": args.prompt,
            "token_ids": token_ids_list,
            # Legacy alias for backward compat with older compare.py.
            "first_layer_inputs": captured_module_inputs.get(0, {}),
            "module_inputs_by_layer": captured_module_inputs,
            "layer_outputs": captured_layer_outputs,
            "logits": logits_cpu,
        }
        torch.save(payload, args.output)
        print(f"[megatron] saved capture to {args.output}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
