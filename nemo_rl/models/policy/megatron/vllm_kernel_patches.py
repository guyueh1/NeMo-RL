# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Monkey-patches that make Megatron-LM produce bit-identical forward outputs
to vLLM on Blackwell for the BF16 and MXFP8 batch-invariant paths.

Originally developed under ``my_script/megatron_forward.py`` as part of the
cross-engine numerical-matching effort (see
``my_docs/llama3_8b_numeric_mismatch.md`` and
``my_docs/llama3_8b_mxfp8_numeric_mismatch.md``). Ported here so an RL run
can install the same patches inside ``MegatronPolicyWorker.__init__`` when
``policy.megatron_cfg.match_vllm_kernels`` is set to True.

All six patches are class-level / module-level monkey-patches: they take
effect for every layer of every model from the moment they're applied,
without any per-layer instrumentation.

The four BF16-only patches are safe to apply unconditionally; the two
MXFP8-only patches are gated on ``mxfp8=True`` in
``install_match_vllm_kernels``.
"""

from __future__ import annotations

import importlib
import math

import torch


def install_vllm_style_rmsnorm() -> None:
    """Route Megatron's BI RMSNorm through vLLM's exact Triton kernel.

    Replaces ``MegatronCore``'s ``BatchInvariantRMSNormFn`` (PyTorch
    ``mean_dim(x*x) + torch.sqrt`` chain) with one that dispatches to
    vLLM's ``rms_norm_batch_invariant`` Triton kernel. Both engines then
    invoke the literally identical kernel on identical inputs, producing
    byte-identical output.

    Must be called AFTER ``enable_batch_invariant_mode()`` so the
    ``_te_rmsnorm_forward_patched`` global lookup resolves to the
    patched ``BatchInvariantRMSNormFn``.
    """
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )
    from vllm.model_executor.layers.batch_invariant import (
        rms_norm as vllm_rms_norm_triton,
    )

    class _VllmStyleBatchInvariantRMSNormFn(torch.autograd.Function):
        """RMSNorm autograd Fn dispatching to vLLM's BI Triton kernel.

        Forward dispatches to vLLM's `rms_norm` Triton kernel for bit-identical
        cross-engine output. Backward follows the standard RMSNorm gradient
        formula in fp32; rsigma is recomputed (cheap) instead of being saved.

        With `y_i = x_i * w_eff_i * r` where `r = (mean(x^2) + eps)^(-1/2)`:
            dL/dw_i = sum_batch(go_i * x_i * r)
            dL/dx_i = go_i * w_eff_i * r
                      - (r^3 / H) * x_i * sum_j(go_j * w_eff_j * x_j)
        and `w_eff = w + 1` if `zero_centered_gamma`, else `w_eff = w` (the
        +1 is a constant offset, so dL/dw = dL/dw_eff unchanged).
        """

        @staticmethod
        def forward(ctx, x, weight, eps, zero_centered_gamma):
            print("[guyueh] _VllmStyleBatchInvariantRMSNormFn.forward")
            if not x.is_cuda:
                raise RuntimeError("Batch-invariant RMSNorm requires CUDA tensors.")
            w_eff = (weight + 1.0) if zero_centered_gamma else weight
            out = vllm_rms_norm_triton(x, w_eff, eps)
            ctx.eps = eps
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.save_for_backward(x, weight)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            eps = ctx.eps
            w_eff = (weight + 1.0) if ctx.zero_centered_gamma else weight

            x_fp32 = x.float()
            w_fp32 = w_eff.to(device=x.device, dtype=torch.float32)
            go_fp32 = grad_output.float()
            hidden_size = x.shape[-1]

            # Recompute rsigma in fp32 (~1 fp32 op per element; matches the
            # `1.0 / torch.sqrt(mean(x^2) + eps)` formula used by vLLM's
            # Triton kernel up to sqrt precision).
            mean_sq = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
            rsigma = 1.0 / torch.sqrt(mean_sq + eps)

            # grad_weight: sum_batch(go * x * rsigma). Reduce over every
            # leading dim; keep only the last (hidden) dim.
            reduce_dims = tuple(range(go_fp32.ndim - 1))
            grad_weight = (
                (go_fp32 * x_fp32 * rsigma).sum(dim=reduce_dims).to(weight.dtype)
            )

            # grad_x: combine the direct term and the through-rsigma term.
            inner = (go_fp32 * x_fp32 * w_fp32).sum(dim=-1, keepdim=True)
            rsigma_cubed = rsigma * rsigma * rsigma
            grad_x_fp32 = (
                go_fp32 * w_fp32 * rsigma
                - (w_fp32 * rsigma_cubed) * inner * x_fp32 / hidden_size
            )
            grad_x = grad_x_fp32.to(x.dtype)

            return grad_x, grad_weight, None, None

    bik_mod.BatchInvariantRMSNormFn = _VllmStyleBatchInvariantRMSNormFn


def install_vllm_style_rmsnom_to_te() -> None:
    from transformer_engine.pytorch.module import _common as te_common_mod
    from vllm.model_executor.layers.batch_invariant import (
        rms_norm as vllm_rms_norm_triton,
    )

    orig_apply_norm = te_common_mod.apply_normalization
    def apply_norm(*args, **kwargs):
        nomalization = args[7] if len(args) > 7 else kwargs["normalization"]
        if not normalization == "RMSNorm":
            return orig_apply_norm(*args, **kwargs)
        print("[guyueh] apply_norm")
        x, weight, eps, zero_centered_gamma = args[0], args[2], args[4], args[-1]
        w_eff = (weight + 1.0) if zero_centered_gamma else weight
        out = vllm_rms_norm_triton(x, w_eff, eps)
        _, mu, sigma = orig_apply_norm(*args, **kwargs)
        return out, mu, sigma

    te_common_mod.apply_normalization = apply_norm
    
    from transformer_engine.pytorch import module as te_module_mod
    te_module_mod.apply_normalization = apply_norm


def install_vllm_style_rope() -> None:
    """Replace Megatron's ``apply_rotary_pos_emb`` to match vLLM's RoPE numerics.

    vLLM's recipe (per ``csrc/pos_encoding_kernels.cu``):
      1. fp32 cos/sin from fp32 freqs.
      2. cast cos/sin to bf16 at module init (lossy).
      3. inside the C++ wrapper, upcast cos_sin_cache back to fp32.
      4. read bf16 q/k -> fp32 -> rotation in fp32 -> single bf16 store.

    This patch replicates all four steps in PyTorch so Megatron's
    intermediate bf16 rounding events disappear.
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
        print("[guyueh] _vllm_style_apply_rope")
        assert cu_seqlens is None, "vllm-style RoPE patch only supports non-thd"
        rot_dim = freqs.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]

        # Steps 1-3: compute cos/sin in fp32, cast through bf16 (precision
        # loss), then back to fp32.
        target_dtype = t.dtype
        cos = (torch.cos(freqs) * mscale).to(target_dtype).to(torch.float32)
        sin = (torch.sin(freqs) * mscale).to(target_dtype).to(torch.float32)

        # Megatron passes freqs already duplicated via cat(f, f). Take the
        # first half to apply the same per-element cos to x1 and x2.
        cos_h = cos[..., : rot_dim // 2]
        sin_h = sin[..., : rot_dim // 2]

        # Step 4: rotation in fp32, single bf16 cast at end.
        t_rot_fp32 = t_rot.to(torch.float32)
        if not getattr(config, "rotary_interleaved", False):
            x1, x2 = torch.chunk(t_rot_fp32, 2, dim=-1)
            o1 = x1 * cos_h - x2 * sin_h
            o2 = x2 * cos_h + x1 * sin_h
            out_fp32 = torch.cat((o1, o2), dim=-1)
        else:
            x1 = t_rot_fp32[..., 0::2]
            x2 = t_rot_fp32[..., 1::2]
            o1 = x1 * cos_h - x2 * sin_h
            o2 = x2 * cos_h + x1 * sin_h
            out_fp32 = torch.stack((o1, o2), dim=-1).flatten(-2)

        out = out_fp32.to(target_dtype)
        return torch.cat((out, t_pass), dim=-1) if t_pass.numel() > 0 else out

    rope_utils.apply_rotary_pos_emb = _vllm_style_apply_rope
    # Also patch the re-export in transformer.attention so any imports of
    # ``apply_rotary_pos_emb`` from that module pick up the patched version.
    for mod_name in ("megatron.core.transformer.attention",):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "apply_rotary_pos_emb"):
                mod.apply_rotary_pos_emb = _vllm_style_apply_rope
        except ImportError:
            pass


def install_vllm_style_swiglu() -> None:
    """Replace Megatron's SwiGLU to match vLLM's ``silu_and_mul`` CUDA kernel.

    vLLM's hand-written CUDA kernel does two bf16 rounding events:
    ``silu(gate) -> bf16; bf16 * up_bf16 -> bf16``. Megatron's default
    ``@jit_fuser`` (``torch.compile``) fuses the chain into one Triton
    kernel that keeps the silu output in fp32 (only ONE bf16 round at the
    final store), creating ~1 bf16 ULP drift at ``linear_fc2`` input.

    This patch replaces ``swiglu`` with an eager-mode version that runs as
    two separate kernels (silu materialises bf16, then bf16*bf16 multiply),
    so Megatron matches vLLM bit-for-bit.
    """
    import torch.nn.functional as F
    import megatron.core.fusions.fused_bias_swiglu as swg_mod

    def _vllm_style_swiglu(y):
        print("[guyueh] _vllm_style_swiglu")
        y_1, y_2 = torch.chunk(y, 2, -1)
        silu_out = F.silu(y_1)
        return silu_out * y_2

    def _vllm_style_bias_swiglu(y, bias):
        print("[guyueh] _vllm_style_bias_swiglu")
        return _vllm_style_swiglu(y + bias)

    swg_mod.swiglu = _vllm_style_swiglu
    swg_mod.bias_swiglu = _vllm_style_bias_swiglu


def install_vllm_style_sdpa() -> None:
    """Route TE's ``DotProductAttention`` through vLLM's FA2 kernel.

    On Blackwell, vLLM under ``VLLM_BATCH_INVARIANT=1`` rejects FA4 (per
    ``vllm/v1/attention/backends/fa_utils.py:137-142``) and falls back to
    FA2 with ``num_splits=1``. Megatron/TE on Blackwell does not have an
    FA4 path either, but defaults to a different kernel (cuDNN-fused
    attention) that doesn't match FA2 byte-for-byte. This patch makes
    Megatron call exactly the same ``vllm.vllm_flash_attn.flash_attn_varlen_func``
    that vLLM does, with ``num_splits=1, fa_version=2``.

    Forward-only; backward / kv-cache / fp8_output / sliding-window not
    supported in this wrapper.
    """
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
        print("[guyueh] _vllm_fa2_forward")
        assert core_attention_bias is None, "bias not supported in vllm-fa2 patch"
        assert alibi_slopes is None, "alibi not supported in vllm-fa2 patch"
        assert inference_params is None, (
            "kv-cache inference not supported in vllm-fa2 patch"
        )
        assert fp8_output is False, "fp8 output not supported in vllm-fa2 patch"

        fmt = qkv_format or getattr(self, "qkv_format", "sbhd")
        if fmt == "sbhd":
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

        cu_q = torch.arange(
            0, (b_q + 1) * s_q, s_q, dtype=torch.int32, device=q.device
        )
        cu_k = torch.arange(
            0, (b_kv + 1) * s_kv, s_kv, dtype=torch.int32, device=q.device
        )
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
            num_splits=1,
            fa_version=2,
            deterministic=False,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out.reshape(b_q, s_q, n_q, d)
        if fmt == "sbhd":
            out = out.transpose(0, 1)
            return out.reshape(s_q, b_q, n_q * d).contiguous()
        return out.reshape(b_q, s_q, n_q * d).contiguous()

    dpa_mod.DotProductAttention.forward = _vllm_fa2_forward


def install_mxfp8_compact_scales() -> None:
    """Force MXFP8 quantizers to produce compact (non-swizzled) scales.

    TE's MXFP8 quantizer stores per-block scales in a swizzled
    cuBLASLt-friendly layout when ``optimize_for_gemm=True``, but the C++
    behind ``tex.dequantize`` rejects swizzled tensors. The
    ``install_mxfp8_dequant_for_bi_gemm`` patch calls ``.dequantize()`` on
    every MXFP8 input, so we need compact scales everywhere.

    Patching ``make_empty`` isn't enough — TE explicitly sets
    ``input_quantizer.optimize_for_gemm = True`` just before
    ``tex.quantize`` in ``basic_linear.py:353`` /
    ``forward_grouped_mlp.py:287`` / ``backward_grouped_mlp.py:352``. We
    replace the attribute with a class-level ``property`` that always
    reads ``False`` and silently swallows ``True`` writes.
    """
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    def _get(self):
        return False

    def _set(self, value):  # noqa: ARG001 — silently swallow writes
        pass

    MXFP8Quantizer.optimize_for_gemm = property(_get, _set)


def install_mxfp8_passthrough_for_bi_gemm() -> None:
    """Route MXFP8 GEMMs around the BI matmul; BF16 GEMMs still hit BI Triton.

    Interim fix that lets ``batch_invariant_mode=True`` coexist with MXFP8
    without attempting bit-identity with vLLM. Megatron's BI ``general_gemm``
    patch (``_te_general_gemm_patched``) immediately reads
    ``A.is_cuda`` / ``B.is_cuda`` on every call, but TE's quantised storage
    types (``MXFP8TensorStorage`` / ``Float8TensorStorage``) do not expose
    ``is_cuda``, so any MXFP8 linear under BI mode raises
    ``AttributeError: 'MXFP8TensorStorage' object has no attribute 'is_cuda'``.
    This wrapper detects non-``torch.Tensor`` GEMM operands (TE quantised
    storages) and forwards the call to TE's original ``general_gemm``;
    regular CUDA tensors still go through the BI Triton matmul.
    No numerical matching with vLLM is attempted here — for bit-identical
    vLLM↔Megatron MXFP8 numerics, use
    :func:`install_mxfp8_dequant_for_bi_gemm` instead.

    Must be called AFTER ``enable_batch_invariant_mode()``.
    """
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.linear as te_linear_mod
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import megatron.core.extensions.transformer_engine as meg_te

    if bik_mod._TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError(
            "enable_batch_invariant_mode() must run before "
            "install_mxfp8_passthrough_for_bi_gemm()"
        )

    orig_gemm = bik_mod._TE_GENERAL_GEMM_ORIG
    bi_gemm = bik_mod._te_general_gemm_patched
    extract = bik_mod._extract_te_gemm_args

    def _wrapper(*args, **kwargs):
        a, b, _, _, _, _, _ = extract(args, kwargs)
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return orig_gemm(*args, **kwargs)
        return bi_gemm(*args, **kwargs)

    for mod, attr in (
        (te_cpp, "general_gemm"),
        (te_linear_mod, "general_gemm"),
        (te_layernorm_linear_mod, "general_gemm"),
        (meg_te, "general_gemm"),
    ):
        if hasattr(mod, attr):
            setattr(mod, attr, _wrapper)


def install_mxfp8_dequant_for_bi_gemm() -> None:
    """Route MXFP8 GEMMs through the BF16 BI matmul via dequant.

    Wraps Megatron's BI ``general_gemm`` hook so that any MXFP8 input is
    dequanted to bf16 via TE's ``.dequantize(dtype=bf16)``, then fed into
    ``BatchInvariantTEGemmFn`` (the BF16 BI matmul path). Mirrors vLLM's
    ``BatchInvariantMxfp8LinearKernel`` (which dequants the weight and
    quant-dequant-round-trips the activation, then calls
    ``matmul_persistent``).

    Must be called AFTER ``enable_batch_invariant_mode()``.
    """
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.linear as te_linear_mod
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import megatron.core.extensions.transformer_engine as meg_te

    if bik_mod._TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError(
            "enable_batch_invariant_mode() must run before "
            "install_mxfp8_dequant_for_bi_gemm()"
        )

    extract = bik_mod._extract_te_gemm_args
    bi_gemm_fn = bik_mod.BatchInvariantTEGemmFn

    def _maybe_dequant_to_bf16(t):
        if t is None or isinstance(t, torch.Tensor):
            return t
        if hasattr(t, "dequantize"):
            return t.dequantize(dtype=torch.bfloat16)
        return t

    def _wrapper(*args, **kwargs):
        a, b, out_dtype, layout, out_tensor, bias, grad = extract(args, kwargs)
        extra_output = kwargs.get("extra_output", None)
        ub = kwargs.get("ub", None)
        ub_type = kwargs.get("ub_type", None)
        bulk_overlap = kwargs.get("bulk_overlap", False)
        if (
            extra_output is not None
            or ub is not None
            or ub_type is not None
            or bulk_overlap
        ):
            raise RuntimeError(
                "Batch-invariant GEMM does not support Userbuffers/overlap "
                "(extra_output/ub/ub_type/bulk_overlap)."
            )

        a_bf16 = _maybe_dequant_to_bf16(a)
        b_bf16 = _maybe_dequant_to_bf16(b)

        result = bi_gemm_fn.apply(
            a_bf16, b_bf16, bias if not grad else None, out_dtype, layout
        )

        bias_grad = None
        if grad and bias is not None:
            b_flat = (
                b_bf16.reshape(-1, b_bf16.shape[-1])
                if b_bf16.dim() > 2
                else b_bf16
            )
            bias_grad = b_flat.sum(dim=0)

        if out_tensor is not None:
            out_tensor.copy_(result)
            return (out_tensor, bias_grad, None, extra_output)
        return (result, bias_grad, None, extra_output)

    for mod, attr in (
        (te_cpp, "general_gemm"),
        (te_linear_mod, "general_gemm"),
        (te_layernorm_linear_mod, "general_gemm"),
        (meg_te, "general_gemm"),
    ):
        if hasattr(mod, attr):
            setattr(mod, attr, _wrapper)


class _SplitNormLinear(torch.nn.Module):
    """Drop-in for TE's fused ``LayerNormColumnParallelLinear``.

    Running norm + linear as two ops forces the post-RMSNorm bf16 activation
    to be materialised between them. TE's fused kernel keeps that tensor in
    fp32 registers, while vLLM has a standalone RMSNorm whose bf16 output
    is re-read by ``qkv_proj`` / ``gate_up_proj``. Splitting the fused op
    eliminates that asymmetry.
    """

    def __init__(self, norm: torch.nn.Module, linear: torch.nn.Module):
        super().__init__()
        self.norm = norm
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def _unwrap_megatron_model(m):
    """Strip DDP / FSDP / Float16Module / ``language_model`` wrappers to expose ``GPTModel``."""
    from megatron.core.utils import unwrap_model

    m = unwrap_model(m)
    if hasattr(m, "language_model"):
        m = m.language_model
    return m


def split_all_layers_fused_layernorm_linear(model) -> int:
    """Unfuse TE's ``LayerNormColumnParallelLinear`` on every decoder layer.

    Replaces ``self_attention.linear_qkv`` and ``mlp.linear_fc1`` with a
    :class:`_SplitNormLinear` wrapper so the post-RMSNorm bf16 activation is
    materialised, matching vLLM's standalone-RMSNorm + linear path. Required
    on top of the BF16 (and MXFP8) kernel-matching patches to close the
    deeper-layer drift (see ``llama3_8b_numeric_mismatch.md`` and
    ``llama3_8b_mxfp8_numeric_mismatch.md``).

    Accepts either a single model or a list of model chunks. Returns the
    total number of decoder layers patched across chunks.
    """
    from megatron.core.extensions.transformer_engine import (
        split_te_layernorm_column_parallel_linear,
    )

    chunks = model if isinstance(model, list) else [model]
    total_layers = 0
    for chunk in chunks:
        inner = _unwrap_megatron_model(chunk)
        config = inner.config
        for layer in inner.decoder.layers:
            for owner, attr in (
                (layer.self_attention, "linear_qkv"),
                (layer.mlp, "linear_fc1"),
            ):
                fused = getattr(owner, attr)
                norm, linear = split_te_layernorm_column_parallel_linear(
                    fused, config
                )
                ref_w = fused.weight
                norm = norm.to(device=ref_w.device, dtype=ref_w.dtype)
                linear = linear.to(device=ref_w.device, dtype=ref_w.dtype)
                setattr(owner, attr, _SplitNormLinear(norm, linear))
            total_layers += 1
    return total_layers


def install_match_vllm_kernels() -> None:
    """Install the four BF16 kernel-matching patches for vLLM bit-identity.

    Installs ``install_vllm_style_{rmsnorm, rope, swiglu, sdpa}``. These
    cover the BF16 forward path (every linear, RoPE, SwiGLU, attention).

    Pre-conditions:
      - The RMSNorm patch only takes effect when batch-invariant mode is
        enabled (it replaces ``BatchInvariantRMSNormFn``, which the
        ``_te_rmsnorm_forward_patched`` shim looks up at call time).
        Therefore the caller must have set
        ``policy.megatron_cfg.batch_invariant_mode = true`` so
        ``enable_batch_invariant_mode()`` has already run.
      - Recommended call site: ``MegatronPolicyWorkerImpl.__init__``,
        immediately after ``setup_model_and_optimizer`` returns.

    For MXFP8 GEMM matching, separately call
    :func:`install_match_vllm_mxfp8_matmul` (and ensure the model is
    running the MXFP8 fp8 recipe).
    """
    install_vllm_style_rmsnorm()
    install_vllm_style_rmsnom_to_te()
    install_vllm_style_rope()
    install_vllm_style_swiglu()
    install_vllm_style_sdpa()


def install_match_vllm_mxfp8_matmul() -> None:
    """Install MXFP8-specific patches: compact scales + dequant-for-BI-GEMM.

    Installs ``install_mxfp8_compact_scales`` (property override forcing
    MXFP8 tensors to compact-scale layout) and
    ``install_mxfp8_dequant_for_bi_gemm`` (TE ``general_gemm`` hook that
    dequants both inputs to bf16 and routes through the BF16 BI matmul).

    Pre-conditions:
      - ``policy.megatron_cfg.batch_invariant_mode = true`` (the dequant
        hook is wired into Megatron's BI ``general_gemm`` patch).
      - The model is running the MXFP8 fp8 recipe
        (``policy.megatron_cfg.fp8_cfg.fp8_recipe == "mxfp8"``); otherwise
        the patches are no-ops because no MXFP8 tensors will reach the
        GEMM hook.
      - Typically used together with :func:`install_match_vllm_kernels`
        — without the BF16 patches the per-layer activations leading into
        the MXFP8 GEMMs already disagree across engines, so matching the
        GEMM alone is not enough.
    """
    install_mxfp8_compact_scales()
    install_mxfp8_dequant_for_bi_gemm()
