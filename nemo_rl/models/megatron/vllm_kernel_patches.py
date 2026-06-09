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

"""Megatron-side true-on-policy runtime patches.

These monkey-patches target Blackwell BF16 and MXFP8 batch-invariant paths.

Originally developed under ``my_script/megatron_forward.py`` as part of the
cross-engine numerical-matching effort (see
``skills/debug-generation-training-mismatch/SKILL.md``). Ported here so an RL
run can install the same patches inside ``MegatronPolicyWorker.__init__``.

``policy.bf16_true_on_policy`` enables Megatron-Core batch-invariant mode before
this module is called. The remaining BF16 Megatron-side patch here is the SDPA
path, which still routes TE attention through vLLM's FA2 wrapper. RMSNorm, RoPE,
and SwiGLU parity are handled in the opposite direction by patching vLLM to
match Megatron.

MXFP8-specific patches are gated by ``policy.mxfp8_matmul_batch_invariant`` and
``NEMO_RL_MXFP8_MATMUL_BI_BACKEND``.
"""

from __future__ import annotations

import math

import torch

from nemo_rl.models.true_on_policy import get_mxfp8_matmul_bi_backend

G_VLLM_STYLE_SDPA_SEQ_LENS: list[int] | None = None


def set_vllm_style_sdpa_sequence_lengths(seq_lens: torch.Tensor | None) -> None:
    """Set per-forward sequence lengths for the vLLM-style SDPA patch.

    Megatron/TE sees padded tensors in both unpacked ``sbhd``/``bshd`` and
    packed ``thd`` layouts. vLLM's decoder prefill path operates on actual
    request lengths. The worker calls this before each forward when
    ``input_lengths`` is present so the attention patch can pack only real
    tokens and ignore pad tokens.
    """
    global G_VLLM_STYLE_SDPA_SEQ_LENS
    if seq_lens is None:
        G_VLLM_STYLE_SDPA_SEQ_LENS = None
        return
    G_VLLM_STYLE_SDPA_SEQ_LENS = [int(item) for item in seq_lens.detach().cpu()]


def install_vllm_style_sdpa(
    *, paged_kv: bool = True, paged_block_size: int = 16
) -> None:
    """Route TE's ``DotProductAttention`` through vLLM's FA2 kernel.

    On Blackwell, vLLM under ``VLLM_BATCH_INVARIANT=1`` rejects FA4 (per
    ``vllm/v1/attention/backends/fa_utils.py:137-142``) and falls back to
    FA2 with ``num_splits=1``. Megatron/TE on Blackwell does not have an
    FA4 path either, but defaults to a different kernel (cuDNN-fused
    attention) that doesn't match FA2 byte-for-byte. This patch makes
    Megatron call the same vLLM FA2 wrapper with ``num_splits=1,
    fa_version=2``.

    In no-grad forwards, ``paged_kv=True`` mirrors vLLM decoder prefill more
    closely by writing K/V through ``reshape_and_cache_flash`` and reading them
    with ``key_cache`` / ``value_cache`` plus ``block_table`` and ``seqused_k``.
    When gradients are enabled, the wrapper falls back to direct packed FA2 so
    training keeps a differentiable attention path.

    Sliding-window / bias / fp8_output are not supported in this wrapper.
    """
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as dpa_mod,
    )
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        reshape_and_cache_flash,
    )

    def _global_seq_lens(batch_size: int, max_seqlen: int) -> list[int] | None:
        if G_VLLM_STYLE_SDPA_SEQ_LENS is None:
            return None
        if len(G_VLLM_STYLE_SDPA_SEQ_LENS) != batch_size:
            raise RuntimeError(
                "vLLM-style SDPA sequence length batch mismatch: "
                f"{len(G_VLLM_STYLE_SDPA_SEQ_LENS)} vs {batch_size}"
            )
        if max(G_VLLM_STYLE_SDPA_SEQ_LENS) > max_seqlen:
            raise RuntimeError(
                "vLLM-style SDPA sequence length exceeds tensor sequence dim: "
                f"{max(G_VLLM_STYLE_SDPA_SEQ_LENS)} > {max_seqlen}"
            )
        return G_VLLM_STYLE_SDPA_SEQ_LENS

    def _cu_from_seq_lens(seq_lens: list[int], device: torch.device) -> torch.Tensor:
        lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(lens, dim=0, dtype=torch.int32),
            ]
        )

    def _seq_lens_from_cu(cu_seqlens: torch.Tensor) -> list[int]:
        diffs = cu_seqlens[1:] - cu_seqlens[:-1]
        return [int(item) for item in diffs.detach().cpu()]

    def _pack_bshd(tensor: torch.Tensor, seq_lens: list[int]) -> torch.Tensor:
        return torch.cat(
            [tensor[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(seq_lens)],
            dim=0,
        ).contiguous()

    def _scatter_bshd(
        packed: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        seq_lens: list[int],
    ) -> torch.Tensor:
        out = packed.new_zeros((batch_size, seq_len, packed.size(1), packed.size(2)))
        offset = 0
        for batch_idx, cur_seq_len in enumerate(seq_lens):
            next_offset = offset + cur_seq_len
            out[batch_idx, :cur_seq_len] = packed[offset:next_offset]
            offset = next_offset
        return out

    def _pack_thd(
        tensor: torch.Tensor,
        *,
        seq_lens: list[int],
        padded_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        chunks = []
        for seq_idx, seq_len in enumerate(seq_lens):
            start = int(padded_cu_seqlens[seq_idx].item())
            chunks.append(tensor[start : start + seq_len])
        return torch.cat(chunks, dim=0).contiguous()

    def _scatter_thd(
        packed: torch.Tensor,
        *,
        total_tokens: int,
        seq_lens: list[int],
        padded_cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        out = packed.new_zeros((total_tokens, packed.size(1), packed.size(2)))
        offset = 0
        for seq_idx, seq_len in enumerate(seq_lens):
            next_offset = offset + seq_len
            start = int(padded_cu_seqlens[seq_idx].item())
            out[start : start + seq_len] = packed[offset:next_offset]
            offset = next_offset
        return out

    def _block_metadata(
        seq_lens: list[int],
        *,
        block_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        max_blocks = math.ceil(max(seq_lens) / block_size)
        block_table = torch.zeros(
            (len(seq_lens), max_blocks), dtype=torch.int32, device=device
        )
        slot_mapping = []
        next_block = 0
        for batch_idx, seq_len in enumerate(seq_lens):
            num_blocks = math.ceil(seq_len / block_size)
            blocks = torch.arange(
                next_block,
                next_block + num_blocks,
                dtype=torch.int32,
                device=device,
            )
            block_table[batch_idx, :num_blocks] = blocks
            for pos in range(seq_len):
                slot_mapping.append(
                    (next_block + pos // block_size) * block_size + pos % block_size
                )
            next_block += num_blocks
        return (
            block_table,
            torch.tensor(slot_mapping, dtype=torch.int64, device=device),
            next_block,
        )

    def _direct_fa2(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_lens: list[int],
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        cu_seqlens = _cu_from_seq_lens(seq_lens, q.device)
        out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=max(seq_lens),
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max(seq_lens),
            cu_seqlens_k=cu_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            num_splits=1,
            fa_version=2,
            deterministic=False,
        )
        if isinstance(out, tuple):
            return out[0]
        return out

    def _paged_fa2(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_lens: list[int],
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        block_table, slot_mapping, num_blocks = _block_metadata(
            seq_lens,
            block_size=paged_block_size,
            device=q.device,
        )
        key_cache = torch.empty(
            (num_blocks, paged_block_size, k.size(1), k.size(2)),
            dtype=k.dtype,
            device=k.device,
        )
        value_cache = torch.empty_like(key_cache)
        scale = torch.ones((), dtype=torch.float32, device=q.device)
        reshape_and_cache_flash(
            k,
            v,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            scale,
            scale,
        )
        out = torch.empty_like(q)
        result = flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=out,
            max_seqlen_q=max(seq_lens),
            cu_seqlens_q=_cu_from_seq_lens(seq_lens, q.device),
            max_seqlen_k=max(seq_lens),
            seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=q.device),
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            num_splits=1,
            fa_version=2,
        )
        if isinstance(result, tuple):
            return result[0]
        return out

    def _run_fa2(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        seq_lens: list[int],
        softmax_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        if paged_kv and not torch.is_grad_enabled():
            return _paged_fa2(
                q,
                k,
                v,
                seq_lens=seq_lens,
                softmax_scale=softmax_scale,
                causal=causal,
            )
        return _direct_fa2(
            q,
            k,
            v,
            seq_lens=seq_lens,
            softmax_scale=softmax_scale,
            causal=causal,
        )

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
        assert inference_params is None, (
            "kv-cache inference not supported in vllm-fa2 patch"
        )
        assert fp8_output is False, "fp8 output not supported in vllm-fa2 patch"

        fmt = qkv_format or getattr(self, "qkv_format", "sbhd")
        if fmt == "sbhd":
            s_q, b_q, n_q, d = query_layer.shape
            s_kv, b_kv, n_kv, _ = key_layer.shape
            if b_q != b_kv:
                raise NotImplementedError("cross-attention is not supported")
            q_bshd = query_layer.transpose(0, 1).contiguous()
            k_bshd = key_layer.transpose(0, 1).contiguous()
            v_bshd = value_layer.transpose(0, 1).contiguous()
            seq_lens = _global_seq_lens(b_q, s_q) or [s_q for _ in range(b_q)]
            q = _pack_bshd(q_bshd, seq_lens)
            k = _pack_bshd(k_bshd, seq_lens)
            v = _pack_bshd(v_bshd, seq_lens)
            out = _run_fa2(
                q,
                k,
                v,
                seq_lens=seq_lens,
                softmax_scale=1.0 / math.sqrt(d),
                causal=(attn_mask_type or "causal").startswith("causal"),
            )
            out = _scatter_bshd(
                out,
                batch_size=b_q,
                seq_len=s_q,
                seq_lens=seq_lens,
            ).transpose(0, 1)
            return out.reshape(s_q, b_q, n_q * d).contiguous()
        elif fmt == "bshd":
            b_q, s_q, n_q, d = query_layer.shape
            b_kv, s_kv, n_kv, _ = key_layer.shape
            if b_q != b_kv:
                raise NotImplementedError("cross-attention is not supported")
            seq_lens = _global_seq_lens(b_q, s_q) or [s_q for _ in range(b_q)]
            q = _pack_bshd(query_layer.contiguous(), seq_lens)
            k = _pack_bshd(key_layer.contiguous(), seq_lens)
            v = _pack_bshd(value_layer.contiguous(), seq_lens)
            out = _run_fa2(
                q,
                k,
                v,
                seq_lens=seq_lens,
                softmax_scale=1.0 / math.sqrt(d),
                causal=(attn_mask_type or "causal").startswith("causal"),
            )
            out = _scatter_bshd(
                out,
                batch_size=b_q,
                seq_len=s_q,
                seq_lens=seq_lens,
            )
            return out.reshape(b_q, s_q, n_q * d).contiguous()
        elif fmt == "thd":
            if query_layer.dim() != 3:
                raise RuntimeError(
                    "qkv_format='thd' expected query/key/value tensors with "
                    f"rank 3, got {tuple(query_layer.shape)}"
                )
            total_tokens, n_q, d = query_layer.shape
            if cu_seqlens_q is None:
                raise RuntimeError("qkv_format='thd' requires cu_seqlens_q")
            padded_cu = (
                cu_seqlens_q_padded if cu_seqlens_q_padded is not None else cu_seqlens_q
            )
            seq_lens = _global_seq_lens(len(cu_seqlens_q) - 1, total_tokens)
            if seq_lens is None:
                seq_lens = _seq_lens_from_cu(cu_seqlens_q)
            q = _pack_thd(
                query_layer,
                seq_lens=seq_lens,
                padded_cu_seqlens=padded_cu,
            )
            k = _pack_thd(
                key_layer,
                seq_lens=seq_lens,
                padded_cu_seqlens=padded_cu,
            )
            v = _pack_thd(
                value_layer,
                seq_lens=seq_lens,
                padded_cu_seqlens=padded_cu,
            )
            out = _run_fa2(
                q,
                k,
                v,
                seq_lens=seq_lens,
                softmax_scale=1.0 / math.sqrt(d),
                causal=(attn_mask_type or "causal").endswith("causal"),
            )
            return _scatter_thd(
                out,
                total_tokens=total_tokens,
                seq_lens=seq_lens,
                padded_cu_seqlens=padded_cu,
            )
        else:
            raise NotImplementedError(f"qkv_format={fmt!r} not supported")

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
    import megatron.core.extensions.transformer_engine as meg_te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import transformer_engine.pytorch.module.linear as te_linear_mod
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )

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
    import megatron.core.extensions.transformer_engine as meg_te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import transformer_engine.pytorch.module.linear as te_linear_mod
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )

    if bik_mod._TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError(
            "enable_batch_invariant_mode() must run before "
            "install_mxfp8_dequant_for_bi_gemm()"
        )

    extract = bik_mod._extract_te_gemm_args
    bi_gemm_fn = bik_mod.BatchInvariantTEGemmFn
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        dequant_mxfp8_to_bf16,
        mxfp8_e4m3_quantize,
    )

    def _maybe_dequant_to_bf16(
        t,
        *,
        quantize_plain_tensor: bool,
    ):
        if t is None or isinstance(t, torch.Tensor):
            if isinstance(t, torch.Tensor):
                if (
                    quantize_plain_tensor
                    and t.dtype == torch.bfloat16
                    and t.shape[-1] % MXFP8_BLOCK_SIZE == 0
                ):
                    q_tensor, q_scales = mxfp8_e4m3_quantize(
                        t.contiguous(),
                        is_sf_swizzled_layout=False,
                    )
                    return dequant_mxfp8_to_bf16(q_tensor, q_scales)
            return t
        data = getattr(t, "_rowwise_data", None)
        scale_inv = getattr(t, "_rowwise_scale_inv", None)
        if data is not None and scale_inv is not None:
            if getattr(t, "_with_gemm_swizzled_scales", False):
                raise RuntimeError(
                    "vLLM-style MXFP8 dequant requires compact rowwise scales."
                )
            if data.dtype == torch.uint8:
                data = data.view(torch.float8_e4m3fn)
            leading_dim = math.prod(data.shape[:-1])
            scale_cols = data.shape[-1] // MXFP8_BLOCK_SIZE
            scale_inv = scale_inv[:leading_dim, :scale_cols].contiguous()
            scale_inv = scale_inv.view(*data.shape[:-1], scale_cols)
            return dequant_mxfp8_to_bf16(data, scale_inv)
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

        # vLLM's MXFP8 linear quantizes plain BF16 activations before GEMM and
        # dequants both operands before the BI matmul patch. TE sometimes sends
        # BF16 tensors through this hook under the MXFP8 recipe, so mirror that
        # quantize-dequant round trip for inference/logprob forwards.
        quantize_plain_tensor = not torch.is_grad_enabled()
        a_bf16 = _maybe_dequant_to_bf16(
            a,
            quantize_plain_tensor=quantize_plain_tensor,
        )
        b_bf16 = _maybe_dequant_to_bf16(
            b,
            quantize_plain_tensor=quantize_plain_tensor,
        )

        result = bi_gemm_fn.apply(
            a_bf16, b_bf16, bias if not grad else None, out_dtype, layout
        )

        bias_grad = None
        if grad and bias is not None:
            b_flat = (
                b_bf16.reshape(-1, b_bf16.shape[-1]) if b_bf16.dim() > 2 else b_bf16
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


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _is_mxfp8_operand(t) -> bool:
    return (
        not isinstance(t, torch.Tensor)
        and getattr(t, "_rowwise_data", None) is not None
        and getattr(t, "_rowwise_scale_inv", None) is not None
    )


def _swizzle_mxfp8_scale(scale_2d: torch.Tensor, *, m: int, k: int) -> torch.Tensor:
    scale_cols = k // 32
    num_m_tiles = _ceil_div(m, 128)
    num_k_tiles = _ceil_div(scale_cols, 4)
    padded = torch.zeros(
        (num_m_tiles * 128, num_k_tiles * 4),
        dtype=scale_2d.dtype,
        device=scale_2d.device,
    )
    padded[:m, :scale_cols] = scale_2d[:m, :scale_cols]
    return (
        padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def _as_swizzled_mxfp8_scale(
    scale: torch.Tensor,
    *,
    m: int,
    k: int,
    is_swizzled: bool,
) -> torch.Tensor:
    scale_cols = k // 32
    swizzled_numel = _ceil_div(m, 128) * _ceil_div(scale_cols, 4) * 512
    if is_swizzled:
        if scale.numel() < swizzled_numel:
            raise RuntimeError(
                "MXFP8 swizzled scale is too small for native BI matmul: "
                f"got {scale.numel()}, need {swizzled_numel}."
            )
        return scale.flatten()[:swizzled_numel].contiguous()
    if scale.dim() == 1:
        scale = scale[: m * scale_cols].view(m, scale_cols)
    return _swizzle_mxfp8_scale(scale, m=m, k=k)


def _mxfp8_operand_from_te_storage(t) -> tuple[torch.Tensor, torch.Tensor]:
    data = getattr(t, "_rowwise_data")
    scale = getattr(t, "_rowwise_scale_inv")
    if data.dtype == torch.uint8:
        data = data.view(torch.float8_e4m3fn)
    if data.dim() > 2:
        data = data.reshape(-1, data.shape[-1])
    if data.dim() != 2:
        raise RuntimeError(
            f"MXFP8 native BI matmul requires 2D data, got {data.shape}."
        )
    scale = _as_swizzled_mxfp8_scale(
        scale,
        m=data.shape[0],
        k=data.shape[1],
        is_swizzled=bool(getattr(t, "_with_gemm_swizzled_scales", False)),
    )
    return data.contiguous(), scale


def _mxfp8_operand_from_plain_tensor(
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

    if t.dim() > 2:
        t = t.reshape(-1, t.shape[-1])
    if t.dim() != 2:
        raise RuntimeError(f"MXFP8 native BI matmul requires 2D data, got {t.shape}.")
    return mxfp8_quantize(t.contiguous())


def install_mxfp8_native_for_bi_gemm() -> None:
    """Route MXFP8 forward GEMMs through the native MXFP8 BI matmul."""
    import megatron.core.extensions.transformer_engine as meg_te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear_mod
    import transformer_engine.pytorch.module.linear as te_linear_mod
    from megatron.core.transformer.custom_layers import (
        batch_invariant_kernels as bik_mod,
    )

    if bik_mod._TE_GENERAL_GEMM_ORIG is None:
        raise RuntimeError(
            "enable_batch_invariant_mode() must run before "
            "install_mxfp8_native_for_bi_gemm()"
        )

    orig_gemm = bik_mod._TE_GENERAL_GEMM_ORIG
    extract = bik_mod._extract_te_gemm_args
    native_mxfp8_matmul = bik_mod.mxfp8_matmul_persistent
    bf16_bi_gemm = bik_mod._te_general_gemm_patched

    def _wrapper(*args, **kwargs):
        a, b, out_dtype, layout, out_tensor, bias, grad = extract(args, kwargs)
        if not _is_mxfp8_operand(a) and not _is_mxfp8_operand(b):
            return bf16_bi_gemm(*args, **kwargs)
        if grad or layout.upper() != "TN":
            return orig_gemm(*args, **kwargs)

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
                "Native MXFP8 batch-invariant GEMM does not support "
                "Userbuffers/overlap (extra_output/ub/ub_type/bulk_overlap)."
            )

        if torch.is_grad_enabled():
            return orig_gemm(*args, **kwargs)

        if _is_mxfp8_operand(a):
            weight_mxfp8 = _mxfp8_operand_from_te_storage(a)
        else:
            weight_mxfp8 = _mxfp8_operand_from_plain_tensor(a)

        if _is_mxfp8_operand(b):
            leading_shape = getattr(b, "_rowwise_data").shape[:-1]
            activation_mxfp8 = _mxfp8_operand_from_te_storage(b)
        else:
            leading_shape = b.shape[:-1]
            activation_mxfp8 = _mxfp8_operand_from_plain_tensor(b)

        result_2d = native_mxfp8_matmul(
            activation_mxfp8,
            weight_mxfp8,
            bias=bias,
            output_dtype=out_dtype or torch.bfloat16,
        )
        result = result_2d.reshape(*leading_shape, result_2d.shape[-1])

        if out_tensor is not None:
            out_tensor.copy_(result)
            return (out_tensor, None, None, extra_output)
        return (result, None, None, extra_output)

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
                norm, linear = split_te_layernorm_column_parallel_linear(fused, config)
                ref_w = fused.weight
                norm = norm.to(device=ref_w.device, dtype=ref_w.dtype)
                linear = linear.to(device=ref_w.device, dtype=ref_w.dtype)
                setattr(owner, attr, _SplitNormLinear(norm, linear))
            total_layers += 1
    return total_layers


def install_megatron_true_on_policy_patches() -> None:
    """Install Megatron-side BF16 true-on-policy patches.

    Megatron keeps its own BI RMSNorm, RoPE, and SwiGLU implementations. The
    corresponding vLLM operations are patched on the generation side to match
    Megatron. The remaining Megatron-side BF16 patch is attention, which routes
    TE ``DotProductAttention`` through vLLM's FA2 wrapper.

    Pre-conditions:
      - The caller must have set ``policy.bf16_true_on_policy=true`` so
        Megatron's ``enable_batch_invariant_mode()`` has already run.
      - Recommended call site: ``MegatronPolicyWorkerImpl.__init__``,
        immediately after ``setup_model_and_optimizer`` returns.

    For MXFP8 GEMM matching, separately call
    :func:`install_bi_mxfp8_matmul_qdq` or
    :func:`install_bi_mxfp8_matmul` (and ensure the model is running the
    MXFP8 fp8 recipe).
    """
    install_vllm_style_sdpa()


def install_bi_mxfp8_matmul_qdq() -> None:
    """Install MXFP8-specific QDQ patches: compact scales + dequant-for-BI-GEMM.

    Installs ``install_mxfp8_compact_scales`` (property override forcing
    MXFP8 tensors to compact-scale layout) and
    ``install_mxfp8_dequant_for_bi_gemm`` (TE ``general_gemm`` hook that
    dequants both inputs to bf16 and routes through the BF16 BI matmul).

    Pre-conditions:
      - ``policy.bf16_true_on_policy=true`` (the dequant
        hook is wired into Megatron's BI ``general_gemm`` patch).
      - The model is running the MXFP8 fp8 recipe
        (``policy.megatron_cfg.fp8_cfg.fp8_recipe == "mxfp8"``); otherwise
        the patches are no-ops because no MXFP8 tensors will reach the
        GEMM hook.
      - Typically used together with
        :func:`install_megatron_true_on_policy_patches`
        — without the BF16 patches the per-layer activations leading into
        the MXFP8 GEMMs already disagree across engines, so matching the
        GEMM alone is not enough.
    """
    install_mxfp8_compact_scales()
    install_mxfp8_dequant_for_bi_gemm()


def install_bi_mxfp8_matmul() -> None:
    """Install native MXFP8 batch-invariant GEMM patches."""
    install_mxfp8_native_for_bi_gemm()


def install_true_on_policy_patches(
    *,
    bf16_true_on_policy: bool,
    mxfp8_matmul_batch_invariant: bool,
    mxfp8_active: bool,
) -> dict[str, str]:
    """Install Megatron true-on-policy patches controlled by policy flags."""
    installed: dict[str, str] = {}

    if mxfp8_matmul_batch_invariant and not bf16_true_on_policy:
        raise ValueError(
            "policy.mxfp8_matmul_batch_invariant=True requires "
            "policy.bf16_true_on_policy=True because that flag enables "
            "Megatron batch-invariant mode and the BF16 true-on-policy patches."
        )

    if bf16_true_on_policy:
        install_megatron_true_on_policy_patches()
        installed["bf16_true_on_policy"] = "megatron_true_on_policy_patches"

    if mxfp8_matmul_batch_invariant:
        if not mxfp8_active:
            raise ValueError(
                "policy.mxfp8_matmul_batch_invariant=True requires "
                "policy.megatron_cfg.fp8_cfg.enabled=True with "
                'fp8_cfg.fp8_recipe="mxfp8".'
            )

        backend = get_mxfp8_matmul_bi_backend()
        if backend == "qdq":
            install_bi_mxfp8_matmul_qdq()
        else:
            install_bi_mxfp8_matmul()
        installed["mxfp8_matmul_batch_invariant"] = backend
    elif mxfp8_active and bf16_true_on_policy:
        # Megatron's BI general_gemm patch reads `A.is_cuda` on every call,
        # but TE's MXFP8TensorStorage does not expose `is_cuda`, so any
        # MXFP8 linear under BI mode raises AttributeError. Route MXFP8
        # GEMMs to TE's original general_gemm; BF16 GEMMs still hit BI.
        install_mxfp8_passthrough_for_bi_gemm()
        installed["mxfp8_matmul_batch_invariant"] = "passthrough"

    return installed
