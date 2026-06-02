# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Batch-invariant MXFP8 linear kernel.

Dequantises the MXFP8 weight to bf16 **and** quant-dequant round-trips the
bf16 activation through MXFP8 (matching what Megatron-LM's
``fp8_autocast`` does to every linear's input), then routes the GEMM
through the BF16 batch-invariant matmul kernel (``matmul_persistent``)
from ``vllm.model_executor.layers.batch_invariant``.

The activation round-trip is the new piece (vs the earlier W8A16 version)
— it injects the same lossy bf16→fp8→bf16 step that TE's
``Linear.forward`` applies under ``fp8_autocast``, so the two engines'
GEMM inputs match bit-for-bit (the vLLM and TE MXFP8 quantisers produce
byte-identical ``(uint8 scale, fp8 data)`` tuples on Blackwell, verified
in ``my_script/compare_mxfp8_quant.py``).
"""

import torch
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.model_executor.layers.batch_invariant import matmul_persistent
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _quant_dequant_bf16_via_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """Lossy bf16 → MXFP8 → bf16 round-trip on activations.

    Mirrors TE's per-call activation quantisation under ``fp8_autocast``:
    quantise to (fp8 data, uint8 E8M0 scale) via vLLM's ``mxfp8_e4m3_quantize``
    (FlashInfer on Blackwell — byte-identical to TE's quantiser), then
    dequant back to bf16 via ``dequant_mxfp8_to_bf16``.
    """
    x_2d = x.reshape(-1, x.shape[-1])
    x_fp8, x_scale = mxfp8_e4m3_quantize(x_2d, is_sf_swizzled_layout=False)
    x_bf16 = dequant_mxfp8_to_bf16(x_fp8, x_scale)
    return x_bf16.reshape(x.shape)


class BatchInvariantMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A8 linear via dequant-to-bf16 + BF16 batch-invariant matmul.

    Both the weight and the activation are round-tripped through MXFP8
    before the GEMM. The GEMM itself runs in bf16 through the
    batch-invariant ``matmul_persistent`` kernel.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not envs.VLLM_BATCH_INVARIANT:
            return False, "only used under VLLM_BATCH_INVARIANT=1"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K] fp8_e4m3fn
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_scale = layer.weight_scale
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"BI MXFP8 backend requires {MXFP8_SCALE_DTYPE} "
                f"weight_scale dtype, got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"BI MXFP8 backend requires 2D weight_scale, "
                f"got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )

        # Dequant weight: [N, K] fp8 + [N, K/32] uint8 -> [N, K] bf16.
        weight_bf16 = dequant_mxfp8_to_bf16(layer.weight, weight_scale)

        # Quant-dequant round-trip on the activation (matches Megatron's
        # fp8_autocast per-call activation quantisation).
        x_bf16 = _quant_dequant_bf16_via_mxfp8(x)

        # Flatten x to 2D for the persistent kernel: [..., K] -> [M, K].
        leading_shape = x_bf16.shape[:-1]
        x_2d = x_bf16.reshape(-1, x_bf16.shape[-1])

        # BF16 BI matmul: [M, K] @ [K, N] -> [M, N]. weight is stored as
        # [N, K] so we transpose. matmul_persistent handles the strided view.
        out_2d = matmul_persistent(x_2d, weight_bf16.t())

        if bias is not None:
            out_2d = out_2d + bias

        return out_2d.reshape(*leading_shape, -1).to(x.dtype)
