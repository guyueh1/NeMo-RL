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

import torch

MXFP4_BLOCK_SIZE = 32
MXFP4_E2M1_MAX = 6.0
_MXFP4_BLOCKS_PER_CHUNK = 262_144


def quantize_dequantize_mxfp4(weight: torch.Tensor) -> torch.Tensor:
    """Round a logical weight through block-scaled MXFP4 and dequantize it.

    MXFP4 uses blocks of 32 E2M1 values with one power-of-two E8M0 scale per
    block. The returned tensor has the input shape and dtype, so it can be fed
    to an existing MXFP8 quantizer without changing its runtime representation.
    """
    if not weight.is_floating_point():
        raise TypeError(
            f"MXFP4 quantize/dequantize requires a float tensor, got {weight.dtype}"
        )
    if weight.shape[-1] % MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "MXFP4 quantize/dequantize requires the last dimension to be divisible "
            f"by {MXFP4_BLOCK_SIZE}, got shape {tuple(weight.shape)}."
        )

    original_shape = weight.shape
    blocks = weight.reshape(-1, MXFP4_BLOCK_SIZE)
    dequantized = torch.empty_like(blocks)
    # Midpoints between the non-negative E2M1 values
    # {0, .5, 1, 1.5, 2, 3, 4, 6}. bucketize gives nearest-value rounding.
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=weight.device,
    )
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=weight.device,
    )
    # Routed expert tensors are large enough that materializing a full FP32 copy
    # can consume multiple GiB. Process complete MX blocks in bounded chunks;
    # the only model-sized allocation is the dequantized output itself.
    for start in range(0, blocks.shape[0], _MXFP4_BLOCKS_PER_CHUNK):
        stop = min(start + _MXFP4_BLOCKS_PER_CHUNK, blocks.shape[0])
        float_blocks = blocks[start:stop].to(torch.float32)
        amax = float_blocks.abs().amax(dim=-1, keepdim=True)
        finite_amax = torch.where(torch.isfinite(amax), amax, torch.zeros_like(amax))
        exponent = torch.ceil(torch.log2(finite_amax / MXFP4_E2M1_MAX))
        exponent = torch.where(
            finite_amax == 0, torch.zeros_like(exponent), exponent
        ).clamp(-127, 127)
        scale = torch.exp2(exponent)
        normalized = (float_blocks / scale).abs().clamp(max=MXFP4_E2M1_MAX)
        quantized = levels[torch.bucketize(normalized, boundaries)]
        dequantized[start:stop].copy_(torch.copysign(quantized, float_blocks) * scale)

    return dequantized.reshape(original_shape)
