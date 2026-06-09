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

"""Native MXFP8 batch-invariant matmul runtime kernel.

This lives in NeMo-RL rather than Megatron-Core so true-on-policy runtime
patches can install the native MXFP8 path without modifying third-party source.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

    def _identity_jit(*args, **kwargs):  # noqa: ARG001
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

    triton = SimpleNamespace(jit=_identity_jit, cdiv=lambda a, b: (a + b - 1) // b)
    tl = SimpleNamespace(constexpr=object)


def _matmul_launch_metadata(
    grid: Callable[..., Any],
    kernel: Any,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Build launch metadata for Triton matmul kernels used by the profiler."""
    del grid
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={m}, N={n}, K={k}, "
            f"tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _mxfp8_scale_offsets(rows, cols, n_col_blocks):
    """Map logical MXFP8 scale coordinates to cuBLAS swizzled scale offsets."""
    macro_row_block = rows // 128
    macro_col_block = cols // 4
    local_row = rows % 128
    local_col = cols % 4
    group = local_row // 32
    sub_row = local_row % 32
    tile_id = macro_row_block * n_col_blocks + macro_col_block
    return tile_id * 512 + sub_row * 16 + group * 4 + local_col


@triton.jit(launch_metadata=_matmul_launch_metadata)
def _mxfp8_matmul_kernel_persistent(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    a_scale_n_col_blocks,
    b_scale_n_col_blocks,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Persistent MXFP8 block-scaled matmul kernel."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_m_base = tl.arange(0, BLOCK_SIZE_M)
    offs_n_base = tl.arange(0, BLOCK_SIZE_N)
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)
    offs_scale_k_base = tl.arange(0, BLOCK_SIZE_K // 32)

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id,
            num_pid_in_group,
            num_pid_m,
            GROUP_SIZE_M,
            NUM_SMS,
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N

        offs_am = start_m + offs_m_base
        offs_bn = start_n + offs_n_base
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am_safe = tl.where(offs_am < M, offs_am, 0)
        offs_bn_safe = tl.where(offs_bn < N, offs_bn, 0)
        offs_am_safe = tl.max_contiguous(
            tl.multiple_of(offs_am_safe, BLOCK_SIZE_M),
            BLOCK_SIZE_M,
        )
        offs_bn_safe = tl.max_contiguous(
            tl.multiple_of(offs_bn_safe, BLOCK_SIZE_N),
            BLOCK_SIZE_N,
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + offs_k_base.to(tl.int64)
                offs_scale_k = ki * (BLOCK_SIZE_K // 32) + offs_scale_k_base.to(
                    tl.int64
                )
            else:
                offs_k = ki * BLOCK_SIZE_K + offs_k_base
                offs_scale_k = ki * (BLOCK_SIZE_K // 32) + offs_scale_k_base

            k_mask = offs_k < K
            scale_k_mask = offs_scale_k < K // 32

            a_ptrs = a_ptr + (
                offs_am_safe[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_bn_safe[:, None] * stride_bn + offs_k[None, :] * stride_bk
            )
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_bn[:, None] < N) & k_mask[None, :],
                other=0.0,
            )

            a_scale_offsets = _mxfp8_scale_offsets(
                offs_am_safe[:, None],
                offs_scale_k[None, :],
                a_scale_n_col_blocks,
            )
            b_scale_offsets = _mxfp8_scale_offsets(
                offs_bn_safe[:, None],
                offs_scale_k[None, :],
                b_scale_n_col_blocks,
            )
            a_scale = tl.load(
                a_scale_ptr + a_scale_offsets,
                mask=(offs_am[:, None] < M) & scale_k_mask[None, :],
                other=0,
            )
            b_scale = tl.load(
                b_scale_ptr + b_scale_offsets,
                mask=(offs_bn[:, None] < N) & scale_k_mask[None, :],
                other=0,
            )
            accumulator = tl.dot_scaled(
                a,
                a_scale,
                "e4m3",
                b.T,
                b_scale,
                "e4m3",
                accumulator,
            )

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c,
            num_pid_in_group,
            num_pid_m,
            GROUP_SIZE_M,
            NUM_SMS,
        )
        offs_cm = pid_m * BLOCK_SIZE_M + offs_m_base
        offs_cn = pid_n * BLOCK_SIZE_N + offs_n_base
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_compute_units(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    if device.type == "xpu":
        return torch.xpu.get_device_properties(device).max_compute_units
    return torch.get_num_threads()


def _unpack_mxfp8_operand(x: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(x, tuple) and len(x) == 2:
        data, scale = x
    elif hasattr(x, "data") and hasattr(x, "scale"):
        data = x.data
        scale = x.scale
    else:
        raise TypeError(
            f"{name} must be an MXFP8Tensor-like object or a (data, scale) tuple, "
            f"got {type(x).__name__}"
        )
    if not isinstance(data, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name}.data and {name}.scale must be torch.Tensor instances")
    return data, scale


def _is_mxfp8_scale_dtype(dtype: torch.dtype) -> bool:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return dtype == torch.uint8 or (e8m0_dtype is not None and dtype == e8m0_dtype)


def mxfp8_matmul_persistent(
    a: Any,
    b: Any,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Persistent batch-invariant MXFP8 matmul.

    Args:
        a: MXFP8Tensor-like operand for A with data shaped [M, K].
        b: MXFP8Tensor-like operand for B with data shaped [N, K].
        bias: Optional 1D bias added along N.
        output_dtype: Output dtype. Defaults to BF16.

    Returns:
        Tensor shaped [M, N] containing ``A @ B.T``.
    """
    if not HAVE_TRITON:
        raise RuntimeError("mxfp8_matmul_persistent requires Triton")
    if not hasattr(tl, "dot_scaled"):
        raise RuntimeError(
            "mxfp8_matmul_persistent requires Triton tl.dot_scaled support"
        )

    a_data, a_scale = _unpack_mxfp8_operand(a, "a")
    b_data, b_scale = _unpack_mxfp8_operand(b, "b")

    if not (a_data.is_cuda and b_data.is_cuda):
        raise RuntimeError("MXFP8 matmul requires CUDA tensors")
    if not (a_scale.is_cuda and b_scale.is_cuda):
        raise RuntimeError("MXFP8 scales must be CUDA tensors")
    if a_data.dim() != 2 or b_data.dim() != 2:
        raise RuntimeError("MXFP8 data tensors must be 2D")
    if a_data.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"Expected a.data FP8 E4M3, got {a_data.dtype}")
    if b_data.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"Expected b.data FP8 E4M3, got {b_data.dtype}")
    if not _is_mxfp8_scale_dtype(a_scale.dtype):
        raise RuntimeError(f"Expected a.scale E8M0/uint8, got {a_scale.dtype}")
    if not _is_mxfp8_scale_dtype(b_scale.dtype):
        raise RuntimeError(f"Expected b.scale E8M0/uint8, got {b_scale.dtype}")
    if a_data.shape[1] != b_data.shape[1]:
        raise RuntimeError("Incompatible K dimensions")
    if bias is not None and bias.dim() != 1:
        raise RuntimeError("MXFP8 matmul only supports 1D bias")

    major, _minor = torch.cuda.get_device_capability(a_data.device)
    if major < 10:
        raise RuntimeError("MXFP8 block-scaled matmul requires NVIDIA Blackwell")

    m, k = a_data.shape
    n = b_data.shape[0]
    if k % 32 != 0:
        raise RuntimeError(f"K ({k}) must be divisible by the MXFP8 block size 32")
    if bias is not None and bias.numel() != n:
        raise RuntimeError("Bias length must match N")

    num_sms = _get_compute_units(a_data.device)
    c = torch.empty((m, n), device=a_data.device, dtype=output_dtype)

    a_scale_n_col_blocks = _ceil_div(k // 32, 4)
    b_scale_n_col_blocks = _ceil_div(k // 32, 4)

    def grid(meta):
        blocks_m = triton.cdiv(m, meta["BLOCK_SIZE_M"])
        blocks_n = triton.cdiv(n, meta["BLOCK_SIZE_N"])
        return (min(num_sms, blocks_m * blocks_n),)

    configs = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "num_stages": 4,
        "num_warps": 8,
    }
    _mxfp8_matmul_kernel_persistent[grid](
        a_data,
        a_scale.contiguous().flatten().view(torch.uint8),
        b_data,
        b_scale.contiguous().flatten().view(torch.uint8),
        c,
        bias,
        m,
        n,
        k,
        a_scale_n_col_blocks,
        b_scale_n_col_blocks,
        a_data.stride(0),
        a_data.stride(1),
        b_data.stride(0),
        b_data.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        A_LARGE=a_data.numel() > 2**31,
        B_LARGE=b_data.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **configs,
    )
    return c
