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

"""Shared controls for true-on-policy numeric matching."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Literal

MXFP8MatmulBIBackend = Literal["cublas", "native", "qdq"]

G_MXFP8_MATMUL_BI_BACKEND_ENV = "NEMO_RL_MXFP8_MATMUL_BI_BACKEND"
G_DEFAULT_MXFP8_MATMUL_BI_BACKEND: MXFP8MatmulBIBackend = "qdq"
G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV = "NEMO_RL_TE_CUBLAS_WORKSPACE_SIZE_BYTES"
G_TE_SITE_PACKAGES_ENV = "NEMO_RL_TE_SITE_PACKAGES"


def get_mxfp8_matmul_bi_backend() -> MXFP8MatmulBIBackend:
    """Return the selected MXFP8 batch-invariant matmul backend.

    The policy config only controls whether MXFP8 BI matmul is enabled. The
    implementation backend is an environment-level operational choice.
    """
    raw_backend = os.environ.get(
        G_MXFP8_MATMUL_BI_BACKEND_ENV,
        G_DEFAULT_MXFP8_MATMUL_BI_BACKEND,
    )
    normalized_backend = raw_backend.strip().lower().replace("-", "_")
    aliases: dict[str, MXFP8MatmulBIBackend] = {
        "cublas": "cublas",
        "cublaslt": "cublas",
        "cu_blas": "cublas",
        "cu_blaslt": "cublas",
        "native": "native",
        "native_fp8": "native",
        "fp8": "native",
        "qdq": "qdq",
        "te": "cublas",
        "transformer_engine": "cublas",
    }
    if normalized_backend in aliases:
        return aliases[normalized_backend]

    valid_values = ", ".join(sorted(aliases))
    raise ValueError(
        f"{G_MXFP8_MATMUL_BI_BACKEND_ENV} must be one of {{{valid_values}}}; "
        f"got {raw_backend!r}."
    )


def get_te_cublas_workspace_size_bytes_from_env() -> int | None:
    raw_size = os.environ.get(G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV)
    if raw_size is None or raw_size.strip() == "":
        return None
    try:
        size_bytes = int(raw_size)
    except ValueError as exc:
        raise ValueError(
            f"{G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV} must be an integer byte "
            f"count; got {raw_size!r}."
        ) from exc
    if size_bytes < 0:
        raise ValueError(
            f"{G_TE_CUBLAS_WORKSPACE_SIZE_BYTES_ENV} must be non-negative; "
            f"got {raw_size!r}."
        )
    return size_bytes


def ensure_transformer_engine_importable_from_env() -> None:
    """Append a caller-provided TE site-packages path when TE is not installed."""
    if importlib.util.find_spec("transformer_engine") is not None:
        return

    raw_site_packages = os.environ.get(G_TE_SITE_PACKAGES_ENV)
    if raw_site_packages is None or raw_site_packages.strip() == "":
        return

    for site_packages in raw_site_packages.split(os.pathsep):
        site_packages = site_packages.strip()
        if site_packages and site_packages not in sys.path:
            sys.path.append(site_packages)
    importlib.invalidate_caches()


def install_te_cublas_workspace_limit_from_env() -> dict[str, object]:
    """Clamp Transformer Engine's cuBLASLt workspace size when requested."""
    ensure_transformer_engine_importable_from_env()
    size_bytes = get_te_cublas_workspace_size_bytes_from_env()
    if size_bytes is None:
        return {
            "patched": False,
            "workspace_limit_bytes": None,
            "cache_cleared": False,
        }

    te_gemm_mod = importlib.import_module(
        "transformer_engine.pytorch.cpp_extensions.gemm"
    )

    def _get_cublas_workspace_size_bytes() -> int:
        return size_bytes

    te_gemm_mod.get_cublas_workspace_size_bytes = _get_cublas_workspace_size_bytes
    cache_clear = getattr(
        getattr(te_gemm_mod, "get_cublas_workspace", None),
        "cache_clear",
        None,
    )
    cache_cleared = callable(cache_clear)
    if cache_cleared:
        cache_clear()

    return {
        "patched": True,
        "workspace_limit_bytes": size_bytes,
        "cache_cleared": cache_cleared,
    }
