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

import os
from typing import Literal

MXFP8MatmulBIBackend = Literal["native", "qdq"]

G_MXFP8_MATMUL_BI_BACKEND_ENV = "NEMO_RL_MXFP8_MATMUL_BI_BACKEND"
G_DEFAULT_MXFP8_MATMUL_BI_BACKEND: MXFP8MatmulBIBackend = "qdq"


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
        "native": "native",
        "native_fp8": "native",
        "fp8": "native",
        "qdq": "qdq",
    }
    if normalized_backend in aliases:
        return aliases[normalized_backend]

    valid_values = ", ".join(sorted(aliases))
    raise ValueError(
        f"{G_MXFP8_MATMUL_BI_BACKEND_ENV} must be one of {{{valid_values}}}; "
        f"got {raw_backend!r}."
    )
