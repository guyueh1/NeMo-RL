#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Discovered by the L1 H100 test plan. Bypass run_unit.sh's two-GPU mask:
# these combined TP=2/CP=2 backward checks require four visible GPUs.
set -xeuo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
cd "${PROJECT_ROOT}/tests"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
uv run --no-sync python -c 'import torch; assert torch.cuda.device_count() >= 4, "Score-centering TP=2/CP=2 backward checks require four visible GPUs"'
uv run --no-sync python -m pytest unit/distributed/test_model_utils.py \
    -k 'test_score_centering_packed_tp_cp or test_get_policy_logprobs_at_sampler_topk_tp_cp' \
    --hf-gated --cov=nemo_rl --cov-report=term-missing
