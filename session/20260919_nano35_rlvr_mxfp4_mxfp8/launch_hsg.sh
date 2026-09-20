#!/bin/bash
set -euo pipefail

export PATH=/cm/local/apps/slurm/current/bin:${PATH}
source /home/${USER}/.my_secrets
test -n "${HF_TOKEN:-}"
test -n "${WANDB_API_KEY:-}"

export PIPELINE_ROOT=/lustre/fsw/portfolios/llmservice/users/guyueh/git-worktrees/post-training-pipelines-nemorl-v2-launcher
export NEMO_RL_ROOT=/lustre/fsw/portfolios/llmservice/users/guyueh/git-worktrees/RL-mxfp4-mxfp8
export RESULTS_ROOT=/lustre/fsw/portfolios/llmservice/users/guyueh/grpo-runs
export PERSISTENT_CACHE=/lustre/fsw/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/guyueh/cache
export CONTAINER=/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/nvidian+nemo-rl+nightly.sqsh
export SLURM_ACCOUNT=coreai_chef_posttrain
export SLURM_PARTITION=batch
export SLURM_QOS=normal
export SLURM_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"other","description":"Nano 3.5 RLVR startup builds isolated Ray venvs and model loading can leave GPUs idle before rollout and optimizer steps"}}'

export EXP_NAME=nano35-rlvr-v2-smoke-native-mxfp4-mxfp8-10step-20260920-retry13
export CONFIG_PATH_HOST=${PIPELINE_ROOT}/RLVR/nemotron-3.5-nano/configs/rlvr_sc_smoke_small_mxfp4_mxfp8.yaml
export NRL_MAX_STEPS=10
export WALLTIME=${WALLTIME:-04:00:00}
export WANDB_ENABLED=True
export WANDB_ENTITY=nvidia
export WANDB_PROJ=nano35-rlvr-mxfp4-mxfp8-smoke
export WANDB_NAME=${EXP_NAME}
exec bash "${PIPELINE_ROOT}/RLVR/nemotron-3.5-nano/scripts/launch_nano35_rlvr_v2_smoke_small.sh" \
  '~checkpointing.model_save_format' \
  '~checkpointing.save_consolidated' \
  "$@"
