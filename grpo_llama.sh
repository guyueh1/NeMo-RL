#!/bin/bash
#
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

NUM_NODES=${NUM_NODES:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export GPUS_PER_NODE
PRECISION=${PRECISION:-"fp8"}
MXFP8_MATMUL_BI_BACKEND=${MXFP8_MATMUL_BI_BACKEND:-"native"}
MEGATRON_BIAS_ACTIVATION_FUSION=${MEGATRON_BIAS_ACTIVATION_FUSION:-"false"}
if [ "${PRECISION}" == "bf16" ]; then
    VLLM_PRECISION="bfloat16"
    VLLM_IS_MX="false"
    MEGATRON_FP8_ENABLED=false
else
    VLLM_PRECISION="fp8"
    VLLM_IS_MX="true"
    MEGATRON_FP8_ENABLED=true
fi

BF16_TRUE_ON_POLICY=${BF16_TRUE_ON_POLICY:-"true"}
if [ "${PRECISION}" == "fp8" ]; then
    MXFP8_MATMUL_BATCH_INVARIANT=${MXFP8_MATMUL_BATCH_INVARIANT:-"true"}
else
    MXFP8_MATMUL_BATCH_INVARIANT=${MXFP8_MATMUL_BATCH_INVARIANT:-"false"}
fi

if [ "${MXFP8_MATMUL_BATCH_INVARIANT}" == "true" ] && [ "${BF16_TRUE_ON_POLICY}" != "true" ]; then
    echo "policy.mxfp8_matmul_batch_invariant=true requires policy.bf16_true_on_policy=true"
    exit 1
fi

TRUE_ON_POLICY_ARGS="policy.bf16_true_on_policy=${BF16_TRUE_ON_POLICY} policy.mxfp8_matmul_batch_invariant=${MXFP8_MATMUL_BATCH_INVARIANT}"

WANDB_PROJECT=${WANDB_PROJECT:-"guyueh-nemo-rl-mxfp8-lp"}
JOB_NAME=grpo-llama-nodes-${NUM_NODES}-gpus-${GPUS_PER_NODE}-precision-${PRECISION}-mxfp8-backend-${MXFP8_MATMUL_BI_BACKEND}-true-on-policy-${BF16_TRUE_ON_POLICY}-mxfp8-bi-${MXFP8_MATMUL_BATCH_INVARIANT}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-"/lustre/fsw/portfolios/llmservice/users/guyueh/grpo-runs"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${CHECKPOINT_ROOT}/${JOB_NAME}"}

TRAIN_CMD="\
NEMO_RL_MCORE_VLLM_USE_MCORE_WITH_VLLM_PATH=${NEMO_RL_MCORE_VLLM_USE_MCORE_WITH_VLLM_PATH:-1} \
NEMO_RL_MXFP8_MATMUL_BI_BACKEND=${MXFP8_MATMUL_BI_BACKEND} \
uv run examples/run_grpo.py \
--config examples/configs/grpo_math_8B_megatron.yaml \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.precision=${VLLM_PRECISION} \
++policy.generation.vllm_cfg.is_mx=${VLLM_IS_MX} \
policy.generation.vllm_cfg.async_engine=false \
policy.megatron_cfg.pipeline_model_parallel_size=1 \
policy.megatron_cfg.fp8_cfg.enabled=${MEGATRON_FP8_ENABLED} \
policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8 \
policy.megatron_cfg.fp8_cfg.fp8_param=false \
grpo.num_prompts_per_step=16 \
grpo.num_generations_per_prompt=16 \
policy.train_global_batch_size=256 \
checkpointing.enabled=true \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
checkpointing.save_period=50 \
logger.wandb_enabled=true \
logger.wandb.project=${WANDB_PROJECT} \
logger.wandb.name=${JOB_NAME} \
policy.sequence_packing.enabled=false \
++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN \
++policy.megatron_cfg.attention_backend=flash \
policy.megatron_cfg.apply_rope_fusion=false \
policy.megatron_cfg.bias_activation_fusion=${MEGATRON_BIAS_ACTIVATION_FUSION} \
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false \
loss_fn.reference_policy_kl_penalty=0.0 \
loss_fn.use_importance_sampling_correction=true \
++grpo.skip_reference_policy_logprobs_calculation=true \
${TRUE_ON_POLICY_ARGS} \
${*}
"

echo "BF16_TRUE_ON_POLICY: ${BF16_TRUE_ON_POLICY}"
echo "MXFP8_MATMUL_BATCH_INVARIANT: ${MXFP8_MATMUL_BATCH_INVARIANT}"
echo "MXFP8_MATMUL_BI_BACKEND: ${MXFP8_MATMUL_BI_BACKEND}"
echo "VLLM_PRECISION: ${VLLM_PRECISION}"
echo "VLLM_IS_MX: ${VLLM_IS_MX}"
echo "MEGATRON_FP8_ENABLED: ${MEGATRON_FP8_ENABLED}"
echo "MEGATRON_BIAS_ACTIVATION_FUSION: ${MEGATRON_BIAS_ACTIVATION_FUSION}"
echo "JOB_NAME: ${JOB_NAME}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "TRUE_ON_POLICY_ARGS: ${TRUE_ON_POLICY_ARGS}"

INTERACTIVE=${INTERACTIVE:-"0"}
if [ $INTERACTIVE -eq 0 ]; then
export COMMAND=${TRAIN_CMD}
export CONTAINER=/lustre/fsw/portfolios/coreai/users/guyueh/container_image/RL_custom_vllm.sqsh
export HSG_FSW_USER_DIR=/lustre/fsw/portfolios/coreai/users/guyueh
export HSG_FS1_USER_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/guyueh
SUBMIT_DIR_LOGICAL=${PWD}
SUBMIT_DIR_PHYSICAL=$(pwd -P)
REPO_MOUNTS="${SUBMIT_DIR_LOGICAL}:${SUBMIT_DIR_LOGICAL},${SUBMIT_DIR_LOGICAL}:/opt/nemo-rl"
if [ "${SUBMIT_DIR_PHYSICAL}" != "${SUBMIT_DIR_LOGICAL}" ]; then
    REPO_MOUNTS="${REPO_MOUNTS},${SUBMIT_DIR_PHYSICAL}:${SUBMIT_DIR_PHYSICAL}"
fi
export BASE_LOG_DIR=${BASE_LOG_DIR:-${SUBMIT_DIR_LOGICAL}}
export MOUNTS="\
${HSG_FSW_USER_DIR}:${HSG_FSW_USER_DIR},\
${HSG_FS1_USER_DIR}:${HSG_FS1_USER_DIR},\
/lustre:/lustre,\
${REPO_MOUNTS},\
/home/guyueh/:/home/guyueh/"

sbatch \
    --nodes=${NUM_NODES} \
    --segment=${NUM_NODES} \
    --account=nemotron_n4_post \
    --job-name=nemotron_n4_post-${JOB_NAME} \
    --partition=${PARTITION:-batch} \
    --gres=gpu:${GPUS_PER_NODE} \
    --mem=0 \
    --time=04:00:00 \
    ray.sub

else
    eval $TRAIN_CMD
fi
