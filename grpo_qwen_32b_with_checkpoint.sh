#!/bin/bash

export KITCHEN_RECIPE=6001
export MEGATRON_KITCHEN_RECIPE=6001
export EXP_SUFFIX="grpo_math_qwen32b_megatron_kitchen_with_checkpoint"
export CHECKPOINT_DIR="results/grpo_qwen32b_megatronrl"
export WANDB_NAME=${EXP_SUFFIX}
export WANDB_PROJECT="nemo-rl-grpo-dev-guyueh"
export RAY_DEDUP_LOGS=0
export HF_HOME=/lustre/fs1/portfolios/coreai/users/guyueh/hf_home
export HF_HUB_CACHE=/lustre/fs1/portfolios/coreai/users/guyueh/hf_home/hub
export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/users/guyueh/rl/uvcache
export HF_TOKEN=${HF_TOKEN}
export WANDB_API_KEY=${WANDB_KEY}
export BASE_LOG_DIR="logs/${EXP_SUFFIX}"

export CONTAINER="gitlab-master.nvidia.com/guyueh/container-repo/nemo_rl:fp8_training_kitchen-release"

export MOUNTS="\
/lustre:/lustre:ro,\
/lustre/fs1/portfolios/coreai/users/guyueh:/lustre/fs1/portfolios/coreai/users/guyueh,\
${PWD}:/opt/nemo-rl,\
/home/guyueh:/home/guyueh"

export NUM_ACTOR_NODES=${NUM_NODES:-8}

export COMMAND="\
uv run python examples/run_grpo_math.py \
--config examples/configs/grpo_math_8B_megatron.yaml \
cluster.num_nodes=$NUM_ACTOR_NODES \
policy.model_name=Qwen/Qwen2.5-32B \
loss_fn.use_importance_sampling_correction=true \
policy.generation.vllm_cfg.tensor_parallel_size=8 \
policy.generation.vllm_cfg.gpu_memory_utilization=0.3 \
policy.generation.vllm_cfg.precision=fp8 \
+policy.generation.vllm_cfg.use_kitchen=True \
+policy.generation.vllm_kwargs.max_num_seqs=32 \
policy.generation.vllm_cfg.enforce_eager=True \
policy.megatron_cfg.empty_unused_memory_level=2 \
policy.megatron_cfg.tensor_model_parallel_size=8 \
policy.megatron_cfg.sequence_parallel=True \
policy.megatron_cfg.pipeline_model_parallel_size=1 \
+policy.megatron_cfg.use_kitchen=True \
logger.wandb_enabled=true \
logger.wandb.project=${WANDB_PROJECT} \
logger.wandb.name=${WANDB_NAME} \
checkpointing.enabled=true \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
+checkpointing.only_load_weight_from_last_checkpoint=True "

INTERACTIVE=${INTERACTIVE:-0}
if [ $INTERACTIVE -eq 1 ]; then
    export COMMAND=
fi

export PARTITION=${PARTITION:-batch}

sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=coreai_dlalgo_nemorl-grpo.${EXP_SUFFIX} \
    --partition=${PARTITION} \
    --gres=gpu:8 \
    --time=04:00:00 \
    ray.sub

