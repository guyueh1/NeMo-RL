#! /bin/bash
GPFS=/lustre/fsw/portfolios/llmservice/users/guyueh/RL
PRECISION_RECIPE=${1:-"ds-fp8-gen"}

DS_FP8_GEN_EXTRA_ARGS="policy.generation.vllm_cfg.precision="fp8" \
++policy.generation.vllm_cfg.fp8_cfg.is_mx=false \
++policy.generation.vllm_cfg.fp8_cfg.dynamic_weight_quant=false \
++policy.generation.vllm_cfg.use_deep_gemm=true"

if [ "$PRECISION_RECIPE" == "ds-fp8-gen" ]; then
EXTRA_ARGS="$DS_FP8_GEN_EXTRA_ARGS"
elif [ "$PRECISION_RECIPE" == "bf16" ]; then
EXTRA_ARGS=""
else
    echo "Invalid recipe: $PRECISION_RECIPE"
    exit 1
fi

EXP_SUFFIX="llama3-8b-${PRECISION_RECIPE}-e8m0"
CHECKPOINT_DIR="results/${EXP_SUFFIX}"

export OMP_NUM_THREADS=16

WANDB_PROJ="nemo-rl-grpo-dev-guyueh"
WANDB_NAME="lax-super-v3-llama3-8b-${PRECISION_RECIPE}-e8m0"

# Create code snapshot using the tool (only copies git-tracked files)
mkdir -p ${CHECKPOINT_DIR}

export HF_HOME=/lustre/fsw/portfolios/llmservice/users/guyueh/hf_home
export HF_HUB_CACHE=/lustre/fsw/portfolios/llmservice/users/guyueh/hf_home/hub
export UV_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/guyueh/rl/uvcache


NUM_ACTOR_NODES=1

export COMMAND="\
NRL_VLLM_USE_V1=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_PRECOMPILED=1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
uv run examples/run_grpo_math.py \
--config examples/configs/grpo_math_8B_megatron.yaml \
policy.megatron_cfg.pipeline_model_parallel_size=1 \
loss_fn.force_on_policy_ratio=true \
loss_fn.reference_policy_kl_penalty=0.0 \
loss_fn.use_importance_sampling_correction=true \
cluster.num_nodes=${NUM_ACTOR_NODES} \
grpo.num_prompts_per_step=32 \
grpo.num_generations_per_prompt=16 \
policy.train_global_batch_size=512 \
logger.wandb_enabled=true \
logger.wandb.project=${WANDB_PROJ} \
logger.wandb.name=${WANDB_NAME} \
${EXTRA_ARGS} \
"

export CONTAINER="nvcr.io/nvidia/nemo-rl:v0.5.0"


export MOUNTS="/scratch:/scratch,\
/lustre:/lustre:ro,\
${GPFS}:/opt/nemo-rl"


# --account=llmservice_modelalignment_ppo \
# --account=llmservice_fm_text \

sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=coreai_dlalgo_nemorl \
    --job-name=${WANDB_NAME} \
    --partition=batch \
    --time=4:0:0 \
    --gres=gpu:8 \
    --exclusive \
    --dependency=singleton \
    ray-lbd.sub
