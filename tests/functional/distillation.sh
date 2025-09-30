#!/bin/bash

# clean up checkpoint directory on exit
trap "rm -rf /tmp/distillation_checkpoints" EXIT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -euo pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_distillation_math.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    teacher.model_name=Qwen/Qwen3-0.6B \
    cluster.gpus_per_node=2 \
    policy.train_global_batch_size=16 \
    policy.dtensor_cfg.tensor_parallel_size=1 \
    policy.dtensor_cfg.context_parallel_size=2 \
    policy.max_total_sequence_length=2048 \
    teacher.dtensor_cfg.tensor_parallel_size=2 \
    teacher.dtensor_cfg.context_parallel_size=1 \
    distillation.max_num_steps=3 \
    distillation.num_prompts_per_step=16 \
    distillation.max_val_samples=16 \
    distillation.val_batch_size=8 \
    distillation.val_period=3 \
    data.dataset_name=OpenMathInstruct-2 \
    loss_fn.zero_outside_topk=true \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.save_period=3 \
    checkpointing.checkpoint_dir=/tmp/distillation_checkpoints \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
  'data["train/loss"]["3"] < 1.0'
