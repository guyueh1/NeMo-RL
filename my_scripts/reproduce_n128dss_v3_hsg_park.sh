#!/usr/bin/env bash
# Prepare/submit the 128-node HSG-Park reproduction of W&B run
# n128cmp-20260923171556-0b29. Review-only by default: SUBMIT=1 is required.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

EXP_NAME=${EXP_NAME:-n128dss-v4-guyueh}
RUN_DATE=20260923
RUN_DIR=${RUN_DIR:-/lustre/fsw/portfolios/llmservice/users/guyueh/sft-runs/${EXP_NAME}-${RUN_DATE}}
CONFIG=${CONFIG:-${REPO_DIR}/examples/configs/recipes/vlm/vlm_sft-nemotron-super-67b-128n4g-megatron-tp8ep64cp16-energon-n128dss-v4-guyueh.yaml}
CONTAINER=${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/guyueh/container_images/nvidian+nemo-rl+nightly.sqsh}

ACCOUNT=${ACCOUNT:-nemotron_n4_omni}
PARTITION=${PARTITION:-batch}
NUM_NODES=${NUM_NODES:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
QOS=${QOS:-}
NUM_RUNS=${NUM_RUNS:-1}
SUBMIT=${SUBMIT:-0}

WANDB_PROJECT=${WANDB_PROJECT:-rohit-sft_v2_upstream}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-${EXP_NAME}-${RUN_DATE}}

# The default HF -> Megatron-Bridge conversion cache is $HF_HOME/nemo_rl.
# Submit with `bash -ic './my_scripts/reproduce_n128dss_v3_hsg_park.sh'` so
# HSG's ~/.bashrc exports its shared Lustre HF_HOME before sbatch captures the
# environment for every worker rank.
if [[ "${SUBMIT}" == 1 && "${HF_HOME:-}" != /lustre/* ]]; then
    echo "ERROR: submit through bash -ic so ~/.bashrc provides a shared Lustre HF_HOME (got ${HF_HOME:-<unset>})" >&2
    exit 1
fi

MODEL=/lustre/fsw/portfolios/coreai/users/rohitkumarj/data/models/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained
DATASET_CONFIG=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/rohitkumarj/code/RL/nemo-rl-energon-sft-prodrun/examples/configs/sft_v2_tests/sft_super35_generalist_full_no_fs12.yaml
DSS_CACHE=/home/svc-dss/cache/nemotron

for path in "${REPO_DIR}/ray.sub" "${CONFIG}" "${CONTAINER}" "${MODEL}" "${DATASET_CONFIG}" "${DSS_CACHE}"; do
    if [[ ! -r "${path}" ]]; then
        echo "ERROR: required HSG-Park path is not readable: ${path}" >&2
        exit 1
    fi
done

if [[ "${GPUS_PER_NODE}" != 4 ]]; then
    echo "ERROR: this reproduction requires GPUS_PER_NODE=4" >&2
    exit 1
fi

MOUNTS="${REPO_DIR}/nemo_rl:/opt/nemo-rl/nemo_rl"
MOUNTS+=",${REPO_DIR}/examples:/opt/nemo-rl/examples"
MOUNTS+=",${REPO_DIR}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
MOUNTS+=",${REPO_DIR}/tools:/opt/nemo-rl/tools"
MOUNTS+=",/lustre:/lustre,${DSS_CACHE}:${DSS_CACHE}"
if [[ -r "${HOME}/.netrc" ]]; then
    MOUNTS+=",${HOME}/.netrc:/root/.netrc"
fi
CONFIG_IN_CONTAINER="/opt/nemo-rl/${CONFIG#${REPO_DIR}/}"
HF_MODULES_CACHE_DIR="${RUN_DIR}/hf_modules"
LAUNCH_LOG_DIR="${RUN_DIR}/launcher"
COMMAND="cd /opt/nemo-rl && export HF_MODULES_CACHE='${HF_MODULES_CACHE_DIR}' && export PYTHONPATH='${HF_MODULES_CACHE_DIR}':\${PYTHONPATH:-} && uv run --no-sync python -c \"import nemo_rl; from transformers import AutoConfig, AutoProcessor, AutoTokenizer; p='${MODEL}'; AutoConfig.from_pretrained(p, trust_remote_code=True); AutoProcessor.from_pretrained(p, trust_remote_code=True, use_fast=True); AutoTokenizer.from_pretrained(p, trust_remote_code=True, use_fast=True); print('Prewarmed HF dynamic modules cache')\" && NRL_REPRO_RUN_DIR='${RUN_DIR}' WANDB_PROJECT='${WANDB_PROJECT}' WANDB_RUN_NAME='${WANDB_RUN_NAME}' uv run --no-sync examples/run_sft_v2.py --config '${CONFIG_IN_CONTAINER}' cluster.num_nodes=${NUM_NODES} cluster.gpus_per_node=${GPUS_PER_NODE}"
IDLE_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"data_loading","description":"128-node Nemotron Super SFTv2 loads a 232GB model and initializes DSS/Energon before the first optimizer step"}}'

SBATCH_ARGS=(
    --nodes="${NUM_NODES}"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --job-name="${ACCOUNT}:${EXP_NAME}"
    --time="${TIME_LIMIT}"
    --gres="gpu:${GPUS_PER_NODE}"
    --exclusive
    --comment="${IDLE_COMMENT}"
    --output="${RUN_DIR}/slurm-%j.out"
)
if [[ -n "${QOS}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

echo "Repository:    ${REPO_DIR}"
echo "Config:        ${CONFIG}"
echo "Run directory: ${RUN_DIR}"
echo "Container:     ${CONTAINER}"
echo "Allocation:    ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs (${TIME_LIMIT})"
echo "QoS:           ${QOS:-default}"
echo "W&B:           ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "HF home:       ${HF_HOME:-<set by interactive Bash at submission>}"
echo "HF modules:    ${HF_MODULES_CACHE_DIR}"
echo "Launcher logs: ${LAUNCH_LOG_DIR}"
echo "Resume jobs:   ${NUM_RUNS} (all share ${RUN_DIR}/checkpoints)"

if [[ "${SUBMIT}" != 1 ]]; then
    printf '\nReview mode; no directories or jobs were created. To submit later:\n\n'
    printf '  SUBMIT=1 bash -ic %q  # sources ~/.bashrc; requires WANDB_API_KEY or ~/.netrc\n\n' "exec $0"
    printf 'The first sbatch invocation would be:\n  '
    printf '%q ' env "NRL_REPRO_RUN_DIR=${RUN_DIR}" "WANDB_PROJECT=${WANDB_PROJECT}" "WANDB_RUN_NAME=${WANDB_RUN_NAME}" "GPUS_PER_NODE=${GPUS_PER_NODE}" "BASE_LOG_DIR=${LAUNCH_LOG_DIR}" "CONTAINER=${CONTAINER}" "MOUNTS=${MOUNTS}" "COMMAND=${COMMAND}" sbatch "${SBATCH_ARGS[@]}" "${REPO_DIR}/ray.sub"
    printf '\n'
    exit 0
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -r "${HOME}/.netrc" ]]; then
    echo "ERROR: configure W&B through WANDB_API_KEY or ${HOME}/.netrc before submitting" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"
mkdir -p "${LAUNCH_LOG_DIR}"
if git -C "${REPO_DIR}" rev-parse HEAD > "${RUN_DIR}/git-commit.txt"; then
    git -C "${REPO_DIR}" status --short > "${RUN_DIR}/git-status.txt"
else
    echo "unavailable: staged worktree has no usable Git metadata" > "${RUN_DIR}/git-commit.txt"
    echo "unavailable: staged worktree has no usable Git metadata" > "${RUN_DIR}/git-status.txt"
    echo "WARNING: could not record Git metadata for staged worktree ${REPO_DIR}" >&2
fi
cp "${CONFIG}" "${RUN_DIR}/$(basename "${CONFIG}")"

previous_job_id=
for ((run_index = 1; run_index <= NUM_RUNS; run_index++)); do
    run_args=("${SBATCH_ARGS[@]}")
    if [[ -n "${previous_job_id}" ]]; then
        run_args+=(--dependency="afterany:${previous_job_id}")
    fi

    submission=$(
        NRL_REPRO_RUN_DIR="${RUN_DIR}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
        HF_HOME="${HF_HOME}" \
        GPUS_PER_NODE="${GPUS_PER_NODE}" \
        BASE_LOG_DIR="${LAUNCH_LOG_DIR}" \
        CONTAINER="${CONTAINER}" \
        MOUNTS="${MOUNTS}" \
        COMMAND="${COMMAND}" \
        sbatch "${run_args[@]}" "${REPO_DIR}/ray.sub"
    )
    echo "${submission}"
    previous_job_id=${submission##* }
    if [[ ! "${previous_job_id}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: could not parse job id from: ${submission}" >&2
        exit 1
    fi
done

echo "Submitted ${NUM_RUNS} job(s); final dependency-chain job: ${previous_job_id}"
