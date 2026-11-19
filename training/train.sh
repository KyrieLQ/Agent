#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: bash training/train.sh {sft|grpo} [Hydra overrides...]" >&2; }
mode="${1:-}"
case "$mode" in
  -h|--help) usage; exit 0 ;;
  sft|grpo) shift ;;
  *) usage; exit 2 ;;
esac

: "${MERA_ROOT:?set MERA_ROOT to the MERA-Evolve checkout}"
: "${TRAIN_VENV:?set TRAIN_VENV to the VeRL training environment}"
: "${MODEL_PATH:?set MODEL_PATH to the base model checkpoint}"
export MODEL_PATH TRAIN_VENV
cd "$MERA_ROOT"

case "$mode" in
  sft)
    : "${SFT_DATA:?set SFT_DATA to the training parquet}"
    : "${SFT_OUTPUT_DIR:?set SFT_OUTPUT_DIR to a workspace path}"
    export SFT_DATA SFT_OUTPUT_DIR
    export CUDA_VISIBLE_DEVICES="${SFT_GPU:-0}"
    export PYTHON="$TRAIN_VENV/bin/python"
    export SFT_LR="${SFT_LR:-1e-5}"
    export SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-2}"
    export SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-2}"
    export SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-1}"
    export SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-10240}"
    export SFT_ENABLE_THINKING=False SFT_ATTN_IMPLEMENTATION=sdpa SFT_USE_TORCH_COMPILE=False
    export SFT_DATASET_PATH="$PWD/experiments/tau-2/tau2_evolve/sft_dataset.py"
    export SFT_DATASET_NAME=Qwen35MultiTurnSFTDataset SFT_SAVE_FREQ=-1
    export LORA_RANK="${LORA_RANK:-16}" LORA_ALPHA="${LORA_ALPHA:-32}"
    exec bash scripts/train_sft.sh data.num_workers=0 model.use_fused_kernels=True model.use_remove_padding=False "$@"
    ;;
  grpo)
    : "${TRAIN_FILE:?set TRAIN_FILE to the GRPO training parquet}"
    : "${TAU2_WORKSPACE:?set TAU2_WORKSPACE to the router-skills-evolve checkout}"
    : "${LORA_ADAPTER_PATH:?set LORA_ADAPTER_PATH to the SFT adapter}"
    : "${OUTPUT_DIR:?set OUTPUT_DIR to a workspace path}"
    : "${TRAIN_GPU:?set TRAIN_GPU to the actor training GPU}"
    : "${ROLLOUT_GPU:?set ROLLOUT_GPU to the rollout GPU}"
    if [[ "$TRAIN_GPU" == "$ROLLOUT_GPU" ]]; then
      echo "Training and rollout must use different GPUs." >&2
      exit 2
    fi
    export TRAIN_FILE TAU2_WORKSPACE LORA_ADAPTER_PATH OUTPUT_DIR TRAIN_GPU ROLLOUT_GPU
    export TOTAL_STEPS="${TOTAL_STEPS:-4}" SAVE_FREQ="${SAVE_FREQ:-1}"
    export PPO_MICRO_BATCH_SIZE=1 TRAIN_BATCH_SIZE=2 N_GENERATIONS=8
    export MAX_TURNS=24 MAX_RESPONSE_LENGTH=6144 AGENT_LOOP_WORKERS=8
    export TAU2_OFFICIAL_REWARD_WEIGHT=1
    exec bash experiments/tau-2/03_grpo_only.sh actor_rollout_ref.model.use_fused_kernels=True \
      trainer.max_actor_ckpt_to_keep=1 "$@"
    ;;
esac
