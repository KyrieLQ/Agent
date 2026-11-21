#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: bash training/train.sh {sft|grpo|sft-4b|grpo-4b} [Hydra overrides...]" >&2; }
mode="${1:-}"
case "$mode" in
  -h|--help) usage; exit 0 ;;
  sft|grpo|sft-4b|grpo-4b) shift ;;
  *) usage; exit 2 ;;
esac

: "${MERA_ROOT:?set MERA_ROOT to the MERA-Evolve checkout}"
: "${TRAIN_VENV:?set TRAIN_VENV to the VeRL training environment}"
: "${MODEL_PATH:?set MODEL_PATH to the base model checkpoint}"
export MODEL_PATH TRAIN_VENV
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$MERA_ROOT"

case "$mode" in
  sft|sft-4b)
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
    launcher=(bash scripts/train_sft.sh)
    if [[ "$mode" == sft-4b ]]; then
      if ! [[ "$CUDA_VISIBLE_DEVICES" =~ ^([0-9]+),([0-9]+)$ ]] || [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
        echo "Set SFT_GPU to two distinct GPU IDs for sft-4b." >&2
        exit 2
      fi
      export N_GPUS=2
      export QWEN35_TRAIN_TRITON_OVERLAY="${QWEN35_TRAIN_TRITON_OVERLAY:-$PWD/.deps/qwen35-triton33}"
      launcher=("$PYTHON" "$here/launch_4b.py" sft scripts/train_sft.sh)
    fi
    exec "${launcher[@]}" data.num_workers=0 model.use_fused_kernels=True model.use_remove_padding=False "$@"
    ;;
  grpo|grpo-4b)
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
    if [[ "$mode" == grpo-4b ]]; then
      if ! [[ "$TRAIN_GPU" =~ ^([0-9]+),([0-9]+)$ && "$ROLLOUT_GPU" =~ ^[0-9]+$ ]]; then
        echo "Set TRAIN_GPU to two GPU IDs and ROLLOUT_GPU to one GPU ID for grpo-4b." >&2
        exit 2
      fi
      IFS=, read -r gpu0 gpu1 <<< "$TRAIN_GPU"
      if [[ "$gpu0" == "$gpu1" || "$gpu0" == "$ROLLOUT_GPU" || "$gpu1" == "$ROLLOUT_GPU" ]]; then
        echo "The training and rollout GPU IDs must be distinct." >&2
        exit 2
      fi
      stack="${MERA_GRPO_ROOT:-$PWD}"
      experiment="$stack/experiments/tau-2"
      export CUDA_VISIBLE_DEVICES="$TRAIN_GPU,$ROLLOUT_GPU"
      export N_GPUS=2 ROLLOUT_N_GPUS=1 SEPARATE_ROLLOUT=1
      export QWEN35_ENABLE_VERL_PATCHES=1
      export QWEN35_TRAIN_TRITON_OVERLAY="${QWEN35_TRAIN_TRITON_OVERLAY:-$PWD/.deps/qwen35-triton33}"
      export PYTHONPATH="$experiment/compat/qwen35_torch_fallback:${PYTHONPATH:-}"
      export ACTOR_LR=2e-6 MAX_PROMPT_LENGTH=8192 ROLLOUT_TEMPERATURE=1.0 ROLLOUT_TOP_P=0.98
      export ROLLOUT_GPU_MEMORY_UTILIZATION=0.70 ROLLOUT_ENFORCE_EAGER=True
      export REWARD_WORKERS=2 RAY_NUM_CPUS=24 AGENT_THINKING=False TOOL_FORMAT=hermes
      export MODEL_ATTN_IMPLEMENTATION=sdpa USE_REMOVE_PADDING=False
      export CHECKPOINT_BACKEND=nixl_tau2 CHECKPOINT_CUSTOM_BACKEND_MODULE=tau2_evolve.nixl_checkpoint
      export VLLM_GDN_PREFILL_BACKEND=triton VLLM_LANGUAGE_MODEL_ONLY=True
      export CIRRUS_GROUP_MODE=plain CIRRUS_ROLLOUT_BUDGET="$((TOTAL_STEPS * TRAIN_BATCH_SIZE * N_GENERATIONS))"
      export CIRRUS_ROLLOUT_LEDGER="$OUTPUT_DIR/rollout_ledger.jsonl"
      exec "$TRAIN_VENV/bin/python" "$here/launch_4b.py" grpo "$experiment/scripts/train_verl_grpo.sh" \
        data.shuffle=False actor_rollout_ref.model.use_fused_kernels=True \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 actor_rollout_ref.model.lora.merge=True \
        trainer.max_actor_ckpt_to_keep=1 "$@"
    fi
    exec bash experiments/tau-2/03_grpo_only.sh actor_rollout_ref.model.use_fused_kernels=True \
      trainer.max_actor_ckpt_to_keep=1 "$@"
    ;;
esac
