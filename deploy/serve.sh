#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "$#" -ne 2 ]]; then
    echo "Usage: bash deploy/serve.sh {agent|simulator} /path/to/checkpoint"
    [[ "${1:-}" == "--help" ]] && exit 0
    exit 2
fi
role="$1"
checkpoint="$2"
vllm_bin="${VLLM_BIN:-vllm}"
profile="${SERVING_PROFILE:-default}"
runtime=(--no-enable-prefix-caching --no-async-scheduling --gdn-prefill-backend triton)
max_seqs=1

case "$role" in
    agent)
        cards="${AGENT_GPUS:-0}"
        port="${AGENT_PORT:-8260}"
        model=evol-llm-agent
        memory="${AGENT_MEMORY_FRACTION:-0.40}"
        length="${AGENT_CONTEXT:-40960}"
        extra=(--generation-config vllm)
        ;;
    simulator)
        cards="${SIM_GPUS:-1,2}"
        port="${SIM_PORT:-8261}"
        model=evol-llm-user
        memory="${SIM_MEMORY_FRACTION:-0.90}"
        length="${SIM_CONTEXT:-32768}"
        extra=(--reasoning-parser qwen3 --disable-custom-all-reduce)
        ;;
    *) echo "Unknown role: $role" >&2; exit 2 ;;
esac
case "$profile" in
    default) ;;
    4b)
        runtime=(--enable-prefix-caching --gdn-prefill-backend flashinfer)
        if [[ "$role" == agent ]]; then
            memory="${AGENT_MEMORY_FRACTION:-0.90}"
            length="${AGENT_CONTEXT:-32768}"
            max_seqs=8
            extra+=(--reasoning-parser qwen3 --disable-custom-all-reduce)
        else
            max_seqs=24
        fi
        ;;
    *) echo "Unknown serving profile: $profile" >&2; exit 2 ;;
esac
if ! [[ "$cards" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU IDs must be comma-separated integers." >&2
    exit 2
fi
IFS=, read -r -a gpu_ids <<< "$cards"
export CUDA_VISIBLE_DEVICES="$cards"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TRITON_CACHE_AUTOTUNING="${TRITON_CACHE_AUTOTUNING:-1}"
exec "$vllm_bin" serve "$checkpoint" \
    --host 127.0.0.1 --port "$port" --served-model-name "$model" \
    --dtype bfloat16 --language-model-only --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --tensor-parallel-size "${#gpu_ids[@]}" \
    --gpu-memory-utilization "$memory" --max-model-len "$length" \
    --max-num-seqs "${SERVE_MAX_SEQS:-$max_seqs}" --max-num-batched-tokens 4096 --enable-chunked-prefill \
    --enforce-eager "${runtime[@]}" "${extra[@]}"
