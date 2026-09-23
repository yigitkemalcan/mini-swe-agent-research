#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'Usage: bash start-vllm.sh (run in tmux; uses existing Qwen/vLLM environment)'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Unexpected arguments; use --help.' >&2; exit 1; }
qwen_root="${MSWEA_QWEN_ROOT:-/mnt/jovans_models/experiment-yigit/qwen-experiment}"
export HF_HOME="$qwen_root/huggingface-cache"
export VLLM_CACHE_ROOT="$qwen_root/vllm-cache"
export UV_CACHE_DIR="$qwen_root/uv-cache"
snapshot="$HF_HOME/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e"
[[ -d "$snapshot" ]] || { echo "Missing model snapshot: $snapshot" >&2; exit 1; }
if [[ -n $(ss -H -ltn 'sport = :8000') ]]; then
    echo 'Port 8000 already listening; check the existing server.' >&2
    exit 1
fi
source "$qwen_root/envs/vllm/bin/activate"
cd "$qwen_root"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "$qwen_root/envs/vllm/bin/vllm" serve "$snapshot" \
    --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice --tool-call-parser hermes --host 127.0.0.1 --port 8000
