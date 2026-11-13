#!/usr/bin/env bash
# Serve the two models SoongTunes needs on one 24GB GPU with vLLM (OpenAI-compatible API).
#
#   ./serve.sh start | stop | status | logs
#
# Env (or ~/stunes-llm/.env): STUNES_LLM_HOST (bind address, default: this host's
# Tailscale IP), STUNES_LLM_API_KEY (required), VLLM (path to the vllm binary).
#
# GPU budget on an RTX 4090 (24GB): the embedder starts first and takes ~12%, then
# gpt-oss-20b (MXFP4 weights, ~13GB) takes 75% including KV cache. vLLM sizes each
# process from its own fraction of *total* memory, so the two must sum below ~0.9.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a

VLLM=${VLLM:-./.venv/bin/vllm}
HOST=${STUNES_LLM_HOST:-$(tailscale ip -4 2>/dev/null | head -1)}
KEY=${STUNES_LLM_API_KEY:?set STUNES_LLM_API_KEY}
# chat model: gpt-oss-20b by default. Interim alternative while it downloads (troubleshooting #8):
#   STUNES_CHAT_MODEL=~/models/Qwen3-8B STUNES_CHAT_NAME=qwen3-8b STUNES_CHAT_GPU=0.90 \
#   STUNES_CHAT_ARGS="--reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice" \
#   STUNES_EMBEDDER=off ./serve.sh start
CHAT_MODEL=${STUNES_CHAT_MODEL:-openai/gpt-oss-20b}
CHAT_NAME=${STUNES_CHAT_NAME:-gpt-oss-20b}
CHAT_GPU=${STUNES_CHAT_GPU:-0.75}
CHAT_LEN=${STUNES_CHAT_LEN:-32768}
CHAT_ARGS=${STUNES_CHAT_ARGS:---reasoning-parser openai_gptoss --tool-call-parser openai --enable-auto-tool-choice}
EMBEDDER=${STUNES_EMBEDDER:-on}
mkdir -p logs run

# vLLM shells out to `ninja` (FlashInfer JIT) by name, so the venv's bin must be on PATH
# even though we don't `activate` it. Requests are greedy (temperature 0), so the
# FlashInfer sampler isn't worth a JIT build on hosts without nvcc.
export PATH="$(cd "$(dirname "$VLLM")" && pwd):$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

start_embedder() {
  nohup "$VLLM" serve microsoft/harrier-oss-v1-0.6b \
    --served-model-name harrier-oss-v1-0.6b \
    --runner pooling --convert embed \
    --host "$HOST" --port 8201 --api-key "$KEY" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.12 \
    > logs/embedder.log 2>&1 < /dev/null &
  echo $! > run/embedder.pid
}

start_chat() {
  # shellcheck disable=SC2086  # CHAT_ARGS is a list of flags
  nohup "$VLLM" serve "$CHAT_MODEL" \
    --served-model-name "$CHAT_NAME" \
    --host "$HOST" --port 8200 --api-key "$KEY" \
    --max-model-len "$CHAT_LEN" \
    --gpu-memory-utilization "$CHAT_GPU" \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    $CHAT_ARGS \
    > logs/chat.log 2>&1 < /dev/null &
  echo $! > run/chat.pid
}

wait_ready() {  # port name
  for _ in $(seq 1 180); do
    if curl -fs -H "Authorization: Bearer $KEY" "http://$HOST:$1/v1/models" >/dev/null; then
      echo "$2 ready on $HOST:$1"; return 0
    fi
    sleep 2
  done
  echo "$2 did not become ready, see logs/$2.log" >&2; return 1
}

stop_one() {
  if [ -f "run/$1.pid" ] && kill -0 "$(cat "run/$1.pid")" 2>/dev/null; then
    kill "$(cat "run/$1.pid")" && echo "stopped $1"
  fi
  rm -f "run/$1.pid"
}

case "${1:-}" in
  start)
    if [ "$EMBEDDER" = on ]; then start_embedder; wait_ready 8201 embedder; fi
    start_chat; wait_ready 8200 chat
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader ;;
  stop) stop_one chat; stop_one embedder ;;
  status)
    for p in 8201 8200; do
      curl -fs -H "Authorization: Bearer $KEY" "http://$HOST:$p/v1/models" | head -c 200 || echo "port $p down"
      echo
    done ;;
  logs) tail -n 50 logs/*.log ;;
  *) echo "usage: $0 start|stop|status|logs" >&2; exit 2 ;;
esac
