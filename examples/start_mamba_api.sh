#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt3/mnt2/data3/nlp/ws/proj/mammba}"
MODEL_DIR="${MODEL_DIR:-/mnt3/mnt2/data3/nlp/ws/model/mammba}"
VENV_DIR="${VENV_DIR:-/usr/maintain/.venvs/mammba}"
LOG_DIR="${LOG_DIR:-/usr/maintain/mammba_logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/mamba_api.pid}"
PORT="${PORT:-18080}"
HOST="${HOST:-0.0.0.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mamba-2.8b-hf}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "mamba api already running with pid $(cat "$PID_FILE")"
  exit 0
fi

export LD_LIBRARY_PATH="/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"
nohup "$VENV_DIR/bin/python" examples/mamba_openai_api.py \
  --model-path "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --device cuda \
  --dtype "$DTYPE" \
  > "$LOG_DIR/mamba_api.out" 2>&1 &

echo $! > "$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "started mamba api pid=$(cat "$PID_FILE") host=$HOST port=$PORT"
else
  echo "failed to start mamba api"
  exit 1
fi
