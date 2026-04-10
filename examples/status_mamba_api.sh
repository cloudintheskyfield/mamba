#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/usr/maintain/mammba_logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/mamba_api.pid}"
PORT="${PORT:-18080}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "running pid=$(cat "$PID_FILE") port=$PORT"
else
  echo "not running"
fi

curl -s "http://127.0.0.1:${PORT}/health" || true
echo
