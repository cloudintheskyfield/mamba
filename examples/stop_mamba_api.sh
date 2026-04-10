#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${LOG_DIR:-/usr/maintain/mammba_logs}"
PID_FILE="${PID_FILE:-$LOG_DIR/mamba_api.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "mamba api is not running"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "stopped mamba api pid=$PID"
else
  echo "stale pid file found for pid=$PID"
fi

rm -f "$PID_FILE"
