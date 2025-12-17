#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root regardless of invocation location.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="$ROOT_DIR/.venv"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"
BUN_BIN="${BUN:-bun}"

if ! command -v "$BUN_BIN" >/dev/null 2>&1; then
  echo "bun is required but not installed. Install bun and retry." >&2
  exit 1
fi

# Create virtual environment if missing and install backend dependencies.
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
uv sync

deactivate

# Install frontend dependencies with bun (respect bun.lock when present)
if [ -d "$ROOT_DIR/frontend" ]; then
  pushd "$ROOT_DIR/frontend" >/dev/null
  if [ -f bun.lock ]; then
    "$BUN_BIN" install --frozen-lockfile
  else
    "$BUN_BIN" install
  fi
  popd >/dev/null
fi

# Launch backend (uvicorn) and frontend (Vite dev server) together; ensure clean shutdown.
trap 'kill 0' EXIT

source "$VENV_DIR/bin/activate"
uvicorn app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

pushd "$ROOT_DIR/frontend" >/dev/null
"$BUN_BIN" run dev -- --host &
FRONTEND_PID=$!
popd >/dev/null

wait $BACKEND_PID $FRONTEND_PID
