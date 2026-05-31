#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-demo}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "Stopping local demo..."
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1"
    echo "Install it, then rerun ./run_local_demo.sh"
    exit 1
  fi
}

wait_for_backend() {
  echo "Waiting for backend on http://localhost:$BACKEND_PORT ..."
  for _ in $(seq 1 60); do
    if curl -fsS "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Backend did not become ready. Check backend.log for details."
  exit 1
}

need_command "$PYTHON_BIN"
need_command npm
need_command curl

echo "Post-Disaster Analytics local demo"
echo "Root: $ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Python virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing backend demo dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$BACKEND_DIR/requirements-demo.txt"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
fi

echo "Starting backend on http://localhost:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  PYTHONPATH=. "$VENV_DIR/bin/uvicorn" app.demo_main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload
) > "$ROOT_DIR/backend.log" 2>&1 &
BACKEND_PID="$!"

wait_for_backend

echo "Starting frontend on http://localhost:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  PORT="$FRONTEND_PORT" BROWSER=none npm start
) > "$ROOT_DIR/frontend.log" 2>&1 &
FRONTEND_PID="$!"

echo
echo "Demo is running:"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo "  Backend:  http://localhost:$BACKEND_PORT"
echo "  RAG:      http://localhost:$BACKEND_PORT/rag/status"
echo
echo "Logs:"
echo "  $ROOT_DIR/backend.log"
echo "  $ROOT_DIR/frontend.log"
echo
echo "Press Ctrl+C to stop both servers."

wait "$FRONTEND_PID"
