#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
INSTALL_DEPS=false

usage() {
    cat <<'EOF'
Usage:
  ./run.sh                Start backend + frontend dev servers
  ./run.sh --install      Install dependencies, then start both
  ./run.sh --help         Show help

Optional env vars:
  BACKEND_PORT (default: 8000)
  FRONTEND_PORT (default: 5173)

If .env.local exists at repo root, it is loaded automatically.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --install) INSTALL_DEPS=true ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            exit 1
            ;;
    esac
done

if [[ -f "$ROOT_DIR/.env.local" ]]; then
    echo "Loading env from $ROOT_DIR/.env.local"
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env.local"
    set +a
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Missing dependency: uv"
    echo "Install: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Missing dependency: npm (Node.js)"
    echo "Install Node.js from https://nodejs.org/"
    exit 1
fi

if [[ "$INSTALL_DEPS" == true ]]; then
    echo "Installing backend dependencies (uv sync)..."
    (cd "$BACKEND_DIR" && uv sync)
    echo "Installing frontend dependencies (npm install)..."
    (cd "$FRONTEND_DIR" && npm install)
fi

cleanup() {
    echo "Shutting down..."
    kill 0
    wait
}
trap cleanup SIGINT SIGTERM

echo "Starting backend (FastAPI)..."
(cd "$BACKEND_DIR" && uv run uvicorn server:app --reload --port "$BACKEND_PORT") &

echo "Starting frontend (Vite)..."
(cd "$FRONTEND_DIR" && npm run dev -- --port "$FRONTEND_PORT") &

echo "Backend: http://localhost:$BACKEND_PORT"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "Runtime data folder: $BACKEND_DIR/.annotation_tool"
echo "Press Ctrl+C to stop both servers."

wait
