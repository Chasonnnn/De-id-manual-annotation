#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
INSTALL_DEPS=false
backend_job_pid=""
frontend_job_pid=""
cleanup_started=0

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

port_listener_pids() {
    local port="$1"
    lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

process_cwd() {
    local pid="$1"
    lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | awk 'BEGIN { FS = "" } /^n/ { print substr($0, 2) }'
}

wait_for_port_release() {
    local port="$1"
    local attempts=50
    while [[ $attempts -gt 0 ]]; do
        if [[ -z "$(port_listener_pids "$port")" ]]; then
            return 0
        fi
        sleep 0.1
        attempts=$((attempts - 1))
    done
    echo "Timed out waiting for port $port to become available."
    return 1
}

wait_for_backend_http_ready() {
    local port="$1"
    local url="http://127.0.0.1:${port}/api/agent/methods"
    local attempts=100
    while [[ $attempts -gt 0 ]]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
        attempts=$((attempts - 1))
    done
    echo "Timed out waiting for backend HTTP readiness at $url."
    return 1
}

restart_repo_processes_on_port() {
    local port="$1"
    local label="$2"
    local match_dir="$3"
    shift 3
    local expected_terms=("$@")
    local port_var_name
    case "$label" in
        backend) port_var_name="BACKEND_PORT" ;;
        frontend) port_var_name="FRONTEND_PORT" ;;
        *) port_var_name="PORT" ;;
    esac
    local pids
    pids="$(port_listener_pids "$port")"
    [[ -z "$pids" ]] && return 0

    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        local cwd
        cwd="$(process_cwd "$pid")"
        local matches_repo=false
        if [[ "$cwd" == "$match_dir" ]]; then
            matches_repo=true
        fi

        if [[ "$matches_repo" == true ]]; then
            echo "Restarting existing $label on port $port (PID $pid)..."
            kill "$pid"
        else
            echo "Port $port is already in use by an unrelated process:"
            if [[ -n "$cwd" ]]; then
                echo "  PID $pid (cwd: $cwd)"
            else
                echo "  PID $pid"
            fi
            echo "Stop it manually or set $port_var_name to a different value."
            exit 1
        fi
    done <<< "$pids"

    wait_for_port_release "$port"
}

cleanup() {
    if [[ "$cleanup_started" -eq 1 ]]; then
        return
    fi
    cleanup_started=1
    trap - SIGINT SIGTERM EXIT
    echo "Shutting down..."
    terminate_pid_tree "$backend_job_pid"
    terminate_pid_tree "$frontend_job_pid"
    wait
}

terminate_pid_tree() {
    local pid="$1"
    [[ -z "$pid" ]] && return 0

    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    while IFS= read -r child_pid; do
        [[ -z "$child_pid" ]] && continue
        terminate_pid_tree "$child_pid"
    done <<< "$children"

    kill "$pid" 2>/dev/null || true
}

trap cleanup EXIT SIGINT SIGTERM

restart_repo_processes_on_port "$BACKEND_PORT" "backend" "$BACKEND_DIR" "uvicorn" "server:app"
restart_repo_processes_on_port "$FRONTEND_PORT" "frontend" "$FRONTEND_DIR" "vite"

echo "Starting backend (FastAPI)..."
(cd "$BACKEND_DIR" && uv run uvicorn server:app --reload --port "$BACKEND_PORT") &
backend_job_pid=$!

wait_for_backend_http_ready "$BACKEND_PORT"

echo "Starting frontend (Vite)..."
(cd "$FRONTEND_DIR" && npm run dev -- --port "$FRONTEND_PORT") &
frontend_job_pid=$!

echo "Backend: http://localhost:$BACKEND_PORT"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "Runtime data folder: $BACKEND_DIR/.annotation_tool"
echo "Press Ctrl+C to stop both servers."

wait
