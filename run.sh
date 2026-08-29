#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
install_dependencies=false
backend_pid=""
frontend_pid=""

if [[ "${1:-}" == "--install" ]]; then
  install_dependencies=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: ./run.sh [--install]" >&2
  exit 2
fi

if [[ -f "$repo_dir/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$repo_dir/.env.local"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required and must point to PostgreSQL." >&2
  exit 1
fi

if [[ "$install_dependencies" == true ]]; then
  (cd "$repo_dir/backend" && uv sync)
  (cd "$repo_dir/frontend" && npm install)
fi

if [[ ! -x "$repo_dir/backend/.venv/bin/uvicorn" || ! -d "$repo_dir/frontend/node_modules" ]]; then
  echo "Dependencies are missing. Run ./run.sh --install." >&2
  exit 1
fi

cleanup() {
  trap - EXIT INT TERM
  [[ -z "$backend_pid" ]] || kill "$backend_pid" 2>/dev/null || true
  [[ -z "$frontend_pid" ]] || kill "$frontend_pid" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$repo_dir/backend" && uv run --no-sync uvicorn hosted_app.main:app --reload --host 127.0.0.1 --port 8000) &
backend_pid=$!
(cd "$repo_dir/frontend" && npm run dev -- --host 127.0.0.1 --port 5173) &
frontend_pid=$!

echo "API: http://localhost:8000"
echo "Web: http://localhost:5173"
wait
