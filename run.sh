#!/usr/bin/env bash
set -euo pipefail

# Start both backend and frontend dev servers
# Ctrl+C will kill both processes

cleanup() {
    echo "Shutting down..."
    kill 0
    wait
}
trap cleanup SIGINT SIGTERM

echo "Starting backend (FastAPI)..."
cd "$(dirname "$0")/backend"
uv run uvicorn server:app --reload --port 8000 &

echo "Starting frontend (Vite)..."
cd "$(dirname "$0")/frontend"
npm run dev &

echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both servers."

wait
