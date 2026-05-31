#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker-compose.demo.yml"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose -f "$COMPOSE_FILE")
elif command -v podman-compose >/dev/null 2>&1; then
  COMPOSE=(podman-compose -f "$COMPOSE_FILE")
else
  echo "Install Docker Compose or podman-compose, then rerun this script."
  exit 1
fi

echo "Building and starting the Post-Disaster Analytics Docker demo..."
"${COMPOSE[@]}" up -d --build

echo
echo "Demo is starting:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  RAG:      http://localhost:8000/rag/status"
echo
echo "Useful commands:"
echo "  ${COMPOSE[*]} logs -f"
echo "  ${COMPOSE[*]} down"
