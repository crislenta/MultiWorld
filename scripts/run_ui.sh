#!/usr/bin/env bash
# Build (if needed) and start the MultiWorld Gradio UI container.
#
# Usage:
#   bash scripts/run_ui.sh                # foreground
#   bash scripts/run_ui.sh up -d          # detached
#   bash scripts/run_ui.sh down           # stop
#   bash scripts/run_ui.sh logs           # tail logs
#   bash scripts/run_ui.sh build          # force rebuild

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pick the right docker-compose binary.
if docker compose version &>/dev/null; then
  COMPOSE=(docker compose)
elif command -v docker-compose &>/dev/null; then
  COMPOSE=(docker-compose)
else
  echo "Neither 'docker compose' nor 'docker-compose' was found. Run scripts/setup_gcp_vm.sh first." >&2
  exit 1
fi

mkdir -p checkpoints models outputs .cache

if [[ $# -eq 0 ]]; then
  "${COMPOSE[@]}" up --build
else
  "${COMPOSE[@]}" "$@"
fi
