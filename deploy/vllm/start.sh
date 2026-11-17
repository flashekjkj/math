#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f .env ]]; then
  if [[ -f deployment-lock.json ]]; then
    echo 'Lock exists but .env is missing. Run freeze.py --refresh intentionally to regenerate local settings.' >&2
    exit 1
  fi
  python3 freeze.py
fi
python3 freeze.py --verify
# Existing manual container on port 8001 must be stopped before starting Compose.
docker compose --env-file .env -f compose.yaml up -d
