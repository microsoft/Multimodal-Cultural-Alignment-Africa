#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 main.py \
  --config "${1:-multimodal}" \
  --stories "${2:-examples/sample_stories.json}" \
  "${@:3}"
