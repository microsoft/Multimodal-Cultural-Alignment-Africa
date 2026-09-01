#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${STORYGEN_DATA_DIR:-$SCRIPT_DIR/data/hausa_profile_data}"

python "$SCRIPT_DIR/main_async_args.py" \
    --project-id e321ff7d-3b01-4a85-b815-241a1b9d6007 \
    --file-path "$DATA_DIR/gemma-3-27b-it_non_independent.csv" \
    --model-type huggingface \
    --model-name "google/gemma-3-27b-it" \
    --batch-size 2 \
    --language en-NG \
    --country "Nigeria" \
    --culture "Hausa" \
    --generate-images \
    --upload-storage
