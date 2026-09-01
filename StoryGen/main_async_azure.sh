#!/usr/bin/env bash
set -euo pipefail

python main_async_args.py \
    --project-id e321ff7d-3b01-4a85-b815-241a1b9d6333 \
    --file-path data/xichangana_ronga_profile_data/gpt-4.1_non_independent.csv \
    --model-type azure \
    --model-name "gpt-4.1" \
    --max-concurrent 10 \
    --language pt-PT \
    --country "Mozambique" \
    --culture "Xichangana" \
    --grounding auto \
    --generate-images \
    --generate-audio \
    --upload-storage
