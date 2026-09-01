#!/usr/bin/env bash
set -euo pipefail

: "${filename:?Set filename, for example xichangana_ronga}"
: "${language:?Set language, for example pt-PT}"
: "${country:?Set country, for example Mozambique}"
: "${culture:?Set culture, for example Xichangana}"

python main_async_args.py \
    --project-id e321ff7d-3b01-4a85-b815-241a1b9d6007 \
    --file-path data/"${filename}_profile_data/gemma-3-4b-it_non_independent.csv" \
    --model-type huggingface \
    --model-name "google/gemma-3-27b-it" \
    --batch-size 2 \
    --language "$language" \
    --country "$country" \
    --culture "$culture" \
    --generate-images \
    --upload-storage
