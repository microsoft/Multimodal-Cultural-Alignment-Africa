# StoryGen

`StoryGen` generates culturally grounded health stories and their annotation
media. It supports:

- local Hugging Face text generation;
- Azure OpenAI or an authorized Azure OpenAI-compatible TRAPI endpoint;
- cached web-search retrieval for RAG grounding;
- local Diffusers FLUX or Azure image generation;
- Azure Speech narration;
- optional Supabase image and audio upload.

Participant data, credentials, generated media, web-search caches, and internal
service endpoints are not included in this public repository.

## Modular layout

| File | Responsibility |
|---|---|
| `main_async_args.py` | CLI parsing and validation |
| `pipeline.py` | End-to-end orchestration |
| `providers.py` | Hugging Face and Azure/TRAPI text providers |
| `grounding.py` | Query parsing, web search, and persistent RAG cache |
| `media.py` | FLUX/Azure images and Azure Speech narration |
| `storage.py` | Lazy Supabase uploads |
| `story_generation.py` | CSV schema, study conditions, prompts, JSON, output |

Each cloud or model dependency is initialized only when its feature is enabled.
`--help` and `--validate-only` do not require Azure credentials, CUDA, Speech,
Supabase, or downloaded models.

## Installation

```bash
cd StoryGen
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Do not commit `.env`.

## Validate with synthetic data

```bash
python main_async_args.py \
    --project-id e321ff7d-3b01-4a85-b815-241a1b9d6007 \
    --file-path examples/sample_profiles.csv \
    --model-type huggingface \
    --model-name "google/gemma-3-4b-it" \
    --batch-size 1 \
    --language en \
    --country "Synthetic Country" \
    --culture "Synthetic Community" \
    --validate-only
```

## Azure/TRAPI run

Authenticate with `az login`, then configure the endpoint, deployment, API
version, and audience scope in `.env`. Standard Azure OpenAI deployments use
the `AZURE_OPENAI_*` variables. An authorized TRAPI environment can use either
those variables or the supported `TRAPI_*` aliases. No internal endpoint is
hard-coded.

The requested Xichangana command, with audio generation added, is:

```bash
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
```

The checked-in `main_async_azure.sh` runs the same configuration.
Rows without a `Question` value are reported and skipped rather than sent to a
model with an empty task.

### Azure text and web-search settings

```dotenv
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_SCOPE=https://ai.azure.com/.default
AZURE_WEB_SEARCH_TOOL=web_search
```

Set `AZURE_OPENAI_API_KEY` only when key authentication is required. Otherwise,
the client tries Azure CLI authentication first and managed identity second.

`--grounding auto` reproduces the study condition: only rows whose `Label` is
`EXCELLENT` are grounded. Their `Generated queries` field is parsed as JSON,
a Python-style list, or newline/semicolon-separated text. Missing queries are
replaced by one question-and-culture query. Current Azure deployments use the
`web_search` tool. Set `AZURE_WEB_SEARCH_TOOL=web_search_preview` only when an
older authorized endpoint requires the preview contract. Results are cached at:

```text
results/<project-id>/<model-slug>/grounding-cache.json
```

Use `--grounding websearch` to ground every row or `--grounding none` to disable
retrieval. A response is cached only after the API confirms a
`web_search_call`.

### Image settings

With Azure text generation, `--image-provider auto` selects Azure images:

```dotenv
AZURE_IMAGE_ENDPOINT=
AZURE_IMAGE_DEPLOYMENT=gpt-image-1
AZURE_IMAGE_API_VERSION=2025-04-01-preview
AZURE_IMAGE_SCOPE=
```

Blank image endpoint and scope values fall back to the Azure text endpoint and
scope. The first frame is generated normally; later frames edit the first frame
as a reference to improve identity consistency.

For local FLUX instead:

```bash
--image-provider huggingface \
--image-model black-forest-labs/FLUX.1-schnell
```

### Audio settings

`--generate-audio` creates one MP3 narration for each story paragraph:

```dotenv
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_VOICE=
```

The voice override is optional. Deterministic locale/gender defaults are
included for `en-ZA`, `pt-PT`, `en-NG`, and `en-KE`. Audio is generated once,
independently of image generation.

### Storage settings

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=
SUPABASE_IMAGE_BUCKET=images
SUPABASE_AUDIO_BUCKET=audios
```

`--upload-storage` uploads only media that was actually generated. Without it,
relative local paths are written to the output.

## Study conditions

- `BAD`: a deterministic, seeded subset of profile fields is retained and
  culture, province, and religion are masked.
- `GOOD`: the complete normalized profile is used without automatic RAG.
- `EXCELLENT`: the complete profile is used with cached web grounding when
  `--grounding auto` is selected.

Each row's `Story arc` selects the matching template under
`prompts/arc_types/`. Unknown or empty values use the CLI
`--story-arc-path` fallback.

## Output

```text
results/<project-id>/<model-slug>.json
results/<project-id>/<model-slug>/grounding-cache.json
results/<project-id>/<model-slug>/tasks/<source-id>.json
results/<project-id>/<model-slug>/images/<source-id>/*.png
results/<project-id>/<model-slug>/audio/<source-id>/*.mp3
```

Malformed model output and provider failures are recorded explicitly; the
pipeline does not fabricate replacement stories.

## Privacy and responsible use

Profiles and generated search queries are transmitted to the configured
services. Grounding results and source profile fields are retained in local
output for reproducibility. De-identify inputs, obtain required consent and
institutional approval, review provider data-handling terms, and perform human
medical and cultural review before participant-facing use.
