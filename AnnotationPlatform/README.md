# Story generation in `AnnotationPlatform`

This component contains the minimal public story-generation pipeline used to
prepare multimodal stories for annotation. The directory name is retained to
match the research workflow: generated stories are the inputs to the annotation
platform.

The public version deliberately excludes participant data, annotation exports,
generated media, credentials, internal Microsoft endpoints, RAG caches, and
experimental scripts. Use only profile data that you are authorized to process.

## What is included

- `main_async_args.py`: command-line entry point.
- `story_generation.py`: CSV normalization, prompt construction, Hugging Face
  inference, strict JSON parsing, local FLUX generation, Supabase upload, and
  safe result assembly.
- `prompts/`: the story prompt and classic-change narrative arc.
- `examples/sample_profiles.csv`: a fully synthetic input row.
- `.env.example`: environment variables for gated models and optional storage.

Cloud clients and large models are loaded lazily. `--help` and
`--validate-only` therefore work without Azure, Supabase, CUDA, or model
credentials.

## Installation

Create an isolated environment from this directory:

```bash
cd AnnotationPlatform
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `HF_TOKEN` in `.env` after accepting any access terms associated with the
selected Hugging Face models. Do not commit `.env`.

## Public smoke test

Validate the synthetic CSV and render its prompt without downloading a model:

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

Remove `--validate-only` to run text generation. Add `--generate-images` only
on a CUDA host with enough memory for the configured FLUX model.

## Research reproduction command

The controlled profile CSVs are not distributed in this public repository.
Place an authorized file at the expected path, then define:

```bash
filename=xichangana_ronga
language=pt-PT
country=Mozambique
culture=Xichangana
export filename language country culture
```

Run the requested configuration:

```bash
python main_async_args.py \
    --project-id e321ff7d-3b01-4a85-b815-241a1b9d6007 \
    --file-path data/$filename"_profile_data/gemma-3-4b-it_non_independent.csv" \
    --model-type huggingface \
    --model-name "google/gemma-3-27b-it" \
    --batch-size 2 \
    --language $language \
    --country $country \
    --culture $culture \
    --generate-images \
    --upload-storage
```

The equivalent checked-in launcher is `main_async_hf.sh`. It requires the four
shell variables above.

### Arguments used by this command

| Argument | Meaning |
|---|---|
| `--project-id` | Stable UUID used to namespace results and task IDs. |
| `--file-path` | Authorized participant/profile CSV. |
| `--model-name` | Hugging Face text-generation model. |
| `--batch-size` | Number of prompts sent through each model batch. |
| `--language` | Requested output language, such as `en` or `pt-PT`. |
| `--country` | Country context supplied to the prompt and output metadata. |
| `--culture` | Cultural context supplied to the prompt and output metadata. |
| `--generate-images` | Generate four local images with Diffusers FLUX. |
| `--upload-storage` | Upload those images to configured Supabase Storage. |

The default image model is
`black-forest-labs/FLUX.1-schnell`; override it with `--image-model`.
`--seed` controls image seeds, and input rows are processed in deterministic CSV
order. Existing task files are skipped unless `--overwrite` is supplied.

## Supabase configuration

`--upload-storage` requires:

```dotenv
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-or-upload-key
SUPABASE_IMAGE_BUCKET=images
```

The target bucket must exist. If public URLs are required by the annotation
platform, configure that bucket as public. Without `--upload-storage`, generated
images remain under `results/<project-id>/<model>/images/` and relative paths
are written to the project JSON.

## Output

The command writes:

```text
results/<project-id>/<model-slug>.json
results/<project-id>/<model-slug>/tasks/<source-id>.json
results/<project-id>/<model-slug>/images/<source-id>/*.png
```

Model output must contain all four paragraphs and all four image prompts.
Malformed or incomplete output is reported explicitly; the pipeline does not
replace failures with fabricated success-shaped stories.

## Privacy and safety

Profile fields are sent to the selected model provider and are also present in
the generated task JSON for traceability. Remove direct identifiers and obtain
all required consent and approvals before processing real profile data. Review
generated medical and cultural content before participant-facing use.
