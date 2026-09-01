# Multimodal Cultural Alignment Africa

A compact, reproducible multi-agent pipeline for evaluating how well generated
stories and their images align with a target African cultural context.

Multiple language models independently assign a 0-100 score, identify textual
and visual cultural markers, and enter bounded debate rounds when their scores
do not agree. Results are written as JSONL for analysis or model training.

## What is public

This repository contains the evaluation code, prompts, marker taxonomy, tests,
and a synthetic input example. Original annotation exports, annotator
identities, study data, and unpublished generated results are intentionally
excluded. See [DATA.md](DATA.md).

## Repository structure

```text
.
├── AnnotationPlatform/  # Reserved for the human annotation application
├── StoryGen/            # Reserved for the story and image generation pipeline
├── config/              # Evaluator configuration
├── data/results/        # Generated JSONL output (ignored by Git)
├── examples/            # Synthetic evaluator inputs
├── main.py              # Command-line entry point
├── run.sh               # Convenience runner
├── src/                 # Cultural-alignment evaluator
└── tests/               # Deterministic unit tests
```

`AnnotationPlatform` and `StoryGen` currently contain scope and reproducibility
requirements only. Their application code will be added as separate,
privacy-reviewed components.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/microsoft/Multimodal-Cultural-Alignment-Africa.git
cd Multimodal-Cultural-Alignment-Africa

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The default endpoint is OpenAI, but any
OpenAI-compatible chat-completions endpoint can be selected with
`OPENAI_BASE_URL`. Set `JUDGE_MODELS` to a comma-separated list of models
available from that endpoint.

For an editable development installation with test dependencies:

```bash
pip install -e ".[test]"
```

## Run the example

```bash
python main.py --config multimodal --stories examples/sample_stories.json
```

Use `--config text` to exclude persona and image context. Use `--overwrite` to
rerun story IDs already present in `data/results/`.

The convenience script accepts the same primary inputs:

```bash
./run.sh multimodal examples/sample_stories.json --overwrite
```

## Input schema

The input file is a JSON array. Each record requires:

```json
{
  "id": "unique-story-id",
  "culture": "target culture",
  "question": "the question answered by the story",
  "text": "full story text",
  "paragraphs": ["paragraph one", "paragraph two"],
  "images": ["relative/path.png", "https://example.org/image.png"],
  "person": {
    "name": "optional persona name",
    "location": "optional location",
    "relevant_context": {"diet": "vegetarian"}
  },
  "metadata": {
    "image_prompts": ["description for image one", "description for image two"]
  }
}
```

Local image paths are resolved relative to the input JSON file and encoded as
data URLs before being sent to the model; their filesystem paths are not added
to prompts. HTTP(S) image URLs are passed to the configured provider. Models
must support image input when `images` is nonempty.

## Output

Each completed story is appended to:

```text
data/results/<configuration>_<culture>_results.jsonl
```

Records include the final score, reasoning, deduplicated textual and visual
markers, individual judge scores, debate critiques, and consensus metadata.

## Reproducing a run

1. Record the repository commit.
2. Preserve the input JSON and referenced local images.
3. Record `JUDGE_MODELS`, `OPENAI_BASE_URL`, threshold, round count, and model
   provider versions.
4. Start with an empty `data/results/` directory or pass `--overwrite` to
   replace records having the same story ID.
5. Run the command and archive the resulting JSONL with the configuration.

Hosted model outputs can vary across time even with identical inputs. For
stronger repeatability, use versioned model deployments or a fixed local
OpenAI-compatible inference server.

## Tests

```bash
pytest -q
```

## Responsible AI and ethical use

Cultural alignment scores are decision-support signals, not objective measures
of a culture or substitutes for community experts. Validate the rubric with
people from the represented communities, document model and dataset
limitations, and inspect disagreement rather than reporting only an average.

## Contributing

This project welcomes contributions and suggestions. Most contributions
require agreement to a Microsoft Contributor License Agreement. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community standards.

## Support and security

Community support is available through GitHub Issues as described in
[SUPPORT.md](SUPPORT.md). Do not report security vulnerabilities in public
issues; follow [SECURITY.md](SECURITY.md).

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT License](LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or
services. Authorized use of Microsoft trademarks or logos is subject to and
must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must
not cause confusion or imply Microsoft sponsorship. Any use of third-party
trademarks or logos is subject to those third parties' policies.
