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

## Associated publication

This repository accompanies the paper **"Toward Cultural Alignment:
Human-Centered Evaluation of Multimodal AI Stories Across Five African
Communities,"** accepted to **Findings of EMNLP 2026**. The preprint is
available on [arXiv](https://arxiv.org/abs/2608.29209).

If you use this repository, please cite:

```bibtex
@misc{ochieng2026culturalalignmenthumancenteredevaluation,
  title        = {Toward Cultural Alignment: Human-Centered Evaluation of Multimodal AI Stories Across Five African Communities},
  author       = {Millicent Ochieng and Felermino D. M. A. Ali and Elizabeth A. Ankrah and Najeeb Gambo Abdulhamid and Migisha Boyd and Stephanie Nyairo and Mercy Muchai and Samuel Chege Maina and Aditya Vashistha and Anja Thieme and Jacki O'Neill},
  year         = {2026},
  eprint       = {2608.29209},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2608.29209}
}
```

## Repository structure

```text
.
├── StoryGen/            # Modular grounded story and media generation
├── config/              # Evaluator configuration
├── data/images/         # Anonymized story images referenced by judgments
├── data/judgments/      # De-identified human and LLM judgment data
├── data/results/        # Generated JSONL output (ignored by Git)
├── examples/            # Synthetic evaluator inputs
├── main.py              # Command-line entry point
├── run.sh               # Convenience runner
├── scripts/             # Release data de-identification
├── src/                 # Cultural-alignment evaluator
└── tests/               # Deterministic unit tests
```

`StoryGen` contains the cleaned research reproduction CLI, synthetic profile
data, prompt templates, RAG/web search, Azure and Hugging Face providers,
image generation, Azure Speech narration, and optional storage integrations.

## Released evaluation data

De-identified human annotations and matching multi-agent LLM judgments are
available under `data/judgments/` for Hausa, isiXhosa, Kikuyu, Luo, and
Xichangana. The human and LLM files use shared anonymous story IDs.

Annotator names and email addresses, participant names and profile objects,
internal user/submission/study IDs, prompts, search queries, and annotation
timestamps are not included. Linkable external media URLs are also omitted;
synthetic profile attributes used to generate the stories are retained in free
text, while profile objects and direct identifiers remain excluded. Both
judgment files reference the available images through anonymous
repository-relative paths under `data/images/`; media counts and image judgments
remain. See `data/judgments/README.md` for the schema, counts, sanitization
policy, and reproduction command.

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
