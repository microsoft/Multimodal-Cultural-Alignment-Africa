#!/usr/bin/env python3
"""CLI for modular multimodal cultural story generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from .pipeline import PipelineConfig, run_pipeline, validate_pipeline
    from .story_generation import DEFAULT_ARC_PATH, DEFAULT_PROMPT_PATH
except ImportError:
    from pipeline import PipelineConfig, run_pipeline, validate_pipeline
    from story_generation import DEFAULT_ARC_PATH, DEFAULT_PROMPT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate grounded stories with modular Azure/TRAPI or Hugging Face "
            "text generation, images, Azure Speech narration, and storage upload."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-id", required=True, help="Project UUID.")
    parser.add_argument("--file-path", required=True, type=Path, help="Input CSV.")
    parser.add_argument(
        "--model-type",
        choices=("azure", "huggingface"),
        default="huggingface",
    )
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent Azure text, search, or speech calls.",
    )
    parser.add_argument("--language", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--culture", required=True)
    parser.add_argument(
        "--grounding",
        choices=("auto", "none", "websearch"),
        default="auto",
        help=(
            "auto searches only EXCELLENT rows on Azure; websearch grounds every "
            "row; none disables retrieval."
        ),
    )
    parser.add_argument("--generate-images", action="store_true")
    parser.add_argument(
        "--image-provider",
        choices=("auto", "azure", "huggingface"),
        default="auto",
        help="auto follows the text provider.",
    )
    parser.add_argument(
        "--image-model",
        "--img-gen-model",
        dest="image_model",
        help="Azure image deployment or Hugging Face Diffusers model.",
    )
    parser.add_argument(
        "--generate-audio",
        "--generate-narration",
        dest="generate_audio",
        action="store_true",
        help="Generate one Azure Speech MP3 narration per paragraph.",
    )
    parser.add_argument("--upload-storage", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        "--max-tokens",
        dest="max_new_tokens",
        type=int,
        default=4096,
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--participant-index", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--story-arc-path", type=Path, default=DEFAULT_ARC_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate CSV, condition handling, and prompts without cloud calls.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.upload_storage and not (
        args.generate_images or args.generate_audio
    ):
        parser.error(
            "--upload-storage requires --generate-images or --generate-audio"
        )
    if args.grounding == "websearch" and args.model_type != "azure":
        parser.error("--grounding websearch requires --model-type azure")
    return args


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        project_id=args.project_id,
        file_path=args.file_path,
        model_type=args.model_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        language=args.language,
        country=args.country,
        culture=args.culture,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        participant_index=args.participant_index,
        output_dir=args.output_dir,
        prompt_path=args.prompt_path,
        story_arc_path=args.story_arc_path,
        grounding=args.grounding,
        generate_images=args.generate_images,
        image_provider=args.image_provider,
        image_model=args.image_model,
        generate_audio=args.generate_audio,
        upload_storage=args.upload_storage,
        overwrite=args.overwrite,
    )


async def async_main() -> int:
    load_dotenv()
    args = parse_args()
    config = config_from_args(args)
    try:
        if args.validate_only:
            print(json.dumps(validate_pipeline(config), indent=2))
            return 0
        project_path, task_count, error_count = await run_pipeline(config)
        if error_count:
            print(
                f"Generated {task_count} tasks with {error_count} errors; "
                f"see the tasks/errors.json file below {project_path.parent}",
                file=sys.stderr,
            )
            return 1
        print(f"Wrote {task_count} tasks to {project_path}")
        return 0
    except (
        FileNotFoundError,
        ValueError,
        IndexError,
        RuntimeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
