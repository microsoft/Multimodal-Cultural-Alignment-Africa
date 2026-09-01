#!/usr/bin/env python3
"""CLI for the public cultural story-generation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from story_generation import (
    DEFAULT_ARC_PATH,
    DEFAULT_PROMPT_PATH,
    FluxImageGenerator,
    GenerationOptions,
    HuggingFaceStoryGenerator,
    SupabaseUploader,
    build_prompt,
    compile_project,
    create_task,
    load_profiles,
    parse_generated_story,
    profile_identifier,
    safe_slug,
    validate_project_id,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate culturally grounded stories from an authorized profile CSV. "
            "Cloud and model clients are initialized only when requested."
        )
    )
    parser.add_argument("--project-id", required=True, help="Project UUID.")
    parser.add_argument("--file-path", required=True, type=Path, help="Input CSV.")
    parser.add_argument(
        "--model-type",
        choices=("huggingface",),
        default="huggingface",
        help="Text-generation backend.",
    )
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--language", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--culture", required=True)
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate four local images per story with Diffusers FLUX.",
    )
    parser.add_argument(
        "--upload-storage",
        action="store_true",
        help="Upload generated images to configured Supabase Storage.",
    )
    parser.add_argument(
        "--image-model",
        "--img-gen-model",
        dest="image_model",
        default="black-forest-labs/FLUX.1-schnell",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
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
        help="Validate CSV and prompts without loading model or cloud clients.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.upload_storage and not args.generate_images:
        parser.error("--upload-storage requires --generate-images")
    return args


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        project_id = validate_project_id(args.project_id)
        profiles = load_profiles(args.file_path)
        if args.participant_index is not None:
            if args.participant_index < 0 or args.participant_index >= len(profiles):
                raise IndexError(
                    f"--participant-index must be between 0 and {len(profiles) - 1}"
                )
            profiles = [profiles[args.participant_index]]

        prompts = [
            build_prompt(
                profile,
                language=args.language,
                country=args.country,
                culture=args.culture,
                prompt_path=args.prompt_path,
                arc_path=args.story_arc_path,
            )
            for profile in profiles
        ]
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "profiles": len(profiles),
                        "prompt_characters": [len(prompt) for prompt in prompts],
                        "model": args.model_name,
                    },
                    indent=2,
                )
            )
            return 0

        options = GenerationOptions(
            project_id=project_id,
            model_name=args.model_name,
            language=args.language,
            country=args.country,
            culture=args.culture,
            output_dir=args.output_dir,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        model_slug = safe_slug(args.model_name)
        task_dir = args.output_dir / project_id / model_slug / "tasks"
        image_dir = args.output_dir / project_id / model_slug / "images"

        selected_profiles: list[dict[str, str]] = []
        selected_prompts: list[str] = []
        selected_ids: list[str] = []
        seen_ids: set[str] = set()
        for index, (profile, prompt) in enumerate(zip(profiles, prompts), start=1):
            task_id = profile_identifier(profile, index)
            if task_id in seen_ids:
                raise ValueError(f"Duplicate profile identifier in CSV: {task_id}")
            seen_ids.add(task_id)
            task_path = task_dir / f"{task_id}.json"
            if task_path.exists() and not args.overwrite:
                continue
            selected_profiles.append(profile)
            selected_prompts.append(prompt)
            selected_ids.append(task_id)

        errors: list[dict[str, str]] = []
        if selected_prompts:
            generator = HuggingFaceStoryGenerator(
                args.model_name,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            responses = generator.generate(selected_prompts, args.batch_size)
            if len(responses) != len(selected_prompts):
                raise RuntimeError(
                    "Hugging Face returned fewer responses than input prompts"
                )
            image_generator = (
                FluxImageGenerator(args.image_model) if args.generate_images else None
            )
            uploader = SupabaseUploader() if args.upload_storage else None

            for offset, (profile, task_id, response) in enumerate(
                zip(selected_profiles, selected_ids, responses)
            ):
                task_path = task_dir / f"{task_id}.json"
                if args.overwrite:
                    task_path.unlink(missing_ok=True)
                try:
                    story = parse_generated_story(response)
                    image_references: list[str] = []
                    if image_generator:
                        image_paths = image_generator.generate(
                            story["image_prompts"],
                            image_dir / task_id,
                            task_id,
                            args.seed + offset * 100,
                        )
                        if uploader:
                            for path in image_paths:
                                object_name = (
                                    f"{project_id}/{model_slug}/{task_id}/{path.name}"
                                )
                                image_references.append(
                                    uploader.upload_image(path, object_name)
                                )
                        else:
                            image_references = [
                                str(path.relative_to(args.output_dir))
                                for path in image_paths
                            ]

                    task = create_task(
                        profile,
                        story,
                        task_id=task_id,
                        image_references=image_references,
                        options=options,
                    )
                    write_json(task_path, task)
                except Exception as exc:
                    errors.append({"source_id": task_id, "error": str(exc)})

        project = compile_project(
            task_dir,
            project_id=project_id,
            model_name=args.model_name,
            language=args.language,
            country=args.country,
            culture=args.culture,
        )
        project_path = args.output_dir / project_id / f"{model_slug}.json"
        write_json(project_path, project)
        if errors:
            write_json(task_dir / "errors.json", errors)
            print(
                f"Generated {len(project['tasks'])} tasks with {len(errors)} errors; "
                f"see {task_dir / 'errors.json'}",
                file=sys.stderr,
            )
            return 1

        (task_dir / "errors.json").unlink(missing_ok=True)
        print(f"Wrote {len(project['tasks'])} tasks to {project_path}")
        return 0
    except (FileNotFoundError, ValueError, IndexError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
