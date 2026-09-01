"""Pipeline orchestration independent of provider implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .grounding import WebSearchGrounder
    from .media import AzureSpeechAudioGenerator, create_image_generator
    from .providers import create_text_generator
    from .storage import SupabaseUploader
    from .story_generation import (
        DEFAULT_ARC_PATH,
        DEFAULT_PROMPT_PATH,
        GenerationOptions,
        build_prompt,
        compile_project,
        create_task,
        load_profiles,
        parse_generated_story,
        prepare_profile_for_condition,
        profile_identifier,
        resolve_story_arc_path,
        safe_slug,
        validate_project_id,
        write_json,
    )
except ImportError:
    from grounding import WebSearchGrounder
    from media import AzureSpeechAudioGenerator, create_image_generator
    from providers import create_text_generator
    from storage import SupabaseUploader
    from story_generation import (
        DEFAULT_ARC_PATH,
        DEFAULT_PROMPT_PATH,
        GenerationOptions,
        build_prompt,
        compile_project,
        create_task,
        load_profiles,
        parse_generated_story,
        prepare_profile_for_condition,
        profile_identifier,
        resolve_story_arc_path,
        safe_slug,
        validate_project_id,
        write_json,
    )

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable


@dataclass(frozen=True)
class PipelineConfig:
    project_id: str
    file_path: Path
    model_type: str
    model_name: str
    batch_size: int
    max_concurrent: int
    language: str
    country: str
    culture: str
    max_new_tokens: int
    temperature: float
    seed: int
    participant_index: int | None
    output_dir: Path
    prompt_path: Path = DEFAULT_PROMPT_PATH
    story_arc_path: Path = DEFAULT_ARC_PATH
    grounding: str = "auto"
    generate_images: bool = False
    image_provider: str = "auto"
    image_model: str | None = None
    generate_audio: bool = False
    upload_storage: bool = False
    overwrite: bool = False


def _load_selected_profiles(
    config: PipelineConfig,
) -> tuple[list[dict[str, str]], list[str], list[int]]:
    profiles = load_profiles(config.file_path)
    indexed_profiles = list(enumerate(profiles, start=1))
    if config.participant_index is not None:
        if config.participant_index < 0 or config.participant_index >= len(profiles):
            raise IndexError(
                f"--participant-index must be between 0 and {len(profiles) - 1}"
            )
        indexed_profiles = [indexed_profiles[config.participant_index]]

    identifiers = [
        profile_identifier(profile, original_index)
        for original_index, profile in indexed_profiles
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Input CSV contains duplicate profile identifiers")
    return (
        [profile for _, profile in indexed_profiles],
        identifiers,
        [original_index for original_index, _ in indexed_profiles],
    )


def _prompt_context(
    profile: dict[str, str], config: PipelineConfig
) -> tuple[str, str, str | None]:
    if profile.get("label", "").upper() == "BAD":
        return (
            "Unspecified",
            "Unspecified",
            "Do not infer culture, religion, province, or country-specific details.",
        )
    return config.country, config.culture, None


def validate_pipeline(config: PipelineConfig) -> dict[str, Any]:
    validate_project_id(config.project_id)
    profiles, _, original_indices = _load_selected_profiles(config)
    prompts = []
    grounded = 0
    for profile, original_index in zip(profiles, original_indices):
        prepared, _ = prepare_profile_for_condition(
            profile, config.seed + original_index
        )
        if config.grounding == "websearch" or (
            config.grounding == "auto"
            and config.model_type == "azure"
            and prepared.get("label", "").upper() == "EXCELLENT"
        ):
            grounded += 1
        prompt_country, prompt_culture, culture_support = _prompt_context(
            prepared, config
        )
        prompts.append(
            build_prompt(
                prepared,
                language=config.language,
                country=prompt_country,
                culture=prompt_culture,
                culture_support=culture_support,
                prompt_path=config.prompt_path,
                arc_path=resolve_story_arc_path(
                    prepared, config.story_arc_path
                ),
            )
        )
    return {
        "profiles": len(profiles),
        "prompt_characters": [len(prompt) for prompt in prompts],
        "model_type": config.model_type,
        "model": config.model_name,
        "grounding_mode": config.grounding,
        "profiles_eligible_for_auto_grounding": grounded,
        "generate_images": config.generate_images,
        "generate_audio": config.generate_audio,
    }


async def _upload_media(
    uploader: SupabaseUploader,
    paths: list[Path],
    *,
    media_type: str,
    project_id: str,
    model_slug: str,
    task_id: str,
) -> list[str]:
    references = []
    for path in tqdm(
        paths,
        desc=f"Uploading {media_type} for {task_id}",
        unit="file",
    ):
        object_name = f"{project_id}/{model_slug}/{task_id}/{path.name}"
        references.append(
            await asyncio.to_thread(
                uploader.upload,
                path,
                object_name,
                media_type=media_type,
            )
        )
    return references


async def run_pipeline(config: PipelineConfig) -> tuple[Path, int, int]:
    project_id = validate_project_id(config.project_id)
    profiles, identifiers, original_indices = _load_selected_profiles(config)
    model_slug = safe_slug(config.model_name)
    task_dir = config.output_dir / project_id / model_slug / "tasks"
    image_dir = config.output_dir / project_id / model_slug / "images"
    audio_dir = config.output_dir / project_id / model_slug / "audio"

    selected_profiles = []
    selected_prepared = []
    selected_fields = []
    selected_ids = []
    selected_original_indices = []
    for profile, task_id, original_index in zip(
        profiles, identifiers, original_indices
    ):
        task_path = task_dir / f"{task_id}.json"
        if task_path.exists() and not config.overwrite:
            continue
        if config.overwrite:
            task_path.unlink(missing_ok=True)
        prepared, fields = prepare_profile_for_condition(
            profile, config.seed + original_index
        )
        selected_profiles.append(profile)
        selected_prepared.append(prepared)
        selected_fields.append(fields)
        selected_ids.append(task_id)
        selected_original_indices.append(original_index)

    errors: list[dict[str, str]] = []
    if selected_profiles:
        text_generator = create_text_generator(
            config.model_type,
            config.model_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            max_concurrent=config.max_concurrent,
        )
        grounding_mode = config.grounding
        if grounding_mode == "auto" and config.model_type != "azure":
            grounding_mode = "none"
        if grounding_mode != "none" and not hasattr(text_generator, "search_web"):
            raise ValueError("Web-search grounding requires --model-type azure")

        grounder = WebSearchGrounder(
            text_generator,
            config.output_dir / project_id / model_slug / "grounding-cache.json",
            mode=grounding_mode,
            country=config.country,
            culture=config.culture,
        )
        grounding_results = await grounder.ground_profiles(selected_prepared)
        ready_indices = []
        prompts = []
        for index, grounding_result in enumerate(grounding_results):
            if grounding_result.error:
                errors.append(
                    {
                        "source_id": selected_ids[index],
                        "stage": "grounding",
                        "error": grounding_result.error,
                    }
                )
                continue
            profile = selected_prepared[index]
            prompt_country, prompt_culture, culture_support = _prompt_context(
                profile, config
            )
            prompts.append(
                build_prompt(
                    profile,
                    language=config.language,
                    country=prompt_country,
                    culture=prompt_culture,
                    medical_ground_truth=grounding_result.context,
                    culture_support=culture_support,
                    prompt_path=config.prompt_path,
                    arc_path=resolve_story_arc_path(
                        profile, config.story_arc_path
                    ),
                )
            )
            ready_indices.append(index)

        text_results = await text_generator.generate(prompts, config.batch_size)
        if len(text_results) != len(prompts):
            raise RuntimeError("Text provider returned fewer responses than prompts")

        generated_rows = []
        for index, text_result in zip(ready_indices, text_results):
            if text_result.error:
                errors.append(
                    {
                        "source_id": selected_ids[index],
                        "stage": "text",
                        "error": text_result.error,
                    }
                )
                continue
            if text_result.text is None:
                errors.append(
                    {
                        "source_id": selected_ids[index],
                        "stage": "text",
                        "error": "Text provider returned no content",
                    }
                )
                continue
            generated_rows.append(
                (
                    index,
                    text_result.text,
                    grounding_results[index].context,
                    selected_original_indices[index],
                )
            )

        image_generator = None
        if generated_rows and config.generate_images:
            image_provider = config.image_provider
            if image_provider == "auto":
                image_provider = (
                    "azure" if config.model_type == "azure" else "huggingface"
                )
            image_model = config.image_model or (
                "gpt-image-1"
                if image_provider == "azure"
                else "black-forest-labs/FLUX.1-schnell"
            )
            image_generator = create_image_generator(image_provider, image_model)
        audio_generator = (
            AzureSpeechAudioGenerator(config.max_concurrent)
            if generated_rows and config.generate_audio
            else None
        )
        uploader = (
            SupabaseUploader()
            if generated_rows and config.upload_storage
            else None
        )
        options = GenerationOptions(
            project_id=project_id,
            model_type=config.model_type,
            model_name=config.model_name,
            language=config.language,
            country=config.country,
            culture=config.culture,
            output_dir=config.output_dir,
            seed=config.seed,
            overwrite=config.overwrite,
        )

        for index, response, grounding_context, original_index in generated_rows:
            source_profile = selected_profiles[index]
            prepared_profile = selected_prepared[index]
            fields = selected_fields[index]
            task_id = selected_ids[index]
            task_path = task_dir / f"{task_id}.json"
            try:
                story = parse_generated_story(
                    response, require_audio_prompts=config.generate_audio
                )
                media_tasks = []
                media_names = []
                if image_generator:
                    media_names.append("images")
                    media_tasks.append(
                        image_generator.generate(
                            story["image_prompts"],
                            image_dir / task_id,
                            task_id,
                            config.seed + original_index * 100,
                        )
                    )
                if audio_generator:
                    media_names.append("audio")
                    media_tasks.append(
                        audio_generator.generate(
                            story["paragraphs"],
                            audio_dir / task_id,
                            task_id,
                            language=config.language,
                            gender=prepared_profile.get("gender", ""),
                        )
                    )
                media_results = (
                    await asyncio.gather(*media_tasks, return_exceptions=True)
                    if media_tasks
                    else []
                )
                generated = {}
                media_errors = {}
                for name, result in zip(media_names, media_results):
                    if isinstance(result, Exception):
                        media_errors[name] = str(result)
                        errors.append(
                            {
                                "source_id": task_id,
                                "stage": name,
                                "error": str(result),
                            }
                        )
                    else:
                        generated[name] = result
                image_paths = generated.get("images", [])
                audio_paths = generated.get("audio", [])

                image_references = [
                    str(path.relative_to(config.output_dir))
                    for path in image_paths
                ]
                audio_references = [
                    str(path.relative_to(config.output_dir))
                    for path in audio_paths
                ]
                if uploader:
                    try:
                        image_references = await _upload_media(
                            uploader,
                            image_paths,
                            media_type="image",
                            project_id=project_id,
                            model_slug=model_slug,
                            task_id=task_id,
                        )
                    except Exception as exc:
                        media_errors["image_upload"] = str(exc)
                        errors.append(
                            {
                                "source_id": task_id,
                                "stage": "image_upload",
                                "error": str(exc),
                            }
                        )
                    try:
                        audio_references = await _upload_media(
                            uploader,
                            audio_paths,
                            media_type="audio",
                            project_id=project_id,
                            model_slug=model_slug,
                            task_id=task_id,
                        )
                    except Exception as exc:
                        media_errors["audio_upload"] = str(exc)
                        errors.append(
                            {
                                "source_id": task_id,
                                "stage": "audio_upload",
                                "error": str(exc),
                            }
                        )

                task = create_task(
                    source_profile,
                    story,
                    task_id=task_id,
                    image_references=image_references,
                    audio_references=audio_references,
                    grounding_context=grounding_context,
                    selected_profile_fields=fields,
                    media_errors=media_errors,
                    options=options,
                )
                write_json(task_path, task)
            except Exception as exc:
                errors.append(
                    {
                        "source_id": task_id,
                        "stage": "assembly",
                        "error": str(exc),
                    }
                )

    project = compile_project(
        task_dir,
        project_id=project_id,
        model_name=config.model_name,
        language=config.language,
        country=config.country,
        culture=config.culture,
    )
    project_path = config.output_dir / project_id / f"{model_slug}.json"
    write_json(project_path, project)
    error_path = task_dir / "errors.json"
    if errors:
        write_json(error_path, errors)
    else:
        error_path.unlink(missing_ok=True)
    return project_path, len(project["tasks"]), len(errors)
