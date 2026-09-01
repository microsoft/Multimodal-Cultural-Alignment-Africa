"""Reproducible story generation with optional local images and storage upload."""

from __future__ import annotations

import csv
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError:  # Allows lightweight validation before optional dependencies.
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


COMPONENT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = COMPONENT_DIR / "prompts" / "story_prompt.txt"
DEFAULT_ARC_PATH = COMPONENT_DIR / "prompts" / "arc_types" / "classic_change.txt"

SYSTEM_PROMPT = """You generate culturally grounded health stories from an authorized
participant profile. Return only one JSON object with these keys:
- "paragraphs": an object containing keys "0", "1", "2", "3", and "4".
  Key "0" lists the profile fields used. Keys "1"-"4" contain the story.
- "image_prompts": an object containing keys "1", "2", "3", and "4".
  Each value is a self-contained visual prompt for its corresponding paragraph.
- "audio_prompts": an object containing keys "1", "2", "3", and "4".

Do not invent medical facts, diagnoses, or participant attributes. Avoid stereotypes.
Keep the same named people and physical descriptions across all image prompts."""


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "study_number": ("study_number", "Study Number"),
    "name": ("name", "What is your preferred name?"),
    "province": ("province", "Which province do you live in? (select 1)"),
    "religion": ("religion", "What is your religion?"),
    "gender": ("gender", "Gender  (select 1)"),
    "age": ("age", "Age (select 1)"),
    "community": (
        "community",
        "How would you describe your local community or area?    (select all that apply)",
    ),
    "food_access": (
        "food_access",
        "How would you describe access to food in your local community or area?    "
        "(select all that apply)",
    ),
    "disability": (
        "disability",
        "Do you have a disability? (select all that apply)",
    ),
    "occupation": ("occupation", "What is your current occupation?"),
    "daily_routine": (
        "daily_routine",
        "What best describes your daily routine right now?",
    ),
    "hobbies": (
        "hobbies",
        "What hobbies or interests do you enjoy regularly? (select all that apply)",
    ),
    "important_routines": (
        "important_routines",
        "Which activities or routines are most important to you?",
    ),
    "other_health_conditions": (
        "other_health_conditions",
        "Do you currently have other health conditions besides diabetes? "
        "(check all that apply)",
    ),
    "diabetes_treatment": (
        "diabetes_treatment",
        "Are you currently taking medication or following a treatment for diabetes?",
    ),
    "healthcare_resources": (
        "healthcare_resources",
        "What kind of healthcare resources do you regularly access or use?",
    ),
    "healthcare_access": (
        "healthcare_access",
        "Do you have regular access to healthcare services?",
    ),
    "healthcare_frequency": (
        "healthcare_frequency",
        "How often do you access healthcare services?  (choose 1)",
    ),
    "culture": ("culture", "Culture (select all that apply)"),
    "type_2_diabetes": (
        "type_2_diabetes",
        "Are you currently living with Type 2 Diabetes",
    ),
    "smoking": ("smoking", "Do you smoke tobacco products?"),
    "smoking_frequency": ("smoking_frequency", "tobacco_smoking_freq"),
    "alcohol": ("alcohol", "Do you consume alcohol beverages?"),
    "alcohol_frequency": (
        "alcohol_frequency",
        '"If yes, how often?  (choose 1)"',
        "If yes, how often?  (choose 1)",
    ),
    "physical_activity": (
        "physical_activity",
        "How many days per week do you participate in moderate to vigorous physical "
        "activity?  (choose 1)",
    ),
    "diet": (
        "diet",
        "Select any dietary preferences or restrictions you may have (choose 1)",
    ),
    "support_network": (
        "support_network",
        "Who forms part of your regular support network?  (check all that apply)",
    ),
    "health_goals": (
        "health_goals",
        "What are your main personal health goals? (You may select more than one)",
    ),
    "information_preference": (
        "information_preference",
        "How do you prefer to receive information or stories about your health?  "
        "(check all that apply)",
    ),
    "story_tone": (
        "story_tone",
        "How would you like the story to sound? (Select 1)",
    ),
    "question": ("question", "Question"),
    "category": ("category", "Category"),
    "label": ("label", "Label"),
    "story_arc": ("story_arc", "Story arc"),
    "type": ("type", "Type"),
    "parent": ("parent", "Parent"),
}


QUESTION_TRANSLATIONS = {
    "What can I eat and drink to help me manage my diabetes?": (
        "O que posso comer e beber para me ajudar a controlar a diabetes?"
    ),
    "How can I be more active and manage my weight with diabetes?": (
        "Como posso ser mais ativo e controlar o meu peso com diabetes?"
    ),
    "How can I manage my blood sugar levels effectively?": (
        "Como posso controlar eficazmente os meus níveis de açúcar no sangue?"
    ),
}


@dataclass(frozen=True)
class GenerationOptions:
    project_id: str
    model_name: str
    language: str
    country: str
    culture: str
    output_dir: Path
    seed: int = 42
    overwrite: bool = False


def _first_value(row: Mapping[str, Any], aliases: Iterable[str]) -> str:
    for alias in aliases:
        value = row.get(alias)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_profile(row: Mapping[str, Any]) -> dict[str, str]:
    """Convert supported public or study CSV headers to stable field names."""
    profile = {
        field: _first_value(row, aliases)
        for field, aliases in FIELD_ALIASES.items()
    }
    return {key: value for key, value in profile.items() if value}


def load_profiles(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile CSV does not exist: {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Profile CSV has no header: {path}")
        profiles = [normalize_profile(row) for row in reader]

    if not profiles:
        raise ValueError(f"Profile CSV has no data rows: {path}")
    for index, profile in enumerate(profiles, start=2):
        if "question" not in profile:
            raise ValueError(f"CSV row {index} has no Question value")
    return profiles


def profile_identifier(profile: Mapping[str, str], row_index: int) -> str:
    raw = profile.get("study_number") or f"profile-{row_index:04d}"
    slug = safe_slug(raw)
    if not slug:
        raise ValueError(f"Could not derive an identifier for profile row {row_index}")
    return slug


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return slug.strip("._-")[:120]


def validate_project_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise ValueError("--project-id must be a valid UUID") from exc


def translated_question(question: str, language: str) -> str:
    if language.lower().startswith("pt"):
        return QUESTION_TRANSLATIONS.get(question, question)
    return question


def build_prompt(
    profile: Mapping[str, str],
    *,
    language: str,
    country: str,
    culture: str,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    arc_path: Path = DEFAULT_ARC_PATH,
) -> str:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template does not exist: {prompt_path}")
    if not arc_path.is_file():
        raise FileNotFoundError(f"Story arc template does not exist: {arc_path}")

    template = prompt_path.read_text(encoding="utf-8")
    story_arc = arc_path.read_text(encoding="utf-8").format(max_word_count=180)
    context = dict(profile)
    context.setdefault("culture", culture)
    context["country"] = country
    question = translated_question(profile["question"], language)

    values = {
        "px_name": profile.get("name", "the participant"),
        "persona_needs": question,
        "medical_ground_truth": "",
        "rag_clean_participant_profile": json.dumps(
            context, ensure_ascii=False, indent=2
        ),
        "culture_support": (
            f"Use culturally plausible details for {culture} in {country}; do not "
            "invent unsupported participant attributes."
        ),
        "language": language,
        "language_level": "Use clear, accessible language and short sentences.",
        "px_output_tone": profile.get("story_tone", "warm and practical"),
        "narrative_progression": story_arc,
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Prompt template contains an unsupported placeholder: {exc}"
        ) from exc


def extract_json_object(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    return text[start : end + 1]


def parse_generated_story(text: str) -> dict[str, dict[str, str]]:
    candidate = extract_json_object(text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
        except ImportError as exc:
            raise RuntimeError(
                "Invalid model JSON and json-repair is not installed"
            ) from exc
        value = json.loads(repair_json(candidate))

    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")

    normalized: dict[str, dict[str, str]] = {}
    for section in ("paragraphs", "image_prompts", "audio_prompts"):
        raw_section = value.get(section)
        if not isinstance(raw_section, dict):
            raise ValueError(f"Model response is missing object '{section}'")
        normalized[section] = {
            str(key): str(item).strip()
            for key, item in raw_section.items()
            if str(item).strip()
        }

    missing_paragraphs = [
        key for key in ("1", "2", "3", "4") if key not in normalized["paragraphs"]
    ]
    missing_images = [
        key for key in ("1", "2", "3", "4") if key not in normalized["image_prompts"]
    ]
    if missing_paragraphs or missing_images:
        raise ValueError(
            "Model response is incomplete: "
            f"missing paragraphs={missing_paragraphs}, image_prompts={missing_images}"
        )
    return normalized


class HuggingFaceStoryGenerator:
    """Lazy Hugging Face text-generation pipeline."""

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dependencies are missing; install requirements.txt"
            ) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        kwargs: dict[str, Any] = {
            "model": model_name,
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        if token:
            kwargs["token"] = token

        self.pipeline = pipeline("text-generation", **kwargs)
        tokenizer = self.pipeline.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompts: Sequence[str], batch_size: int) -> list[str]:
        responses: list[str] = []
        starts = range(0, len(prompts), batch_size)
        for start in tqdm(
            starts,
            total=(len(prompts) + batch_size - 1) // batch_size,
            desc="Generating stories",
            unit="batch",
        ):
            batch = [
                f"{SYSTEM_PROMPT}\n\n{prompt}" for prompt in prompts[start : start + batch_size]
            ]
            generation_args: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False,
                "batch_size": batch_size,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                generation_args["temperature"] = self.temperature
            outputs = self.pipeline(batch, **generation_args)
            for output in outputs:
                item = output[0] if isinstance(output, list) else output
                generated_text = item.get("generated_text")
                if not isinstance(generated_text, str):
                    raise ValueError("Hugging Face returned an unexpected output shape")
                responses.append(generated_text)
        return responses


class FluxImageGenerator:
    """Lazy local FLUX generator using Diffusers."""

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Image dependencies are missing; install requirements.txt"
            ) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
        if token:
            kwargs["token"] = token
        self.torch = torch
        self.pipeline = FluxPipeline.from_pretrained(model_name, **kwargs)
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
        else:
            raise RuntimeError("Local FLUX image generation requires a CUDA GPU")

    def generate(
        self,
        prompts: Mapping[str, str],
        output_dir: Path,
        base_name: str,
        seed: int,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        images: list[Path] = []
        ordered = [(key, prompts[key]) for key in ("1", "2", "3", "4")]
        for offset, (paragraph, prompt) in enumerate(
            tqdm(ordered, desc=f"Generating images for {base_name}", unit="image")
        ):
            generator = self.torch.Generator("cpu").manual_seed(seed + offset)
            image = self.pipeline(
                prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=generator,
            ).images[0]
            path = output_dir / f"{base_name}_{paragraph}.png"
            image.save(path)
            images.append(path)
        return images


class SupabaseUploader:
    """Lazy Supabase Storage uploader."""

    def __init__(self) -> None:
        try:
            from supabase import create_client
        except ImportError as exc:
            raise RuntimeError(
                "Supabase dependencies are missing; install requirements.txt"
            ) from exc

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "--upload-storage requires SUPABASE_URL and SUPABASE_KEY"
            )
        self.client = create_client(url, key)
        self.bucket = os.getenv("SUPABASE_IMAGE_BUCKET", "images")

    def upload_image(self, path: Path, object_name: str) -> str:
        with path.open("rb") as handle:
            self.client.storage.from_(self.bucket).upload(
                object_name,
                handle,
                file_options={"content-type": "image/png", "upsert": "true"},
            )
        return self.client.storage.from_(self.bucket).get_public_url(object_name)


def create_task(
    profile: Mapping[str, str],
    story: Mapping[str, Mapping[str, str]],
    *,
    task_id: str,
    image_references: Sequence[str],
    options: GenerationOptions,
) -> dict[str, Any]:
    paragraphs = [story["paragraphs"][str(index)] for index in range(1, 5)]
    return {
        "id": str(uuid.uuid5(uuid.UUID(options.project_id), task_id)),
        "data": {
            "title": f"Story for {task_id}",
            "description": "",
            "images": list(image_references),
            "audio": [],
            "paragraphs": paragraphs,
            "language": options.language,
            "country": options.country,
            "culture": options.culture,
            "profile": dict(profile),
            "model": options.model_name,
        },
        "metadata": {
            "source_id": task_id,
            "image_prompts": dict(story["image_prompts"]),
            "audio_prompts": dict(story["audio_prompts"]),
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compile_project(
    task_dir: Path,
    *,
    project_id: str,
    model_name: str,
    language: str,
    country: str,
    culture: str,
) -> dict[str, Any]:
    tasks = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(task_dir.glob("*.json"))
        if path.name != "errors.json"
    ]
    return {
        "id": project_id,
        "title": f"Cultural alignment stories: {culture}",
        "description": (
            f"Stories generated with {model_name} for {culture} in {country}."
        ),
        "language": language,
        "tasks": tasks,
    }
