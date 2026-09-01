"""Reproducible story generation with optional local images and storage upload."""

from __future__ import annotations

import csv
import json
import random
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


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
    "generated_queries": (
        "generated_queries",
        "Generated queries",
        "Generated Queries",
        "Consultas Geradas",
    ),
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
    model_type: str
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
        profiles = []
        skipped_rows = []
        for row_number, row in enumerate(reader, start=2):
            profile = normalize_profile(row)
            if "question" not in profile:
                skipped_rows.append(row_number)
                continue
            profiles.append(profile)

    if not profiles:
        raise ValueError(f"Profile CSV has no rows with a Question value: {path}")
    if skipped_rows:
        warnings.warn(
            "Skipped CSV rows without a Question value: "
            + ", ".join(map(str, skipped_rows)),
            stacklevel=2,
        )
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


def resolve_story_arc_path(
    profile: Mapping[str, str], fallback: Path = DEFAULT_ARC_PATH
) -> Path:
    value = profile.get("story_arc", "").strip().lower()
    if not value:
        return fallback
    filename = re.sub(r"[^a-z0-9]+", "_", value).strip("_") + ".txt"
    candidate = DEFAULT_ARC_PATH.parent / filename
    if not candidate.is_file():
        warnings.warn(
            f"Unknown story arc '{value}'; using {fallback.name}",
            stacklevel=2,
        )
        return fallback
    return candidate


def prepare_profile_for_condition(
    profile: Mapping[str, str], seed: int
) -> tuple[dict[str, str], list[str]]:
    """Apply the study condition without mutating the source profile."""
    prepared = dict(profile)
    if prepared.get("label", "").upper() != "BAD":
        return prepared, sorted(prepared)

    required = {
        "study_number",
        "question",
        "category",
        "gender",
        "story_tone",
        "label",
        "generated_queries",
        "type",
        "story_arc",
        "parent",
        "name",
    }
    important = {
        "age",
        "culture",
        "religion",
        "occupation",
        "type_2_diabetes",
        "diabetes_treatment",
        "other_health_conditions",
    }
    available = set(prepared)
    rng = random.Random(seed)

    def sample_half(fields: set[str]) -> set[str]:
        ordered = sorted(fields)
        count = len(ordered) // 2
        return set(rng.sample(ordered, count)) if count else set()

    optional = available - required - important
    selected = (
        (required & available)
        | sample_half(important & available)
        | sample_half(optional)
    )
    filtered = {key: prepared[key] for key in prepared if key in selected}
    for field in ("culture", "province", "religion"):
        filtered[field] = "None"
    return filtered, sorted(filtered)


def build_prompt(
    profile: Mapping[str, str],
    *,
    language: str,
    country: str,
    culture: str,
    medical_ground_truth: str = "",
    culture_support: str | None = None,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    arc_path: Path = DEFAULT_ARC_PATH,
) -> str:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template does not exist: {prompt_path}")
    if not arc_path.is_file():
        raise FileNotFoundError(f"Story arc template does not exist: {arc_path}")

    template = prompt_path.read_text(encoding="utf-8")
    story_arc = arc_path.read_text(encoding="utf-8").format(
        max_word_count="180 words maximum.",
        px_name=profile.get("name", "the participant"),
    )
    context = dict(profile)
    context.setdefault("culture", culture)
    context["country"] = country
    question = translated_question(profile["question"], language)

    values = {
        "px_name": profile.get("name", "the participant"),
        "persona_needs": question,
        "medical_ground_truth": medical_ground_truth,
        "rag_clean_participant_profile": json.dumps(
            context, ensure_ascii=False, indent=2
        ),
        "culture_support": culture_support
        or (
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


def parse_generated_story(
    text: str, *, require_audio_prompts: bool = False
) -> dict[str, dict[str, str]]:
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
    missing_audio = (
        [
            key
            for key in ("1", "2", "3", "4")
            if key not in normalized["audio_prompts"]
        ]
        if require_audio_prompts
        else []
    )
    if missing_paragraphs or missing_images or missing_audio:
        raise ValueError(
            "Model response is incomplete: "
            f"missing paragraphs={missing_paragraphs}, "
            f"image_prompts={missing_images}, audio_prompts={missing_audio}"
        )
    return normalized


def create_task(
    profile: Mapping[str, str],
    story: Mapping[str, Mapping[str, str]],
    *,
    task_id: str,
    image_references: Sequence[str],
    audio_references: Sequence[str],
    grounding_context: str,
    selected_profile_fields: Sequence[str],
    media_errors: Mapping[str, str],
    options: GenerationOptions,
) -> dict[str, Any]:
    paragraphs = [story["paragraphs"][str(index)] for index in range(1, 5)]
    return {
        "id": str(uuid.uuid5(uuid.UUID(options.project_id), task_id)),
        "data": {
            "title": f"Story for {task_id}",
            "description": "",
            "images": list(image_references),
            "audio": list(audio_references),
            "paragraphs": paragraphs,
            "language": options.language,
            "country": options.country,
            "culture": options.culture,
            "profile": dict(profile),
            "model": options.model_name,
            "model_type": options.model_type,
        },
        "metadata": {
            "source_id": task_id,
            "image_prompts": dict(story["image_prompts"]),
            "audio_prompts": dict(story["audio_prompts"]),
            "grounding_context": grounding_context,
            "selected_profile_fields": list(selected_profile_fields),
            "media_errors": dict(media_errors),
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
