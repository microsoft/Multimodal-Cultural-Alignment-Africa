#!/usr/bin/env python3
"""Create de-identified public human and LLM judgment datasets."""

from __future__ import annotations

import argparse
import functools
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from tqdm import tqdm


DATASET_VERSION = "1.1"
RELEASE_DATE = "2026-10-05"
EMAIL_PATTERN = re.compile(
    r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
)
UUID_PATTERN = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ROW_PATTERN = re.compile(
    r"<tr[^>]*>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*</tr>",
    flags=re.IGNORECASE | re.DOTALL,
)
PRIVATE_ANNOTATION_KEYS = {
    "id",
    "timestamp",
    "userEmail",
    "userId",
    "taskId",
    "submissionTaskId",
    "submissionUserId",
}
NAME_LABELS = {
    "name",
    "nome",
    "preferred name",
    "what is your preferred name?",
}
NON_NAME_VALUES = {
    "anonymous",
    "none",
    "not applicable",
    "prefer not to answer",
    "prefer not to say",
    "participant",
    "unknown",
}
NON_PRIVATE_VALUES = {
    "",
    "n/a",
    "na",
    "no",
    "none",
    "not applicable",
    "unknown",
    "yes",
}

LANGUAGES = {
    "hausa": {
        "language_code": "en-NG",
        "country": "Nigeria",
        "culture": "Hausa",
        "human_file": (
            "project-oficial_-culture-connectedness-evaluation---"
            "nigeria---hausa-export(15).json"
        ),
        "llm_file": "conf_5_hausa_results.jsonl",
    },
    "isixhosa": {
        "language_code": "en-ZA",
        "country": "South Africa",
        "culture": "isiXhosa",
        "human_file": (
            "project-oficial_-culture-connectedness-evaluation---"
            "south-africa---amaxhosa-export(16).json"
        ),
        "llm_file": "conf_5_isixhosa_results.jsonl",
    },
    "kikuyu": {
        "language_code": "en-KE",
        "country": "Kenya",
        "culture": "Kikuyu",
        "human_file": (
            "project-oficial_-culture-connectedness-evaluation---"
            "kenya---kikuyu-export(12).json"
        ),
        "llm_file": "conf_5_kikuyu_results.jsonl",
    },
    "luo": {
        "language_code": "en-KE",
        "country": "Kenya",
        "culture": "Luo",
        "human_file": (
            "project-oficial_-culture-connectedness-evaluation---"
            "kenya---luo-export(12).json"
        ),
        "llm_file": "conf_5_luo_results.jsonl",
    },
    "xichangana": {
        "language_code": "pt-PT",
        "country": "Mozambique",
        "culture": "Xichangana",
        "human_file": (
            "project-oficial_-culture-connectedness-evaluation---"
            "mozambique---xichangana-export(15).json"
        ),
        "llm_file": "conf_5_xichangana_results.jsonl",
    },
}


def _plain_text(value: str) -> str:
    return html.unescape(HTML_TAG_PATTERN.sub("", value)).strip()


def names_from_description(description: str) -> list[str]:
    names = []
    for raw_label, raw_value in HTML_ROW_PATTERN.findall(description or ""):
        label = _plain_text(raw_label).lower()
        if label in NAME_LABELS or "preferred name" in label:
            value = _plain_text(raw_value)
            if value:
                names.append(value)
    return names


def names_from_person(person: Mapping[str, Any]) -> list[str]:
    names = []
    for key, value in person.items():
        normalized = str(key).strip().lower().replace("_", " ")
        if normalized in NAME_LABELS or "preferred name" in normalized:
            text = str(value).strip()
            if text:
                names.append(text)
    return names


def private_values_from_person(person: Mapping[str, Any]) -> list[str]:
    values = []
    for key, value in person.items():
        normalized = str(key).strip().lower().replace("_", " ")
        text = str(value).strip()
        if (
            normalized in NAME_LABELS
            or "preferred name" in normalized
            or "culture" in normalized
            or "cultura" in normalized
            or (normalized in {"id", "person id", "participant id"} and len(text) < 4)
        ):
            continue
        if text:
            values.append(text)
    return values


def name_variants(names: Iterable[str]) -> list[str]:
    variants = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if len(name) < 2 or name.lower() in NON_NAME_VALUES:
            continue
        variants.add(name)
        for component in re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)?", name):
            if len(component) >= 3 and component.lower() not in NON_NAME_VALUES:
                variants.add(component)
    return sorted(variants, key=len, reverse=True)


def profile_value_variants(
    person_values: Iterable[str],
) -> list[str]:
    variants = set()
    for raw_value in person_values:
        value = str(raw_value).strip()
        if value.lower() in NON_PRIVATE_VALUES:
            continue
        candidates = [value]
        candidates.extend(
            component.strip()
            for component in re.split(r"[,;|/\n]+", value)
        )
        for candidate in candidates:
            lowered = candidate.lower()
            is_age_bucket = bool(
                re.fullmatch(r"\d{2}\+?", candidate)
                or re.fullmatch(r"\d{2}\s*[-–]\s*\d{2}", candidate)
            )
            if (
                lowered not in NON_PRIVATE_VALUES
                and (len(candidate) >= 4 or is_age_bucket)
            ):
                variants.add(candidate)
    return sorted(variants, key=len, reverse=True)


@functools.lru_cache(maxsize=32)
def _value_pattern(values: tuple[str, ...]) -> re.Pattern[str] | None:
    if not values:
        return None
    return re.compile(
        r"(?i)(?<!\w)(?:" + "|".join(re.escape(value) for value in values) + r")(?!\w)"
    )


def replace_private_text(
    value: str,
    names: Iterable[str],
    private_values: Iterable[str] = (),
) -> str:
    sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    private_pattern = _value_pattern(
        tuple(profile_value_variants(private_values))
    )
    if private_pattern:
        sanitized = private_pattern.sub(
            "[REDACTED_PROFILE_ATTRIBUTE]",
            sanitized,
        )
    name_pattern = _value_pattern(tuple(name_variants(names)))
    if name_pattern:
        sanitized = name_pattern.sub("Participant", sanitized)
    return sanitized


def sanitize_value(
    value: Any,
    names: Iterable[str],
    private_values: Iterable[str] = (),
) -> Any:
    if isinstance(value, str):
        return replace_private_text(value, names, private_values)
    if isinstance(value, list):
        return [
            sanitize_value(item, names, private_values) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: sanitize_value(item, names, private_values)
            for key, item in value.items()
            if key not in PRIVATE_ANNOTATION_KEYS
        }
    return value


def meaningful_judgment(task_data: Mapping[str, Any]) -> bool:
    return bool(
        task_data.get("annotations")
        or task_data.get("imageAnnotations")
        or task_data.get("textConnectness")
        or task_data.get("imageConnectness")
        or task_data.get("generalComment")
        or task_data.get("culturalScore") not in (None, "")
    )


def eligible_annotators(
    human_export: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], int]:
    task_count = len(human_export.get("tasks", []))
    minimum_meaningful = max(1, (task_count + 1) // 2)
    included = []
    excluded = 0
    for annotation_record in human_export.get("annotations", []):
        meaningful_count = sum(
            meaningful_judgment(task_data)
            for task_data in annotation_record.get("taskData", {}).values()
        )
        if meaningful_count >= minimum_meaningful:
            included.append(annotation_record)
        else:
            excluded += 1
    return included, excluded


def sanitize_annotation_item(
    item: Mapping[str, Any],
    names: Iterable[str],
    private_values: Iterable[str],
) -> dict[str, Any]:
    return {
        key: sanitize_value(value, names, private_values)
        for key, value in item.items()
        if key not in PRIVATE_ANNOTATION_KEYS
    }


def sanitize_human_judgment(
    task_data: Mapping[str, Any],
    annotator_id: str,
    names: Iterable[str],
    private_values: Iterable[str],
) -> dict[str, Any]:
    image_annotations = {
        str(frame): [
            sanitize_annotation_item(item, names, private_values)
            for item in annotations
        ]
        for frame, annotations in task_data.get("imageAnnotations", {}).items()
    }
    return {
        "annotator_id": annotator_id,
        "cultural_score": task_data.get("culturalScore"),
        "language_similarity": task_data.get("languageSimilarity"),
        "language_similarity_justification": replace_private_text(
            str(task_data.get("languageSimilarityJustification", "")),
            names,
            private_values,
        ),
        "general_comment": replace_private_text(
            str(task_data.get("generalComment", "")),
            names,
            private_values,
        ),
        "text_connectedness": sanitize_value(
            task_data.get("textConnectness", {}), names, private_values
        ),
        "image_connectedness": sanitize_value(
            task_data.get("imageConnectness", {}), names, private_values
        ),
        "global_feedback": sanitize_value(
            task_data.get("globalFeedback", {}), names, private_values
        ),
        "text_annotations": [
            sanitize_annotation_item(item, names, private_values)
            for item in task_data.get("annotations", [])
        ],
        "image_annotations": image_annotations,
    }


def local_image_paths(
    task: Mapping[str, Any],
    public_id: str,
    image_dir: Path | None,
    image_path_prefix: Path,
) -> list[str] | None:
    if image_dir is None:
        return None

    paths = []
    for index, url in enumerate(task.get("images", [])):
        source_name = Path(urlparse(str(url)).path).name
        suffix = Path(source_name).suffix.lower() or ".png"
        public_name = f"{public_id}_{index:02d}{suffix}"
        source = image_dir / source_name
        target = image_dir / public_name
        if not target.is_file() and source.is_file():
            source.replace(target)
        if target.is_file():
            paths.append((image_path_prefix / public_name).as_posix())
    return paths


def public_story(
    task: Mapping[str, Any],
    public_id: str,
    names: Iterable[str],
    private_values: Iterable[str],
    image_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    metadata = task.get("metadata", {})
    story = {
        "story_id": public_id,
        "story": replace_private_text(
            str(task.get("text", "")), names, private_values
        ),
        "paragraphs": sanitize_value(
            task.get("paragraphs", []), names, private_values
        ),
        "image_count": len(task.get("images", [])),
        "audio_count": len(task.get("audio", [])),
        "question": replace_private_text(
            str(task.get("question", "")), names, private_values
        ),
        "category": task.get("category"),
        "task_type": task.get("taskType"),
        "condition": metadata.get("label"),
        "model": metadata.get("model"),
        "story_arc": metadata.get("story_arc"),
    }
    if image_paths is not None:
        story["image_paths"] = list(image_paths)
    return story


def sanitize_human_export(
    human_export: Mapping[str, Any],
    language: str,
    config: Mapping[str, str],
    global_names: Iterable[str] = (),
    global_private_values: Iterable[str] = (),
    image_dir: Path | None = None,
    image_path_prefix: Path = Path("data/images"),
) -> tuple[
    list[dict[str, Any]],
    dict[str, str],
    dict[str, list[str]],
    dict[str, list[str]],
    int,
    int,
]:
    tasks = human_export.get("tasks", [])
    story_ids = {
        str(task["id"]): f"{language}_story_{index:03d}"
        for index, task in enumerate(tasks, start=1)
    }
    names_by_raw_id = {
        str(task["id"]): names_from_description(str(task.get("description", "")))
        for task in tasks
    }
    annotators, excluded_annotators = eligible_annotators(human_export)
    annotator_ids = {
        index: f"{language}_annotator_{index + 1:02d}"
        for index in range(len(annotators))
    }

    output = []
    judgment_count = 0
    image_paths_by_raw_id = {}
    for task in tqdm(
        tasks,
        desc=f"{language}: human stories",
        unit="story",
        leave=False,
    ):
        raw_id = str(task["id"])
        names = list(global_names) + names_by_raw_id[raw_id]
        image_paths = local_image_paths(
            task,
            story_ids[raw_id],
            image_dir,
            image_path_prefix,
        )
        image_paths_by_raw_id[raw_id] = image_paths or []
        record = {
            "dataset_version": DATASET_VERSION,
            "language": language,
            "language_code": config["language_code"],
            "country": config["country"],
            "culture": config["culture"],
            **public_story(
                task,
                story_ids[raw_id],
                names,
                global_private_values,
                image_paths,
            ),
            "human_annotations": [],
        }
        for index, annotation_record in enumerate(annotators):
            task_data = annotation_record.get("taskData", {}).get(raw_id)
            if not task_data:
                continue
            completed_ids = set(annotation_record.get("completedTaskIds", []))
            if raw_id not in completed_ids and not meaningful_judgment(task_data):
                continue
            record["human_annotations"].append(
                sanitize_human_judgment(
                    task_data,
                    annotator_ids[index],
                    names,
                    global_private_values,
                )
            )
            judgment_count += 1
        output.append(record)

    return (
        output,
        story_ids,
        names_by_raw_id,
        image_paths_by_raw_id,
        len(annotators),
        excluded_annotators,
    )


def sanitize_llm_record(
    record: Mapping[str, Any],
    public_id: str,
    names: Iterable[str],
    private_values: Iterable[str],
    language: str,
    config: Mapping[str, str],
    image_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    input_data = record.get("input", {})
    output_data = record.get("output", {})
    metadata = record.get("metadata", {})
    sanitized = {
        "dataset_version": DATASET_VERSION,
        "language": language,
        "language_code": config["language_code"],
        "country": config["country"],
        "culture": config["culture"],
        "story_id": public_id,
        "input": {
            "story": replace_private_text(
                str(input_data.get("story", "")), names, private_values
            ),
            "culture": input_data.get("culture"),
            "question": replace_private_text(
                str(input_data.get("question", "")), names, private_values
            ),
            "category": input_data.get("category"),
            "task_type": input_data.get("taskType"),
        },
        "output": sanitize_value(output_data, names, private_values),
        "metadata": {
            "configuration": metadata.get("configuration"),
            "consensus_reached": metadata.get("consensus_reached"),
            "total_debate_rounds": metadata.get("total_debate_rounds"),
            "initial_scores": metadata.get("initial_scores", {}),
            "agent_final_scores": metadata.get("agent_final_scores", {}),
            "debate_transcript": sanitize_value(
                metadata.get("debate_transcript", []),
                names,
                private_values,
            ),
        },
    }
    if image_paths is not None:
        sanitized["image_paths"] = list(image_paths)
    return sanitized


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_checksums(output_dir: Path) -> None:
    paths = sorted(output_dir.rglob("*.jsonl"))
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        paths.append(manifest_path)
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def sanitize_language(
    language: str,
    config: Mapping[str, str],
    human_path: Path,
    llm_path: Path,
    output_dir: Path,
    image_dir: Path | None = None,
    image_path_prefix: Path = Path("data/images"),
) -> dict[str, Any]:
    human_export = json.loads(human_path.read_text(encoding="utf-8"))
    source_llm_records = read_jsonl(llm_path)
    global_names = []
    global_private_values = []
    for task in human_export.get("tasks", []):
        global_names.extend(
            names_from_description(str(task.get("description", "")))
        )
    for record in source_llm_records:
        person = record.get("input", {}).get("person", {})
        global_names.extend(
            names_from_person(person)
        )
        global_private_values.extend(private_values_from_person(person))
    public_values = {
        language.casefold(),
        *(str(value).strip().casefold() for value in config.values()),
    }
    global_private_values = [
        value
        for value in global_private_values
        if str(value).strip().casefold() not in public_values
    ]
    (
        human_records,
        story_ids,
        human_names,
        image_paths_by_raw_id,
        annotator_count,
        excluded_annotators,
    ) = sanitize_human_export(
        human_export,
        language,
        config,
        global_names,
        global_private_values,
        image_dir,
        image_path_prefix,
    )

    llm_records = []
    for record in tqdm(
        source_llm_records,
        desc=f"{language}: LLM judgments",
        unit="record",
        leave=False,
    ):
        raw_id = str(record["story_id"])
        if raw_id not in story_ids:
            raise ValueError(
                f"LLM story ID is absent from human export for {language}"
            )
        person = record.get("input", {}).get("person", {})
        names = list(global_names)
        names.extend(human_names.get(raw_id, []))
        names.extend(names_from_person(person))
        llm_records.append(
            sanitize_llm_record(
                record,
                story_ids[raw_id],
                names,
                global_private_values,
                language,
                config,
                (
                    image_paths_by_raw_id[raw_id]
                    if image_dir is not None
                    else None
                ),
            )
        )

    language_dir = output_dir / language
    write_jsonl(language_dir / "human_annotations.jsonl", human_records)
    write_jsonl(language_dir / "llm_judgments.jsonl", llm_records)
    assert_no_source_identifiers(
        language_dir,
        global_names,
        story_ids.keys(),
        global_private_values,
    )
    summary = {
        "language": language,
        "language_code": config["language_code"],
        "country": config["country"],
        "culture": config["culture"],
        "stories": len(human_records),
        "human_judgments": sum(
            len(record["human_annotations"]) for record in human_records
        ),
        "anonymous_annotators": annotator_count,
        "excluded_incomplete_annotator_records": excluded_annotators,
        "llm_judgments": len(llm_records),
    }
    if image_dir is not None:
        summary["local_images"] = sum(
            len(paths) for paths in image_paths_by_raw_id.values()
        )
    return summary


def assert_no_direct_identifiers(output_dir: Path) -> None:
    forbidden_keys = {
        "userEmail",
        "userId",
        "submissionUserId",
        "submissionTaskId",
        "taskId",
        "study_number",
        "person",
        "name",
        "nome",
    }
    violations = []
    for path in output_dir.rglob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if EMAIL_PATTERN.search(line):
                    violations.append(f"{path}:{line_number}: email")
                if UUID_PATTERN.search(line):
                    violations.append(f"{path}:{line_number}: UUID")
                value = json.loads(line)
                stack = [value]
                while stack:
                    current = stack.pop()
                    if isinstance(current, dict):
                        for key, item in current.items():
                            if key in forbidden_keys:
                                violations.append(
                                    f"{path}:{line_number}: key {key}"
                                )
                            stack.append(item)
                    elif isinstance(current, list):
                        stack.extend(current)
    if violations:
        raise ValueError(
            "Direct identifier audit failed:\n" + "\n".join(violations[:50])
        )


def assert_no_source_identifiers(
    output_dir: Path,
    names: Iterable[str],
    raw_story_ids: Iterable[str],
    private_values: Iterable[str] = (),
) -> None:
    name_values = name_variants(names)
    name_pattern = (
        re.compile(
            r"(?i)(?<!\w)(?:"
            + "|".join(re.escape(value) for value in name_values)
            + r")(?!\w)"
        )
        if name_values
        else None
    )
    raw_ids = [value for value in raw_story_ids if value]
    private_pattern = _value_pattern(
        tuple(profile_value_variants(private_values))
    )
    violations = []
    for path in output_dir.rglob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if name_pattern and name_pattern.search(line):
                    violations.append(
                        f"{path}:{line_number}: participant name variant"
                    )
                if any(raw_id in line for raw_id in raw_ids):
                    violations.append(
                        f"{path}:{line_number}: raw story identifier"
                    )
                if private_pattern and private_pattern.search(line):
                    violations.append(
                        f"{path}:{line_number}: source profile attribute"
                    )
    if violations:
        raise ValueError(
            "Source identifier audit failed:\n" + "\n".join(violations[:50])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", required=True, type=Path)
    parser.add_argument("--llm-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "judgments",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help=(
            "Opt in to anonymous local image paths by supplying the directory "
            "containing images downloaded with their source basenames."
        ),
    )
    parser.add_argument(
        "--image-path-prefix",
        type=Path,
        default=Path("data") / "images",
        help="Repository-relative path written into released records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = {
        "dataset_version": DATASET_VERSION,
        "release_date": RELEASE_DATE,
        "description": (
            "De-identified human annotations and multi-agent LLM judgments."
        ),
        "sanitization": {
            "story_ids": "Sequential language-local identifiers.",
            "annotator_ids": "Sequential language-local identifiers without a mapping.",
            "removed": [
                "annotator names and email addresses",
                "user and submission identifiers",
                "raw story and study identifiers",
                "participant profile objects and HTML profile descriptions",
                "generation prompts and generated search queries",
                "annotation timestamps",
                "external media URLs containing raw story identifiers",
            ],
            "text_redaction": (
                "All full participant names and name components observed in a "
                "language source are replaced with 'Participant' throughout that "
                "language's released text. Other source profile values are "
                "replaced with '[REDACTED_PROFILE_ATTRIBUTE]'."
            ),
            "local_images": (
                "When --image-dir is supplied, available images are renamed "
                "with anonymous story IDs and referenced by repository-relative "
                "paths in both aligned judgment files."
            ),
            "incomplete_annotations": (
                "Annotator records with meaningful judgments for fewer than half "
                "of the language's stories are excluded."
            ),
        },
        "languages": [],
    }
    for language, config in tqdm(
        LANGUAGES.items(),
        total=len(LANGUAGES),
        desc="Sanitizing languages",
        unit="language",
    ):
        manifest["languages"].append(
            sanitize_language(
                language,
                config,
                args.human_dir / config["human_file"],
                args.llm_dir / config["llm_file"],
                args.output_dir,
                args.image_dir,
                args.image_path_prefix,
            )
        )
    assert_no_direct_identifiers(args.output_dir)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_checksums(args.output_dir)
    print(f"Wrote sanitized datasets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
