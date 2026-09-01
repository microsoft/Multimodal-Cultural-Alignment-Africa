"""Save / load judge results as JSONL; export for fine-tuning."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from config.settings import RESULTS_DIR
from src.schemas.models import JudgeResult


def _ensure_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _culture_slug(culture: str) -> str:
    normalized = unicodedata.normalize("NFKD", culture).encode(
        "ascii", "ignore"
    ).decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
    if not slug:
        raise ValueError("Culture must contain at least one letter or number")
    return slug


def save_result(result: JudgeResult, replace_existing: bool = False) -> Path:
    """Write one result to the config-and-culture-specific JSONL file."""
    _ensure_dir()
    culture_slug = _culture_slug(result.story.culture)
    filename = f"{result.configuration}_{culture_slug}_results.jsonl"
    path = Path(RESULTS_DIR) / filename

    # Build a compact record suitable for analysis or model training.
    markers = []
    seen_markers = set()
    image_markers = []
    seen_image_markers = set()
    for r in result.final_ratings:
        for m in r.cultural_markers_found:
            key = (m.category, m.span)
            if key not in seen_markers:
                markers.append({"category": m.category, "span": m.span})
                seen_markers.add(key)
        for im in r.image_cultural_markers_found:
            key = (im.category, im.comment)
            if key not in seen_image_markers:
                image_markers.append({"category": im.category, "comment": im.comment})
                seen_image_markers.add(key)

    record = {
        "story_id": result.story.id,
        "input": {
            "story": result.story.text,
            "culture": result.story.culture,
            "person": result.story.person.model_dump() if result.story.person else {},
            "question": result.story.question,
            "category": getattr(result.story, "category", ""),
            "taskType": getattr(result.story, "taskType", ""),
        },
        "output": {
            "score": round(result.final_score),
            "bracket": result.final_ratings[0].bracket if result.final_ratings else "",
            "reasoning": result.final_ratings[0].reasoning if result.final_ratings else "",
            "cultural_markers_found": markers,
            "Image_cultural_markers_found": image_markers,
        },
        "metadata": {
            "configuration": result.configuration,
            "consensus_reached": result.consensus_reached,
            "total_debate_rounds": result.total_rounds,
            "initial_scores": {r.agent_name: r.score for r in result.initial_ratings},
            "agent_final_scores": {r.agent_name: r.score for r in result.final_ratings},
            "debate_transcript": [
                {
                    "round": dr.round_number,
                    "scores": {r.agent_name: r.score for r in dr.ratings},
                    "critiques": dr.critiques,
                }
                for dr in result.debate_rounds
            ],
        },
    }

    serialized = json.dumps(record, ensure_ascii=False) + "\n"
    if replace_existing and path.exists():
        retained = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                existing = json.loads(line)
                if existing.get("story_id") != result.story.id:
                    retained.append(line)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.writelines(retained)
            handle.write(serialized)
        temporary.replace(path)
    else:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)

    return path


def load_results(config_id: str, culture: str | None = None) -> list[dict]:
    """Load all results for a given configuration (optionally filtered by culture)."""
    if culture:
        culture_slug = _culture_slug(culture)
        pattern = f"{config_id}_{culture_slug}_results.jsonl"
    else:
        pattern = f"{config_id}_*_results.jsonl"

    results = []
    for path in Path(RESULTS_DIR).glob(pattern):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def load_completed_story_ids(config_id: str, culture: str | None = None) -> set[str]:
    """Return set of story_ids already present in results for the given config."""
    results = load_results(config_id, culture)
    return {r["story_id"] for r in results if "story_id" in r}
