"""Evaluation presets controlling the context supplied to each judge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeConfig:
    id: str
    name: str
    include_persona: bool = False
    include_images: bool = False
    include_paragraph_image_alignment: bool = False


CONFIGS = {
    "text": JudgeConfig(
        id="text",
        name="Story text only",
    ),
    "multimodal": JudgeConfig(
        id="multimodal",
        name="Story, persona, and paragraph-image alignment",
        include_persona=True,
        include_images=True,
        include_paragraph_image_alignment=True,
    ),
}


def get_config(config_id: str) -> JudgeConfig:
    try:
        return CONFIGS[config_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown configuration {config_id!r}; choose from {list(CONFIGS)}"
        ) from exc
