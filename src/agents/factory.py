"""OpenAI-compatible cultural-alignment judge agents."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI

from config.settings import (
    MAX_OUTPUT_TOKENS,
    MAX_TOKENS_PARAMETER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    judge_specs,
)
from src.agents.prompts import (
    DEBATE_ROUND_SYSTEM,
    INITIAL_RATING_SYSTEM,
    build_user_message_debate,
    build_user_message_initial,
)
from src.schemas.models import (
    MARKER_CATEGORIES,
    CulturalMarker,
    CulturalRating,
    ImageCulturalMarker,
    bracket_for_score,
)
from src.utils import extract_json

logger = logging.getLogger(__name__)


def _image_url(value: str) -> str:
    """Return an HTTP/data URL, encoding a local image when necessary."""
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@dataclass
class JudgeAgent:
    name: str
    model_alias: str
    debate: bool = True
    _client: AsyncOpenAI = field(repr=False, default=None)

    max_retries: int = 4
    retry_base_delay: float = 2.0
    parse_retries: int = 2

    async def _chat(
        self,
        system: str,
        user: str,
        image_urls: list[str] | None = None,
        temperature: float = 0.2,
    ) -> str:
        user_content: str | list[dict[str, Any]] = user
        if image_urls:
            user_content = [{"type": "text", "text": user}]
            user_content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_url(image), "detail": "auto"},
                }
                for image in image_urls
            )

        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_alias,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    **{MAX_TOKENS_PARAMETER: MAX_OUTPUT_TOKENS},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError(f"{self.name} returned an empty response")
                return re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
            except openai.RateLimitError:
                if attempt + 1 == self.max_retries:
                    raise
            except (openai.APIConnectionError, openai.APITimeoutError):
                if attempt + 1 == self.max_retries:
                    raise
            except openai.APIStatusError as exc:
                if exc.status_code < 500 or attempt + 1 == self.max_retries:
                    raise

            delay = self.retry_base_delay * (2**attempt)
            logger.warning("%s request failed; retrying in %.1fs", self.name, delay)
            await asyncio.sleep(delay)

        raise RuntimeError("Request retry loop ended unexpectedly")

    def _parse_rating(self, raw: str, story_text: str) -> CulturalRating:
        data = extract_json(raw)
        if "score" not in data:
            raise KeyError("score")

        score = int(data["score"])
        markers = [
            CulturalMarker(
                category=(
                    item.get("category", "")
                    if item.get("category", "") in MARKER_CATEGORIES
                    else "unclassified"
                ),
                span=item.get("span", ""),
            )
            for item in data.get("cultural_markers_found", [])
            if isinstance(item, dict) and item.get("span", "")
        ]

        image_items = data.get(
            "Image_cultural_markers_found",
            data.get("image_cultural_markers_found", []),
        )
        image_markers = [
            ImageCulturalMarker(
                category=(
                    item.get("category", "")
                    if item.get("category", "") in MARKER_CATEGORIES
                    else "unclassified"
                ),
                comment=item.get("comment", item.get("reason", "")),
            )
            for item in image_items
            if isinstance(item, dict) and item.get("comment", item.get("reason", ""))
        ]

        evidence = [
            item
            if isinstance(item, str)
            else item.get("span", item.get("text", str(item)))
            if isinstance(item, dict)
            else str(item)
            for item in data.get("evidence", [])
        ]

        return CulturalRating(
            agent_name=self.name,
            model=self.model_alias,
            score=score,
            reasoning=data.get("reasoning", ""),
            evidence=evidence,
            bracket=data.get("bracket", bracket_for_score(score)),
            cultural_markers_found=markers,
            image_cultural_markers_found=image_markers,
        )

    async def _rate(
        self,
        system: str,
        user_message: str,
        story_text: str,
        image_urls: list[str] | None,
    ) -> tuple[CulturalRating, dict[str, Any]]:
        prompt = user_message
        for attempt in range(self.parse_retries + 1):
            raw = await self._chat(system, prompt, image_urls=image_urls)
            try:
                data = extract_json(raw)
                return self._parse_rating(raw, story_text), data
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                if attempt == self.parse_retries:
                    raise
                prompt += (
                    "\n\nReturn one valid JSON object only. It must include an "
                    "integer `score` from 0 to 100."
                )
        raise RuntimeError("Parsing retry loop ended unexpectedly")

    async def rate_initial(
        self,
        story_text: str,
        question: str,
        culture: str,
        persona: str | None = None,
        image_urls: list[str] | None = None,
        image_descriptions: list[str] | None = None,
        paragraphs: list[str] | None = None,
    ) -> CulturalRating:
        message = build_user_message_initial(
            story_text=story_text,
            question=question,
            culture=culture,
            persona=persona,
            image_urls=image_urls,
            image_descriptions=image_descriptions,
            paragraphs=paragraphs,
        )
        rating, _ = await self._rate(
            INITIAL_RATING_SYSTEM, message, story_text, image_urls
        )
        return rating

    async def rate_debate(
        self,
        story_text: str,
        question: str,
        culture: str,
        previous_ratings: list[dict[str, Any]],
        round_number: int,
        persona: str | None = None,
        image_urls: list[str] | None = None,
        image_descriptions: list[str] | None = None,
        paragraphs: list[str] | None = None,
    ) -> tuple[CulturalRating, list[str]]:
        message = build_user_message_debate(
            story_text=story_text,
            question=question,
            culture=culture,
            previous_ratings=previous_ratings,
            round_number=round_number,
            persona=persona,
            image_urls=image_urls,
            image_descriptions=image_descriptions,
            paragraphs=paragraphs,
        )
        rating, data = await self._rate(
            DEBATE_ROUND_SYSTEM, message, story_text, image_urls
        )
        critiques = [
            str(item) for item in data.get("critiques", []) if str(item).strip()
        ]
        return rating, critiques


def create_judges(overrides: list[dict[str, object]] | None = None) -> list[JudgeAgent]:
    """Create judge agents using one OpenAI-compatible endpoint."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add a key."
        )

    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    specs = overrides or judge_specs()
    return [
        JudgeAgent(
            name=str(spec["name"]),
            model_alias=str(spec["model"]),
            debate=bool(spec.get("debate", True)),
            _client=client,
        )
        for spec in specs
    ]
