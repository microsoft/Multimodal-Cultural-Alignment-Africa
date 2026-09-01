"""Cached retrieval-augmented grounding through Azure web search."""

from __future__ import annotations

import ast
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from .providers import WebSearcher
    from .story_generation import write_json
except ImportError:
    from providers import WebSearcher
    from story_generation import write_json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable


def parse_queries(value: str | None) -> list[str]:
    if not value or not value.strip():
        return []
    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            break
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    if isinstance(parsed, str):
        parsed = [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [
        item.strip()
        for item in value.replace("\r", "\n").replace(";", "\n").splitlines()
        if item.strip()
    ]


@dataclass(frozen=True)
class GroundingResult:
    context: str = ""
    error: str | None = None


class WebSearchGrounder:
    def __init__(
        self,
        searcher: WebSearcher,
        cache_path: Path,
        *,
        mode: str,
        country: str,
        culture: str,
    ) -> None:
        self.searcher = searcher
        self.cache_path = cache_path
        self.mode = mode
        self.country = country
        self.culture = culture

    def _queries_for(self, profile: Mapping[str, str]) -> list[str]:
        if self.mode == "none":
            return []
        if self.mode == "auto" and profile.get("label", "").upper() != "EXCELLENT":
            return []
        queries = parse_queries(profile.get("generated_queries"))
        if queries:
            return queries
        question = profile.get("question", "")
        if not question:
            return []
        return [
            f"{question} Evidence-based guidance relevant to {self.culture} "
            f"communities in {self.country}."
        ]

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.is_file():
            return {}
        value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Grounding cache must be an object: {self.cache_path}")
        return {
            str(query): str(result)
            for query, result in value.items()
            if str(query).strip() and str(result).strip()
        }

    async def ground_profiles(
        self, profiles: Sequence[Mapping[str, str]]
    ) -> list[GroundingResult]:
        profile_queries = [self._queries_for(profile) for profile in profiles]
        all_queries = list(dict.fromkeys(q for queries in profile_queries for q in queries))
        if not all_queries:
            return [GroundingResult() for _ in profiles]

        cache = self._load_cache()
        missing = [query for query in all_queries if query not in cache]

        async def search(query: str) -> tuple[str, str | None, str | None]:
            try:
                return query, await self.searcher.search_web(query), None
            except Exception as exc:
                return query, None, str(exc)

        tasks = [asyncio.create_task(search(query)) for query in missing]
        search_errors: dict[str, str] = {}
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Retrieving web grounding",
            unit="query",
        ):
            query, result, error = await task
            if error:
                search_errors[query] = error
            elif result:
                cache[query] = result
        if any(query in cache for query in missing):
            write_json(self.cache_path, cache)

        contexts = []
        for queries in profile_queries:
            failed = [
                f"{query}: {search_errors[query]}"
                for query in queries
                if query in search_errors
            ]
            if failed:
                contexts.append(
                    GroundingResult(
                        error="; ".join(failed),
                    )
                )
                continue
            contexts.append(
                GroundingResult(
                    context="\n\n".join(
                        f"Web query: {query}\nRetrieved evidence:\n{cache[query]}"
                        for query in queries
                    )
                )
            )
        return contexts
