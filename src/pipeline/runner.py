"""Load story records, execute judges, and save JSONL results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config.judge_configs import JudgeConfig
from src.agents.factory import JudgeAgent
from src.debate.consensus import compute_final_score
from src.debate.orchestrator import DebateOrchestrator
from src.io.results import load_completed_story_ids, save_result
from src.schemas.models import JudgeResult, Story


def load_stories(path: str) -> list[Story]:
    """Load records and resolve local image paths relative to the JSON file."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Story input must be a JSON array")

    stories: list[Story] = []
    for record in records:
        item = dict(record)
        item["images"] = [
            str((source.parent / image).resolve())
            if image and not image.startswith(("http://", "https://", "data:"))
            else image
            for image in item.get("images", [])
        ]
        stories.append(Story.model_validate(item))
    return stories


class PipelineRunner:
    def __init__(
        self,
        config: JudgeConfig,
        agents: list[JudgeAgent],
        overwrite: bool = False,
    ):
        self.config = config
        self.agents = agents
        self.overwrite = overwrite

    async def run(self, stories: list[Story]) -> list[JudgeResult]:
        completed = set() if self.overwrite else load_completed_story_ids(self.config.id)
        orchestrator = DebateOrchestrator(self.agents, self.config)
        results = []

        for story in stories:
            if story.id in completed:
                continue
            initial, rounds, final, consensus = await orchestrator.run(story)
            result = JudgeResult(
                story=story,
                configuration=self.config.id,
                initial_ratings=initial,
                debate_rounds=rounds,
                final_ratings=final,
                final_score=compute_final_score(final),
                consensus_reached=consensus,
                total_rounds=len(rounds),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            save_result(result, replace_existing=self.overwrite)
            results.append(result)
        return results
