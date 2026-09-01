"""Run independent ratings followed by bounded consensus debate."""

from __future__ import annotations

import asyncio
import logging

from config.judge_configs import JudgeConfig
from config.settings import AGREEMENT_THRESHOLD, MAX_DEBATE_ROUNDS
from src.agents.factory import JudgeAgent
from src.debate.consensus import check_consensus
from src.schemas.models import CulturalRating, DebateRound, Story
from src.utils.display import (
    print_consensus,
    print_debate_round_detail,
    print_header,
    print_ratings_table,
    print_story_banner,
)

logger = logging.getLogger(__name__)


class DebateOrchestrator:
    def __init__(self, agents: list[JudgeAgent], config: JudgeConfig):
        if not agents:
            raise ValueError("At least one judge is required")
        self.agents = agents
        self.config = config

    def _story_context(
        self, story: Story
    ) -> tuple[str | None, list[str] | None, list[str] | None, list[str] | None]:
        persona = (
            story.person.model_dump_json()
            if self.config.include_persona and story.person
            else None
        )
        image_urls = story.images if self.config.include_images else None
        descriptions = None
        if self.config.include_images:
            values = story.metadata.get("image_prompts", [])
            if isinstance(values, list):
                descriptions = [value for value in values if isinstance(value, str)]

        paragraphs = None
        if self.config.include_paragraph_image_alignment:
            paragraphs = story.paragraphs or [
                part.strip() for part in story.text.split("\n\n") if part.strip()
            ]
        return persona, image_urls, descriptions, paragraphs

    async def _initial(self, story: Story) -> list[CulturalRating]:
        persona, images, descriptions, paragraphs = self._story_context(story)
        return list(
            await asyncio.gather(
                *[
                    agent.rate_initial(
                        story_text=story.text,
                        question=story.question,
                        culture=story.culture,
                        persona=persona,
                        image_urls=images,
                        image_descriptions=descriptions,
                        paragraphs=paragraphs,
                    )
                    for agent in self.agents
                ]
            )
        )

    async def _debate(
        self,
        story: Story,
        ratings: list[CulturalRating],
        round_number: int,
    ) -> DebateRound:
        persona, images, descriptions, paragraphs = self._story_context(story)
        participants = [agent for agent in self.agents if agent.debate]
        previous = [rating.model_dump() for rating in ratings]

        responses = await asyncio.gather(
            *[
                agent.rate_debate(
                    story_text=story.text,
                    question=story.question,
                    culture=story.culture,
                    previous_ratings=[
                        rating
                        for rating in previous
                        if rating["agent_name"] != agent.name
                    ],
                    round_number=round_number,
                    persona=persona,
                    image_urls=images,
                    image_descriptions=descriptions,
                    paragraphs=paragraphs,
                )
                for agent in participants
            ]
        )
        new_ratings = [rating for rating, _ in responses]
        critiques = {
            agent.name: critique
            for agent, (_, critique) in zip(participants, responses)
        }
        return DebateRound(
            round_number=round_number,
            ratings=new_ratings,
            critiques=critiques,
            consensus_reached=check_consensus(new_ratings),
        )

    async def run(
        self, story: Story
    ) -> tuple[list[CulturalRating], list[DebateRound], list[CulturalRating], bool]:
        print_story_banner(story.id, story.culture, self.config.id)
        initial = await self._initial(story)
        current = initial
        rounds: list[DebateRound] = []

        spread = max(r.score for r in current) - min(r.score for r in current)
        print_ratings_table(
            current, "INITIAL SCORES", spread, AGREEMENT_THRESHOLD
        )
        if check_consensus(current):
            print_consensus(True, sum(r.score for r in current) / len(current), 0)
            return initial, rounds, current, True

        for round_number in range(1, MAX_DEBATE_ROUNDS + 1):
            print_header(f"DEBATE ROUND {round_number} / {MAX_DEBATE_ROUNDS}")
            debate_round = await self._debate(story, current, round_number)
            rounds.append(debate_round)
            current = debate_round.ratings
            spread = max(r.score for r in current) - min(r.score for r in current)
            print_ratings_table(
                current,
                f"ROUND {round_number} SCORES",
                spread,
                AGREEMENT_THRESHOLD,
            )
            print_debate_round_detail(debate_round)
            if debate_round.consensus_reached:
                score = sum(r.score for r in current) / len(current)
                print_consensus(True, score, round_number)
                return initial, rounds, current, True

        score = sum(r.score for r in current) / len(current)
        print_consensus(False, score, len(rounds))
        return initial, rounds, current, False
