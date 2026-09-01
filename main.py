#!/usr/bin/env python3
"""Command-line entry point for cultural-alignment evaluation."""

from __future__ import annotations

import argparse
import asyncio
import logging

from config.judge_configs import CONFIGS, get_config
from src.agents.factory import create_judges
from src.pipeline.runner import PipelineRunner, load_stories
from src.utils.display import print_final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text-image stories with multiple cultural judges."
    )
    parser.add_argument(
        "--config",
        choices=CONFIGS,
        default="multimodal",
        help="Context supplied to judges.",
    )
    parser.add_argument(
        "--stories",
        default="examples/sample_stories.json",
        help="JSON array containing story records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Evaluate records even when their IDs already exist in the output.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stories = load_stories(args.stories)
    judges = create_judges()
    results = await PipelineRunner(
        get_config(args.config),
        judges,
        overwrite=args.overwrite,
    ).run(stories)
    print_final_summary(results)


if __name__ == "__main__":
    asyncio.run(run())
