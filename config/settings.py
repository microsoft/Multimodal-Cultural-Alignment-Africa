"""Environment-driven runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT_DIR / "data" / "results"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODELS = [
    model.strip()
    for model in os.getenv(
        "JUDGE_MODELS",
        "gpt-4o-mini,gpt-4.1-mini,gpt-4o",
    ).split(",")
    if model.strip()
]

AGREEMENT_THRESHOLD = int(os.getenv("AGREEMENT_THRESHOLD", "15"))
MAX_DEBATE_ROUNDS = int(os.getenv("MAX_DEBATE_ROUNDS", "3"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
MAX_TOKENS_PARAMETER = os.getenv("MAX_TOKENS_PARAMETER", "max_tokens")
if MAX_TOKENS_PARAMETER not in {"max_tokens", "max_completion_tokens"}:
    raise ValueError(
        "MAX_TOKENS_PARAMETER must be 'max_tokens' or 'max_completion_tokens'"
    )


def judge_specs() -> list[dict[str, object]]:
    """Build deterministic judge names from the configured model list."""
    return [
        {
            "name": f"judge_{index}",
            "model": model,
            "debate": True,
        }
        for index, model in enumerate(JUDGE_MODELS, start=1)
    ]
