"""Robust JSON extraction from LLM output.

Handles:
- Qwen3 <think>...</think> blocks before the actual content
- Plain-text chain-of-thought before JSON (Qwen3.5 style)
- Markdown code fences (```json ... ```)
- Leading/trailing prose around JSON
- Nested braces inside thinking text
"""

from __future__ import annotations

import json
import re
import logging

logger = logging.getLogger(__name__)

# Pattern to strip Qwen3 thinking blocks
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Fallback: unclosed <think> tag (model emits <think> but no closing tag)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)
# Fallback: content before a bare </think> tag (no opening tag)
_THINK_BARE_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)

# Pattern to extract content from markdown code fences
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

# Pattern to locate the response JSON object (must contain a "score" key)
_SCORE_JSON_RE = re.compile(r'\{[^{}]*"score"\s*:', re.DOTALL)


def _try_parse_brace_block(text: str, start: int) -> dict | None:
    """Try to parse a balanced { ... } block starting at *start*."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try fixing trailing commas
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        return None
    return None


def extract_json(raw: str) -> dict:
    """Extract a JSON object from potentially messy LLM output.

    Raises json.JSONDecodeError if no valid JSON can be found.
    """
    text = raw.strip()

    # 1. Strip <think>...</think> blocks (Qwen3 thinking mode)
    text = _THINK_RE.sub("", text).strip()
    # 1b. Handle unclosed <think> tag
    text = _THINK_UNCLOSED_RE.sub("", text).strip()
    # 1c. Handle bare </think> without opening tag
    text = _THINK_BARE_CLOSE_RE.sub("", text).strip()

    # 2. Try parsing directly first (cleanest case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Try extracting from markdown code fences
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 4. Find the JSON block containing "score" — this is the actual response.
    #    Search ALL occurrences and try each (handles thinking text with braces).
    for m in _SCORE_JSON_RE.finditer(text):
        result = _try_parse_brace_block(text, m.start())
        if result is not None and "score" in result:
            return result

    # 5. Fallback: try every { ... } block from the END of the text backwards
    #    (the response JSON is usually at the end, after any thinking text)
    last_close = text.rfind("}")
    while last_close != -1:
        # Walk backwards to find the matching open brace
        depth = 0
        for i in range(last_close, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    result = _try_parse_brace_block(text, i)
                    if result is not None:
                        return result
                    break
        # Try the next } going backwards
        last_close = text.rfind("}", 0, last_close)

    # 6. Last resort: first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"Could not extract JSON from LLM output ({len(raw)} chars)",
        raw[:200],
        0,
    )
