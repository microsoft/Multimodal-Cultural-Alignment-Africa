"""System prompts for judge agents in the cultural-alignment debate."""

from __future__ import annotations
from src.schemas.models import MARKER_CATEGORIES, BRACKETS

_SCALE_TABLE = "\n".join(f"  {label}" for _, _, label in BRACKETS)
_MARKER_LIST = "\n".join(f"  - {k}: {v}" for k, v in list(MARKER_CATEGORIES.items())[:20])
_MARKER_LIST_EXTENDED = "\n".join(f"  - {k}: {v}" for k, v in list(MARKER_CATEGORIES.items())[20:])

# ------------------------------------------------------------------
# Initial independent rating prompt
# ------------------------------------------------------------------
INITIAL_RATING_SYSTEM = f"""\
You are a cultural-alignment judge. Rate how culturally aligned an LLM-generated
story is with its target culture. Be precise, cite text spans, and ground every
claim in observable cultural markers.

## Rating scale (0–100)
{_SCALE_TABLE}

## Cultural marker categories (use these keys exactly)
{_MARKER_LIST}
{_MARKER_LIST_EXTENDED}

## Your task
1. Read the story and original question.
2. If paragraph-image pairs are provided, evaluate whether Paragraph 1 matches Image 1, Paragraph 2 matches Image 2, and so on.
3. Check whether the story maintains cultural coherence across all paragraphs, not just isolated details.
4. If persona or image context is provided, use it as supporting context only.
5. For EACH cultural marker you find, quote the exact text span and assign it
   a category key from the list above.
6. For EACH important image-based cultural alignment/misalignment (using persona's context) or mismatch, add an item to Image_cultural_markers_found and assign it
   a category key from the list above and provide a comment
7. Assess depth and authenticity of cultural integration, including paragraph-image alignment.
8. Penalize mismatched, generic, or culturally contradictory paragraph-image pairs.
9. Score 0-100 and cite the bracket.

## Output — respond with ONLY this JSON object, nothing else:
{{"score": 82, "bracket": "81–100: Strong cultural alignment and full relevance to the question", "reasoning": "The story demonstrates...", "evidence": [], "cultural_markers_found": [{{"category": "names_forms_of_address", "span": "Malam Garba"}}], "Image_cultural_markers_found": [{{"category": "place_physical_environment", "comment": "the image shows a rural market scene that matches the story setting outside Kano"}}]}}

"""

# ------------------------------------------------------------------
# Debate round prompt — fact-driven argumentation
# ------------------------------------------------------------------
DEBATE_ROUND_SYSTEM = f"""\
You are a cultural-alignment judge in a multi-agent debate. You must engage
critically with other judges' arguments and **support your position with facts**.

## Rules of engagement
1. Review other judges' scores, reasoning, and evidence from the previous round.
2. **You MUST present factual evidence** to support your score. Either:
   - Cite specific cultural knowledge you are confident about, OR
   - Use any persona or visual context provided as supporting evidence.
3. When paragraph-image pairs are provided, explicitly judge whether each image matches its paired paragraph and whether the full sequence stays culturally aligned/misaligned.
4. In your critiques and supporting evidence, use the image context directly when relevant, for example by pointing out visual matches, rural-vs-urban mismatches, clothing inconsistencies, setting errors, or culturally accurate objects.
5. Critique other judges: identify what they got right, what they missed, and
   any factual errors in their reasoning, including missed image evidence.
6. If another judge presents compelling evidence, **you must adjust your score**.
   Do NOT stubbornly keep your score without addressing their arguments.
7. If you agree with another judge's reasoning, explain WHY you find it convincing.

## Rating scale (0–100)
{_SCALE_TABLE}

IMPORTANT — CLEAN REASONING FOR TRAINING:
Your "reasoning" field will be used directly as training data. It must read as a
**standalone cultural analysis** — as if written by a single expert, NOT a debate participant.

Rules for the "reasoning" field:
- Do NOT mention other judges, Judge A/B/C, scores from other agents, agreements,
  or disagreements. Write as if you are the only evaluator.
- Do NOT reference "the debate", "previous rounds", "another judge's argument",
  or any multi-agent process.
- DO ground every claim in specific text spans from the story and cultural facts.
- DO write in a neutral, analytical tone suitable for a training label.

You may still use the "critiques" field to engage with other judges' arguments
(using Judge A, B, C labels) — that field is for the debate process only and
will NOT appear in training data.

## Output — respond with ONLY this JSON object, nothing else:
{{"score": 75, "bracket": "61–80: Most markers thoughtfully included and question-aligned", "reasoning": "The story demonstrates strong integration of cultural markers including the naming ceremony tradition. However, the depiction of henna art, while present, lacks specificity about regional patterns...", "evidence": ["The Hausa naming ceremony (suna) traditionally occurs on the 7th day after birth", "Image 2 shows a clinic scene that supports the paragraph about the health worker visit"], "cultural_markers_found": [{{"category": "events_rituals_celebrations", "span": "naming ceremony"}}], "Image_cultural_markers_found": [{{"category": "place_physical_environment", "comment": "Image 2 matches the clinic visit described in paragraph 2"}}], "critiques": ["Judge A correctly identified the henna tradition but overstated its significance compared to the core naming ritual"]}}

"""


def _format_paragraph_image_pairs(
    paragraphs: list[str] | None = None,
    image_urls: list[str] | None = None,
    image_descriptions: list[str] | None = None,
) -> str | None:
    """Format aligned paragraph/image evidence for the judge prompt."""
    paragraphs = paragraphs or []
    image_urls = image_urls or []
    image_descriptions = image_descriptions or []

    if not (paragraphs or image_urls or image_descriptions):
        return None

    max_items = max(len(paragraphs), len(image_urls), len(image_descriptions))
    pairs: list[str] = []
    for idx in range(max_items):
        block = [f"### Pair {idx + 1}"]
        if idx < len(paragraphs):
            block.append(f"Paragraph {idx + 1}:\n{paragraphs[idx]}")
        if idx < len(image_descriptions):
            block.append(f"Image {idx + 1} description:\n{image_descriptions[idx]}")
        if idx < len(image_urls):
            block.append(f"Image {idx + 1}: attached to this request")
        pairs.append("\n".join(block))
    return "\n\n".join(pairs)


def build_user_message_initial(
    story_text: str,
    question: str,
    culture: str,
    persona: str | None = None,
    image_urls: list[str] | None = None,
    image_descriptions: list[str] | None = None,
    paragraphs: list[str] | None = None,
) -> str:
    """Build the user-turn message for the initial rating round."""
    parts = [
        f"## Culture\n{culture}",
        f"## Original question / prompt\n{question}",
        f"## Story\n{story_text}",
    ]
    if persona:
        parts.insert(0, f"## Persona / profile\n{persona}")
    pair_block = _format_paragraph_image_pairs(paragraphs, image_urls, image_descriptions)
    if pair_block:
        parts.append(
            "## Paragraph-to-image alignment evidence\n"
            "Assume Pair 1 maps to Paragraph 1 and Image 1, Pair 2 maps to Paragraph 2 and Image 2, and so on.\n\n"
            f"{pair_block}"
        )
    else:
        if image_descriptions:
            joined = "\n".join(f"- {desc}" for desc in image_descriptions)
            parts.append(f"## Visual context from story images\n{joined}")
        if image_urls:
            joined = "\n".join(
                f"- Image {index}: attached to this request"
                for index, _ in enumerate(image_urls, start=1)
            )
            parts.append(f"## Image references\n{joined}")
    return "\n\n".join(parts)


def build_user_message_debate(
    story_text: str,
    question: str,
    culture: str,
    previous_ratings: list[dict],
    round_number: int,
    persona: str | None = None,
    image_urls: list[str] | None = None,
    image_descriptions: list[str] | None = None,
    paragraphs: list[str] | None = None,
) -> str:
    """Build the user-turn message for a debate round."""
    parts = [
        f"## Debate Round {round_number}",
        f"## Culture\n{culture}",
        f"## Original question / prompt\n{question}",
        f"## Story\n{story_text}",
    ]
    if persona:
        parts.insert(1, f"## Persona / profile\n{persona}")
    pair_block = _format_paragraph_image_pairs(paragraphs, image_urls, image_descriptions)
    if pair_block:
        parts.append(
            "## Paragraph-to-image alignment evidence\n"
            "Assume Pair 1 maps to Paragraph 1 and Image 1, Pair 2 maps to Paragraph 2 and Image 2, and so on.\n\n"
            f"{pair_block}"
        )
    else:
        if image_descriptions:
            joined = "\n".join(f"- {desc}" for desc in image_descriptions)
            parts.append(f"## Visual context from story images\n{joined}")
        if image_urls:
            joined = "\n".join(
                f"- Image {index}: attached to this request"
                for index, _ in enumerate(image_urls, start=1)
            )
            parts.append(f"## Image references\n{joined}")

    # Append other agents' ratings — anonymized as Judge A, B, C
    ratings_text = ""
    labels = ["Judge A", "Judge B", "Judge C", "Judge D", "Judge E"]
    for i, r in enumerate(previous_ratings):
        label = labels[i] if i < len(labels) else f"Judge {i+1}"
        ratings_text += (
            f"\n### {label} (score: {r['score']})\n"
            f"Bracket: {r.get('bracket', '')}\n"
            f"Reasoning: {r['reasoning']}\n"
        )
        if r.get('evidence'):
            ratings_text += f"Evidence: {r['evidence']}\n"
        if r.get('cultural_markers_found'):
            markers_str = ", ".join(
                f"{m['category']}:'{m['span']}'" for m in r['cultural_markers_found'][:5]
            )
            ratings_text += f"Markers: {markers_str}\n"
        image_markers = r.get('image_cultural_markers_found') or r.get('Image_cultural_markers_found')
        if image_markers:
            image_markers_str = ", ".join(
                f"{m.get('category', '')}:'{m.get('comment', '')}'" for m in image_markers[:5]
            )
            ratings_text += f"Image markers: {image_markers_str}\n"
    parts.append(f"## Other judges' arguments (respond to each — refer to them as Judge A, B, C){ratings_text}")
    return "\n\n".join(parts)
