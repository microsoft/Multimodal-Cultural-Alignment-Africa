import json

from src.debate.consensus import check_consensus, compute_final_score
from src.io.results import _culture_slug
from src.agents.prompts import build_user_message_initial
from src.pipeline.runner import load_stories
from src.schemas.models import CulturalRating
from src.utils import extract_json


def rating(score: int) -> CulturalRating:
    return CulturalRating(
        agent_name=f"judge-{score}",
        model="test-model",
        score=score,
        reasoning="test",
    )


def test_consensus_and_mean():
    ratings = [rating(70), rating(80), rating(75)]
    assert check_consensus(ratings, threshold=10)
    assert compute_final_score(ratings) == 75.0


def test_extract_json_from_fenced_response():
    result = extract_json(
        """analysis
```json
{"score": 42}
```"""
    )
    assert result == {"score": 42}


def test_load_stories_resolves_local_images(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    source = tmp_path / "stories.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "text": "Story",
                    "culture": "Test",
                    "question": "Question?",
                    "images": ["image.png"],
                }
            ]
        ),
        encoding="utf-8",
    )

    stories = load_stories(str(source))
    assert stories[0].images == [str(image.resolve())]


def test_culture_slug_cannot_escape_results_directory():
    assert _culture_slug("../../IsiXhosa") == "isixhosa"


def test_local_image_path_is_not_added_to_prompt():
    prompt = build_user_message_initial(
        story_text="Story",
        question="Question?",
        culture="Test",
        image_urls=["/home/person/private/image.png"],
    )
    assert "/home/person" not in prompt
    assert "attached to this request" in prompt
