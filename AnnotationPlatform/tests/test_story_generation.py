import json

import pytest

from AnnotationPlatform.story_generation import (
    build_prompt,
    normalize_profile,
    parse_generated_story,
    safe_slug,
    validate_project_id,
)


def test_normalizes_study_headers():
    profile = normalize_profile(
        {
            "Study Number": "S-1",
            "What is your preferred name?": "Amara",
            "Question": "How can I manage my blood sugar levels effectively?",
        }
    )

    assert profile == {
        "study_number": "S-1",
        "name": "Amara",
        "question": "How can I manage my blood sugar levels effectively?",
    }


def test_builds_prompt_without_private_services():
    prompt = build_prompt(
        {"study_number": "S-1", "question": "A health question"},
        language="en",
        country="Kenya",
        culture="Kikuyu",
    )

    assert "A health question" in prompt
    assert "Kikuyu" in prompt
    assert "internal endpoint" not in prompt


def test_parses_complete_model_json():
    value = {
        "paragraphs": {str(i): f"p{i}" for i in range(5)},
        "image_prompts": {str(i): f"i{i}" for i in range(1, 5)},
        "audio_prompts": {str(i): f"a{i}" for i in range(1, 5)},
    }

    parsed = parse_generated_story(f"```json\n{json.dumps(value)}\n```")

    assert parsed["paragraphs"]["4"] == "p4"
    assert parsed["image_prompts"]["1"] == "i1"


def test_rejects_incomplete_model_json():
    value = {
        "paragraphs": {"1": "only one"},
        "image_prompts": {"1": "only one"},
        "audio_prompts": {},
    }

    with pytest.raises(ValueError, match="incomplete"):
        parse_generated_story(json.dumps(value))


def test_safe_paths_and_project_ids():
    assert safe_slug("google/gemma-3-27b-it") == "google-gemma-3-27b-it"
    assert (
        validate_project_id("e321ff7d-3b01-4a85-b815-241a1b9d6007")
        == "e321ff7d-3b01-4a85-b815-241a1b9d6007"
    )
    with pytest.raises(ValueError):
        validate_project_id("../../not-a-uuid")
