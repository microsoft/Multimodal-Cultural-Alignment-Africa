import asyncio
import json

import pytest

from StoryGen.grounding import WebSearchGrounder, parse_queries
from StoryGen.media import voice_for
from StoryGen.pipeline import PipelineConfig, _load_selected_profiles, _prompt_context
from StoryGen.providers import AzureOpenAISettings, AzureStoryGenerator
from StoryGen.story_generation import (
    build_prompt,
    load_profiles,
    normalize_profile,
    parse_generated_story,
    prepare_profile_for_condition,
    resolve_story_arc_path,
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


def test_loader_reports_and_skips_rows_without_questions(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "Study Number,Question\nS-1,Valid question\nS-2,\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="Skipped CSV rows"):
        profiles = load_profiles(csv_path)

    assert [profile["study_number"] for profile in profiles] == ["S-1"]


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


def test_resolves_profile_story_arc():
    arc_path = resolve_story_arc_path({"story_arc": "social support"})
    prompt = build_prompt(
        {
            "name": "Amara",
            "question": "A health question",
            "story_arc": "social support",
        },
        language="en",
        country="Kenya",
        culture="Kikuyu",
        arc_path=arc_path,
    )

    assert arc_path.name == "social_support.txt"
    assert "shared or collective" in prompt


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


def test_requires_audio_prompts_when_audio_is_enabled():
    value = {
        "paragraphs": {str(i): f"p{i}" for i in range(1, 5)},
        "image_prompts": {str(i): f"i{i}" for i in range(1, 5)},
        "audio_prompts": {},
    }

    with pytest.raises(ValueError, match="audio_prompts"):
        parse_generated_story(json.dumps(value), require_audio_prompts=True)


def test_bad_condition_is_deterministic_and_masks_culture():
    profile = {
        "study_number": "S-1",
        "question": "Question",
        "label": "BAD",
        "culture": "Example culture",
        "province": "Example province",
        "religion": "Example religion",
        "occupation": "Farmer",
        "hobbies": "Walking",
    }

    first, first_fields = prepare_profile_for_condition(profile, seed=17)
    second, second_fields = prepare_profile_for_condition(profile, seed=17)

    assert first == second
    assert first_fields == second_fields
    assert first["culture"] == "None"
    assert first["province"] == "None"
    assert first["religion"] == "None"


def test_parses_generated_search_queries():
    assert parse_queries('["one", "two"]') == ["one", "two"]
    assert parse_queries("one; two") == ["one", "two"]


def test_auto_grounding_only_searches_excellent_rows(tmp_path):
    class FakeSearcher:
        def __init__(self):
            self.calls = []

        async def search_web(self, query):
            self.calls.append(query)
            return f"evidence for {query}"

    searcher = FakeSearcher()
    grounder = WebSearchGrounder(
        searcher,
        tmp_path / "cache.json",
        mode="auto",
        country="Mozambique",
        culture="Xichangana",
    )
    contexts = asyncio.run(
        grounder.ground_profiles(
            [
                {
                    "label": "GOOD",
                    "question": "good question",
                    "generated_queries": '["unused"]',
                },
                {
                    "label": "EXCELLENT",
                    "question": "excellent question",
                    "generated_queries": '["searched"]',
                },
            ]
        )
    )

    assert contexts[0].context == ""
    assert contexts[0].error is None
    assert "evidence for searched" in contexts[1].context
    assert searcher.calls == ["searched"]


def test_grounding_failure_is_isolated_to_affected_profile(tmp_path):
    class FakeSearcher:
        async def search_web(self, query):
            if query == "fails":
                raise RuntimeError("search unavailable")
            return f"evidence for {query}"

    grounder = WebSearchGrounder(
        FakeSearcher(),
        tmp_path / "cache.json",
        mode="websearch",
        country="Mozambique",
        culture="Xichangana",
    )
    contexts = asyncio.run(
        grounder.ground_profiles(
            [
                {"question": "one", "generated_queries": '["works"]'},
                {"question": "two", "generated_queries": '["fails"]'},
            ]
        )
    )

    assert contexts[0].error is None
    assert "evidence for works" in contexts[0].context
    assert "search unavailable" in contexts[1].error


def test_azure_settings_are_environment_driven(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "test-deployment")
    monkeypatch.setenv("AZURE_OPENAI_SCOPE", "api://example/.default")

    settings = AzureOpenAISettings.from_env("gpt-4.1")

    assert settings.endpoint == "https://example.invalid"
    assert settings.deployment == "test-deployment"
    assert settings.scope == "api://example/.default"


def test_azure_request_failure_does_not_drop_other_results():
    generator = object.__new__(AzureStoryGenerator)
    generator.semaphore = asyncio.Semaphore(2)

    def generate_one(prompt):
        if prompt == "fails":
            raise RuntimeError("request failed")
        return f"response for {prompt}"

    generator._generate_one = generate_one
    results = asyncio.run(generator.generate(["works", "fails"], batch_size=2))

    assert results[0].text == "response for works"
    assert results[0].error is None
    assert results[1].text is None
    assert results[1].error == "request failed"


def test_deterministic_azure_speech_voice(monkeypatch):
    monkeypatch.delenv("AZURE_SPEECH_VOICE", raising=False)
    assert voice_for("pt-PT", "female") == "pt-PT-RaquelNeural"
    assert voice_for("pt-PT", "male") == "pt-PT-DuarteNeural"


def test_participant_selection_preserves_original_row_index(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text("Question\nfirst\nsecond\n", encoding="utf-8")
    config = PipelineConfig(
        project_id="e321ff7d-3b01-4a85-b815-241a1b9d6007",
        file_path=csv_path,
        model_type="azure",
        model_name="gpt-4.1",
        batch_size=1,
        max_concurrent=1,
        language="en",
        country="Kenya",
        culture="Kikuyu",
        max_new_tokens=100,
        temperature=0,
        seed=42,
        participant_index=1,
        output_dir=tmp_path / "results",
    )

    _, identifiers, original_indices = _load_selected_profiles(config)

    assert identifiers == ["profile-0002"]
    assert original_indices == [2]


def test_bad_prompt_context_does_not_restore_culture(tmp_path):
    config = PipelineConfig(
        project_id="e321ff7d-3b01-4a85-b815-241a1b9d6007",
        file_path=tmp_path / "unused.csv",
        model_type="azure",
        model_name="gpt-4.1",
        batch_size=1,
        max_concurrent=1,
        language="en",
        country="Mozambique",
        culture="Xichangana",
        max_new_tokens=100,
        temperature=0,
        seed=42,
        participant_index=None,
        output_dir=tmp_path / "results",
    )

    country, culture, support = _prompt_context({"label": "BAD"}, config)

    assert country == "Unspecified"
    assert culture == "Unspecified"
    assert "Do not infer culture" in support


def test_safe_paths_and_project_ids():
    assert safe_slug("google/gemma-3-27b-it") == "google-gemma-3-27b-it"
    assert (
        validate_project_id("e321ff7d-3b01-4a85-b815-241a1b9d6007")
        == "e321ff7d-3b01-4a85-b815-241a1b9d6007"
    )
    with pytest.raises(ValueError):
        validate_project_id("../../not-a-uuid")
