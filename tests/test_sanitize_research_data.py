import json

from scripts.sanitize_research_data import (
    assert_no_direct_identifiers,
    eligible_annotators,
    meaningful_judgment,
    sanitize_language,
)


def test_sanitizes_human_and_llm_judgments(tmp_path):
    raw_id = "raw-task-id"
    human = {
        "tasks": [
            {
                "id": raw_id,
                "description": (
                    "<tr><td>What is your preferred name?</td>"
                    "<td>Private Name</td></tr>"
                ),
                "text": "Private Name changed a daily habit with Private Faith.",
                "paragraphs": [
                    "Private Name changed a daily habit with Private Faith."
                ],
                "images": ["https://example.test/image.png"],
                "audio": [],
                "question": "What should Private Name eat?",
                "category": "diet",
                "taskType": "independent",
                "metadata": {
                    "label": "GOOD",
                    "model": "example-model",
                    "prompt": "private prompt",
                    "story_arc": "classic_change",
                },
            }
        ],
        "annotations": [
            {
                "userEmail": "annotator@example.com",
                "userId": "private-user-id",
                "completedTaskIds": [raw_id],
                "taskData": {
                    raw_id: {
                        "culturalScore": 75,
                        "generalComment": (
                            "Private Name is relevant. annotator@example.com"
                        ),
                        "annotations": [
                            {
                                "id": "annotation-id",
                                "comment": "Private Name is appropriate.",
                                "userEmail": "annotator@example.com",
                                "userId": "private-user-id",
                                "taskId": raw_id,
                            }
                        ],
                        "imageAnnotations": {
                            "0": [
                                {
                                    "id": "image-annotation-id",
                                    "comment": (
                                        "Private Name wears Private Faith clothing."
                                    ),
                                    "userEmail": "annotator@example.com",
                                }
                            ]
                        },
                        "textConnectness": {"0": "connected"},
                        "imageConnectness": {},
                        "globalFeedback": {"story_engaging": True},
                    }
                },
            }
        ],
    }
    human_path = tmp_path / "human.json"
    human_path.write_text(json.dumps(human), encoding="utf-8")

    llm = {
        "story_id": raw_id,
        "input": {
            "story": "Private Name changed a daily habit.",
            "culture": "Example",
            "person": {
                "id": "75",
                "name": "Private Name",
                "study_number": "private-study-number",
                "religion": "Private Faith",
                "other_health_conditions": "Example Culture",
            },
            "question": "What should Private Name eat?",
            "category": "diet",
            "taskType": "independent",
        },
        "output": {
            "score": 75,
            "reasoning": "Private Name is culturally aligned.",
        },
        "metadata": {
            "configuration": "test",
            "consensus_reached": True,
            "total_debate_rounds": 0,
            "initial_scores": {"judge": 75},
            "agent_final_scores": {"judge": 75},
            "debate_transcript": [],
        },
    }
    llm_path = tmp_path / "llm.jsonl"
    llm_path.write_text(json.dumps(llm) + "\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "image.png").write_bytes(b"test image")

    summary = sanitize_language(
        "example",
        {
            "language_code": "en",
            "country": "Example Country",
            "culture": "Example Culture",
        },
        human_path,
        llm_path,
        output_dir,
        image_dir=image_dir,
    )
    assert_no_direct_identifiers(output_dir)

    human_output = (
        output_dir / "example" / "human_annotations.jsonl"
    ).read_text(encoding="utf-8")
    llm_output = (
        output_dir / "example" / "llm_judgments.jsonl"
    ).read_text(encoding="utf-8")
    combined = human_output + llm_output

    assert summary["stories"] == 1
    assert summary["anonymous_annotators"] == 1
    assert "example_annotator_01" in human_output
    assert "Private Name" not in combined
    assert "annotator@example.com" not in combined
    assert "private-user-id" not in combined
    assert "private-study-number" not in combined
    assert "Private Faith" not in combined
    assert "[REDACTED_PROFILE_ATTRIBUTE]" in combined
    assert '"culture": "Example Culture"' in combined
    assert '"person"' not in llm_output
    assert "https://example.test/image.png" not in human_output
    assert '"image_count": 1' in human_output
    assert '"image_paths": ["data/images/example_story_001_00.png"]' in combined
    assert not (image_dir / "image.png").exists()
    assert (image_dir / "example_story_001_00.png").is_file()
    assert summary["local_images"] == 1


def test_redacts_participant_name_across_language_records(tmp_path):
    human = {
        "tasks": [
            {
                "id": "first",
                "description": "<tr><td>Name</td><td>Unique Person</td></tr>",
                "text": "A story without the name.",
                "paragraphs": [],
                "images": [],
                "audio": [],
                "metadata": {},
            },
            {
                "id": "second",
                "description": "",
                "text": "Unique Person appears in another story.",
                "paragraphs": [],
                "images": [],
                "audio": [],
                "metadata": {},
            },
        ],
        "annotations": [],
    }
    human_path = tmp_path / "human.json"
    human_path.write_text(json.dumps(human), encoding="utf-8")
    llm_path = tmp_path / "llm.jsonl"
    llm_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "story_id": story_id,
                    "input": {"story": "", "person": {}},
                    "output": {},
                    "metadata": {},
                }
            )
            for story_id in ("first", "second")
        )
        + "\n",
        encoding="utf-8",
    )

    sanitize_language(
        "example",
        {
            "language_code": "en",
            "country": "Example Country",
            "culture": "Example Culture",
        },
        human_path,
        llm_path,
        tmp_path / "output",
    )

    output = (
        tmp_path / "output" / "example" / "human_annotations.jsonl"
    ).read_text(encoding="utf-8")
    assert "Unique Person" not in output
    assert "Unique" not in output
    assert '"image_paths"' not in output


def test_zero_score_is_meaningful():
    assert meaningful_judgment({"culturalScore": 0})


def test_annotator_threshold_uses_half_rounded_up():
    task_data = {
        "one": {"culturalScore": 0},
        "two": {},
        "three": {},
    }
    human_export = {
        "tasks": [{"id": "one"}, {"id": "two"}, {"id": "three"}],
        "annotations": [
            {
                "completedTaskIds": ["one"],
                "taskData": task_data,
            }
        ],
    }

    included, excluded = eligible_annotators(human_export)

    assert included == []
    assert excluded == 1
