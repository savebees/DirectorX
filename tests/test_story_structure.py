import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

from directorx.core.models import Scene, TimeRange, VideoIndex
from directorx.services.providers import OpenAICompatibleStoryStructureModel


class StoryStructureCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        user_content = kwargs["messages"][1]["content"]
        if user_content.startswith("Scene evidence chunk"):
            scene_ids = list(dict.fromkeys(re.findall(r"scene-\d+", user_content)))
            payload = {
                "sequences": [
                    {
                        "title": "The warning",
                        "short_summary": "A person warns another person about a door.",
                        "scene_ids": scene_ids,
                    }
                ]
            }
        else:
            sequence_ids = list(
                dict.fromkeys(re.findall(r"sequence-\d+", user_content))
            )
            payload = {
                "title": "The Door",
                "short_summary": "A warning creates uncertainty about a closed door.",
                "acts": [
                    {
                        "title": "The warning",
                        "short_summary": "A person delivers a warning.",
                        "sequence_ids": sequence_ids,
                    }
                ],
                "character_arcs": [],
                "major_events": [],
            }
        message = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_story_structure_sends_only_scene_caption_and_tags() -> None:
    completions = StoryStructureCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAICompatibleStoryStructureModel(client=client)
    index = VideoIndex(
        video_path=Path("movie.mp4"),
        duration_s=10,
        scenes=[
            Scene(
                id="scene-00001",
                source_range=TimeRange(start_s=1, end_s=9),
                caption="A person warns someone not to open a closed door.",
                short_summary="A warning is delivered.",
                dense_caption="A red door fills the frame.",
                transcript="Do not open that door.",
                tags=["warning", "closed door"],
                characters=["person"],
                actions=["warning"],
                location="warehouse",
                objects=["door"],
            )
        ],
    )

    summary = asyncio.run(model.build(index))

    evidence = completions.calls[0]["messages"][1]["content"]
    assert "scene-00001" in evidence
    assert index.scenes[0].caption in evidence
    assert "warning" in evidence
    assert "closed door" in evidence
    assert "A red door fills the frame" not in evidence
    assert "A warning is delivered" not in evidence
    assert "Do not open that door" not in evidence
    assert "warehouse" not in evidence
    assert "characters=" not in evidence
    assert "actions=" not in evidence
    assert "location=" not in evidence
    assert "objects=" not in evidence
    assert "1.0-9.0" not in evidence
    assert "captions and tags" in completions.calls[0]["messages"][0]["content"]
    assert summary.sequences[0].id == "sequence-0001"
    assert summary.acts[0].id == "act-0001"


def test_story_structure_honors_configured_chunk_size() -> None:
    completions = StoryStructureCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAICompatibleStoryStructureModel(client=client, max_scenes_per_chunk=1)
    scenes = [
        Scene(
            id=f"scene-{number:05d}",
            source_range=TimeRange(start_s=number, end_s=number + 1),
            caption=f"Scene {number}",
            tags=[f"tag-{number}"],
        )
        for number in range(2)
    ]

    summary = asyncio.run(
        model.build(
            VideoIndex(video_path=Path("movie.mp4"), duration_s=2, scenes=scenes)
        )
    )

    assert len(completions.calls) == 3
    assert [sequence.id for sequence in summary.sequences] == [
        "sequence-0001",
        "sequence-0002",
    ]
