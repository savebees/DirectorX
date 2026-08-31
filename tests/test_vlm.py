from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from directorx.core.models import Keyframe, Scene, TimeRange
from directorx.indexing.vlm import OpenAICompatibleDenseCaptioner
from directorx.services.providers import OpenAICompatibleSceneTagger


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(
            content="一名人物站在红色测试画面前，字幕提示不要打开那扇门。",
            refusal=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def _scene(tmp_path: Path) -> Scene:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 12), "red").save(image, format="JPEG")
    return Scene(
        id="scene-00001",
        source_range=TimeRange(start_s=1, end_s=4),
        caption="",
        transcript="不要打开那扇门",
        keyframes=[Keyframe(timestamp_s=2, path=image)],
    )


def test_vlm_returns_plain_text_and_sends_images(tmp_path: Path) -> None:
    client = FakeClient()
    captioner = OpenAICompatibleDenseCaptioner(client=client)
    captions = asyncio.run(captioner.caption_batch([_scene(tmp_path)]))

    assert captions["scene-00001"].startswith("一名人物")
    call = client.completions.calls[0]
    assert call["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert "response_format" not in call
    assert "English" in call["messages"][0]["content"]
    assert "Time range" not in call["messages"][0]["content"]
    image_part = call["messages"][1]["content"][0]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vlm_requires_runtime_secret(monkeypatch) -> None:
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VLM_API_KEY"):
        OpenAICompatibleDenseCaptioner()


class TaggerCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(
            {
                "caption": (
                    "A person stands inside a warehouse while warning someone "
                    "not to open a door."
                ),
                "short_summary": "A person warns someone about a closed door.",
                "tags": ["warehouse", "warning", "door"],
                "characters": ["person"],
                "actions": ["standing", "warning"],
                "location": "warehouse",
                "objects": ["door"],
            }
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_scene_tagger_uses_typed_json_after_dense_caption(tmp_path: Path) -> None:
    completions = TaggerCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    scene = _scene(tmp_path)
    scene.dense_caption = "人物站在仓库里。"
    tagger = OpenAICompatibleSceneTagger(client=client)
    tags = asyncio.run(tagger.tag_batch([scene]))

    assert tags[scene.id].location == "warehouse"
    assert "response_format" not in completions.calls[0]
    assert "JSON Schema" in completions.calls[0]["messages"][0]["content"]
    assert len(tags[scene.id].caption) > 20
    assert tags[scene.id].short_summary == (
        "A person warns someone about a closed door."
    )
    assert tags[scene.id].tags == ["warehouse", "warning", "door"]
    prompt = completions.calls[0]["messages"][0]["content"]
    assert "do not quote" in prompt.lower()
    assert "two" not in prompt.lower()
    assert (
        "dense visual caption" in completions.calls[0]["messages"][1]["content"].lower()
    )
    assert "one concise" in prompt.lower()
