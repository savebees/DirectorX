from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from directorx.core.models import Keyframe, Scene, TimeRange
from directorx.indexing.vlm import OpenAICompatibleSceneAnnotator

ANNOTATION_JSON = """{
  "caption": "一名人物站在红色测试画面前。",
  "tags": ["人物", "红色背景"],
  "characters": ["person_1"],
  "actions": ["站立"],
  "location": null,
  "objects": [],
  "mood_scores": {"neutral": 0.8},
  "plot_event": null,
  "confidence": 0.75
}"""


class FakeSchemaError(Exception):
    status_code = 400


class FakeCompletions:
    def __init__(self, reject_schema: bool = False) -> None:
        self.reject_schema = reject_schema
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if (
            self.reject_schema
            and kwargs.get("response_format", {}).get("type") == "json_schema"
        ):
            raise FakeSchemaError("response_format json_schema is unsupported")
        if kwargs.get("stream"):

            async def chunks():
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content=ANNOTATION_JSON))
                    ]
                )

            return chunks()
        message = SimpleNamespace(content=ANNOTATION_JSON, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, reject_schema: bool = False) -> None:
        self.completions = FakeCompletions(reject_schema)
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


def test_vlm_uses_images_and_strict_json_schema(tmp_path: Path) -> None:
    client = FakeClient()
    annotator = OpenAICompatibleSceneAnnotator(client=client)
    annotations = asyncio.run(annotator.annotate_batch([_scene(tmp_path)]))

    assert annotations["scene-00001"].caption.startswith("一名人物")
    call = client.completions.calls[0]
    assert call["model"] == "Qwen/Qwen3.6-35B-A3B"
    assert call["response_format"]["type"] == "json_schema"
    image_part = call["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vlm_surfaces_schema_compatibility_error(tmp_path: Path) -> None:
    client = FakeClient(reject_schema=True)
    annotator = OpenAICompatibleSceneAnnotator(client=client)
    with pytest.raises(FakeSchemaError):
        asyncio.run(annotator.annotate_batch([_scene(tmp_path)]))
    assert [call["response_format"]["type"] for call in client.completions.calls] == [
        "json_schema"
    ]


def test_vlm_requires_runtime_secret(monkeypatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        OpenAICompatibleSceneAnnotator()


def test_vlm_rejects_provider_schema_variants() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleSceneAnnotator._parse_annotation(
            '{"caption":"x","tags":null,"characters":null,"actions":"站立",'
            '"location":null,"objects":[],"mood_scores":{},'
            '"plot_event":null,"confidence":0.8}'
        )
