from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

from directorx.core.models import (
    DialogueLine,
    Keyframe,
    Scene,
    SceneAnnotation,
    TimeRange,
)
from directorx.indexing import (
    HashingEmbeddingProvider,
    HybridVideoIndexer,
    SceneSearchStore,
    SceneSearchTools,
    SidecarSubtitleTranscriber,
)


def _make_video(path: Path, duration: int = 12) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x90:rate=12:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )


class FixedDetector:
    def __init__(self) -> None:
        self.calls = 0

    async def detect(self, video_path: Path, duration_s: float) -> list[TimeRange]:
        self.calls += 1
        return [
            TimeRange(start_s=0, end_s=4),
            TimeRange(start_s=4, end_s=8),
            TimeRange(start_s=8, end_s=duration_s),
        ]


class FixedTranscriber:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        self.calls += 1
        return [
            DialogueLine(text="我们必须找到秘密箱子", start_s=4.5, end_s=6.0),
            DialogueLine(text="天亮以前离开这里", start_s=8.5, end_s=10.0),
        ]


class FakeKeyframes:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self, video_path: Path, scenes: list[Scene], output_dir: Path
    ) -> dict[str, list[Keyframe]]:
        self.calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for scene in scenes:
            path = output_dir / f"{scene.id}.jpg"
            path.write_bytes(b"test-frame")
            result[scene.id] = [
                Keyframe(timestamp_s=scene.source_range.start_s + 0.5, path=path)
            ]
        return result


class FixedAnnotator:
    def __init__(self) -> None:
        self.calls = 0

    async def annotate_batch(self, scenes: list[Scene]) -> dict[str, SceneAnnotation]:
        self.calls += 1
        captions = [
            "主角进入昏暗仓库",
            "主角在桌下发现秘密箱子",
            "两人在黎明前逃离仓库",
        ]
        return {
            scene.id: SceneAnnotation(
                caption=captions[index],
                tags=["仓库", "悬疑"],
                characters=["主角"],
                actions=["寻找" if index == 1 else "移动"],
                location="仓库",
                mood_scores={"tense": 0.8},
                plot_event="发现线索" if index == 1 else None,
                confidence=0.9,
            )
            for index, scene in enumerate(scenes)
        }


def test_normalize_ranges_fills_gaps_and_splits_long_shots() -> None:
    ranges = HybridVideoIndexer._normalize_ranges(
        [TimeRange(start_s=2, end_s=42)], 45, max_scene_duration_s=15
    )
    assert [(item.start_s, item.end_s) for item in ranges] == [
        (0.0, 2.0),
        (2.0, 17.0),
        (17.0, 32.0),
        (32.0, 42.0),
        (42.0, 45.0),
    ]


def test_sidecar_subtitles_discover_preferred_language_and_parse(
    tmp_path: Path,
) -> None:
    video = tmp_path / "feature.mp4"
    video.write_bytes(b"placeholder")
    subtitles = tmp_path / "Subs"
    subtitles.mkdir()
    (subtitles / "9_French.srt").write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nBonjour.\n", encoding="utf-8"
    )
    english = subtitles / "5_English.srt"
    english.write_text(
        "1\n00:00:03,000 --> 00:00:04,500\nThe name is Bond.\n", encoding="utf-8"
    )

    lines = asyncio.run(SidecarSubtitleTranscriber().transcribe(video))
    assert len(lines) == 1
    assert lines[0].text == "The name is Bond."
    assert lines[0].language == "eng"

    explicit = asyncio.run(
        SidecarSubtitleTranscriber(subtitles / "9_French.srt").transcribe(video)
    )
    assert explicit[0].text == "Bonjour."


def test_screenwriter_context_covers_entire_feature() -> None:
    scenes = [
        Scene(
            id=f"scene-{index:04d}",
            source_range=TimeRange(start_s=index * 10, end_s=(index + 1) * 10),
            caption=f"scene {index}",
            transcript="dialogue",
        )
        for index in range(400)
    ]
    from directorx.services.providers import OpenAICompatibleScreenwriterModel

    selected = OpenAICompatibleScreenwriterModel._select_context_scenes(
        scenes, limit=40
    )
    assert len(selected) == 40
    assert selected[0].source_range.start_s < 100
    assert selected[-1].source_range.start_s > 3800


def test_hybrid_index_cache_and_search_tools(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "movie.mp4"
    _make_video(video)
    detector = FixedDetector()
    transcriber = FixedTranscriber()
    keyframes = FakeKeyframes()
    annotator = FixedAnnotator()
    embeddings = HashingEmbeddingProvider(dimension=128)
    indexer = HybridVideoIndexer(
        cache_dir=tmp_path / "index-cache",
        scene_detector=detector,
        transcriber=transcriber,
        keyframe_extractor=keyframes,
        annotator=annotator,
        embedding_provider=embeddings,
    )

    first = asyncio.run(indexer.build(video))
    assert len(first.scenes) == 3
    assert first.scenes[1].transcript == "我们必须找到秘密箱子"
    assert first.scenes[1].mood == "tense"
    assert first.search_db_path and first.search_db_path.exists()

    def unexpected_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        raise AssertionError("unchanged path metadata should not re-hash the movie")

    monkeypatch.setattr(indexer.cache, "_sha256", unexpected_hash)
    second = asyncio.run(indexer.build(video))
    assert second.content_fingerprint == first.content_fingerprint
    assert (
        detector.calls == transcriber.calls == keyframes.calls == annotator.calls == 1
    )

    monkeypatch.undo()
    current = video.stat()
    os.utime(video, ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000))
    third = asyncio.run(indexer.build(video))
    assert third.content_fingerprint == first.content_fingerprint
    assert (
        detector.calls == transcriber.calls == keyframes.calls == annotator.calls == 1
    )

    tools = SceneSearchTools(SceneSearchStore(first.search_db_path, embeddings))
    scene_hits = asyncio.run(tools.search_scenes("发现秘密箱子", limit=2))
    dialogue_hits = asyncio.run(tools.search_dialogue("秘密箱子", limit=2))
    assert scene_hits[0].scene_id == "scene-00001"
    assert dialogue_hits[0].scene_id == "scene-00001"
    assert all(hit.transcript for hit in dialogue_hits)
    assert (
        "秘密" in scene_hits[0].matched_terms
        or "秘密箱子" in scene_hits[0].matched_terms
    )
    inspection = tools.inspect_scene("scene-00001")
    assert inspection.previous_scene_id == "scene-00000"
    assert inspection.next_scene_id == "scene-00002"
    assert inspection.scene.keyframes[0].path.exists()
