from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from directorx.core.models import (
    DialogueLine,
    Keyframe,
    Scene,
    SceneTags,
    Shot,
    StoryAct,
    StorySequence,
    StorySummary,
    TimeRange,
)
from directorx.indexing import (
    AutoTranscriber,
    HashingEmbeddingProvider,
    HybridVideoIndexer,
    SceneSearchStore,
    SceneSearchTools,
    ShotKeyframeSelector,
    SidecarSubtitleTranscriber,
    VisualSceneGrouper,
    validate_story_summary,
)
from directorx.indexing.backends import NoEmbeddedSubtitleError


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

    async def detect(self, video_path: Path, duration_s: float) -> list[Shot]:
        self.calls += 1
        return [
            Shot(id="shot-00000", source_range=TimeRange(start_s=0, end_s=4)),
            Shot(id="shot-00001", source_range=TimeRange(start_s=4, end_s=8)),
            Shot(
                id="shot-00002",
                source_range=TimeRange(start_s=8, end_s=duration_s),
            ),
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


class MissingSidecar:
    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        raise FileNotFoundError(video_path)


class MissingEmbedded:
    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        raise NoEmbeddedSubtitleError(video_path)


class FixedWhisper:
    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        return [DialogueLine(text="ASR", start_s=0, end_s=1)]


class FakeKeyframes:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(
        self, video_path: Path, shots: list[Shot], output_dir: Path
    ) -> dict[str, list[Keyframe]]:
        self.calls += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for shot in shots:
            path = output_dir / f"{shot.id}.jpg"
            path.write_bytes(b"test-frame")
            result[shot.id] = [
                Keyframe(timestamp_s=shot.source_range.start_s + 0.5, path=path)
            ]
        return result


class FixedVisualEmbedder:
    async def embed(self, shots: list[Shot]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][: len(shots)]


class FixedCaptioner:
    def __init__(self) -> None:
        self.calls = 0

    async def caption_batch(self, scenes: list[Scene]) -> dict[str, str]:
        self.calls += 1
        captions = [
            "主角进入昏暗仓库",
            "主角在桌下发现秘密箱子",
            "两人在黎明前逃离仓库",
        ]
        return {scene.id: captions[index] for index, scene in enumerate(scenes)}


class FixedTagger:
    def __init__(self) -> None:
        self.calls = 0

    async def tag_batch(self, scenes: list[Scene]) -> dict[str, SceneTags]:
        self.calls += 1
        return {
            scene.id: SceneTags(
                caption=scene.dense_caption,
                short_summary=(
                    "主角发现秘密箱子" if index == 1 else scene.dense_caption
                ),
                tags=["仓库", "悬疑"],
                characters=["主角"],
                actions=["寻找" if index == 1 else "移动"],
                location="仓库",
                objects=["秘密箱子"] if index == 1 else [],
            )
            for index, scene in enumerate(scenes)
        }


def test_normalize_shots_fills_gaps_without_splitting_a_shot() -> None:
    shots = HybridVideoIndexer._normalize_shots(
        [Shot(id="source", source_range=TimeRange(start_s=2, end_s=42))], 45
    )
    assert [(item.source_range.start_s, item.source_range.end_s) for item in shots] == [
        (0.0, 2.0),
        (2.0, 42.0),
        (42.0, 45.0),
    ]


def test_visual_scene_grouper_merges_adjacent_similar_shots() -> None:
    shots = [
        Shot(id="shot-1", source_range=TimeRange(start_s=0, end_s=3)),
        Shot(id="shot-2", source_range=TimeRange(start_s=3, end_s=6)),
        Shot(id="shot-3", source_range=TimeRange(start_s=6, end_s=9)),
    ]
    grouper = VisualSceneGrouper(
        FixedVisualEmbedder(), similarity_threshold=0.9, max_scene_duration_s=10
    )
    scenes = asyncio.run(grouper.group(shots))

    assert len(scenes) == 3

    class SimilarVisualEmbedder:
        async def embed(self, shots: list[Shot]) -> list[list[float]]:
            return [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]][: len(shots)]

    merged = asyncio.run(
        VisualSceneGrouper(
            SimilarVisualEmbedder(), similarity_threshold=0.9, max_scene_duration_s=10
        ).group(shots)
    )
    assert len(merged) == 2
    assert [shot.id for shot in merged[0].shots] == ["shot-1", "shot-2"]
    assert merged[0].source_range == TimeRange(start_s=0, end_s=6)

    limited = asyncio.run(
        VisualSceneGrouper(
            SimilarVisualEmbedder(), similarity_threshold=0.9, max_scene_duration_s=5
        ).group(shots)
    )
    assert [len(scene.shots) for scene in limited] == [1, 1, 1]


def test_keyframe_count_scales_with_shot_duration() -> None:
    extractor = ShotKeyframeSelector(
        candidate_fps=2.0,
        target_keyframe_interval_s=3.0,
        max_keyframes_per_shot=5,
    )
    assert extractor._keyframe_count(1.0) == 1
    assert extractor._keyframe_count(6.0) == 2
    assert extractor._keyframe_count(10.0) == 4
    assert extractor._keyframe_count(20.0) == 5


def test_keyframe_extractor_selects_sharp_frames_across_duration(
    tmp_path: Path,
) -> None:
    video = tmp_path / "keyframes.mp4"
    _make_video(video, duration=10)
    shot = Shot(
        id="shot-00001",
        source_range=TimeRange(start_s=0, end_s=10),
    )
    extractor = ShotKeyframeSelector(
        candidate_fps=2.0,
        target_keyframe_interval_s=3.0,
        max_keyframes_per_shot=5,
    )
    frames = asyncio.run(extractor.extract(video, [shot], tmp_path / "keyframes"))[
        shot.id
    ]
    assert len(frames) == 4
    assert all(frame.path.exists() for frame in frames)
    assert [frame.timestamp_s for frame in frames] == sorted(
        frame.timestamp_s for frame in frames
    )


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


def test_auto_transcriber_falls_back_to_whisper_when_no_subtitles() -> None:
    lines = asyncio.run(
        AutoTranscriber(MissingSidecar(), MissingEmbedded(), FixedWhisper()).transcribe(
            Path("movie.mp4")
        )
    )

    assert [line.text for line in lines] == ["ASR"]


def test_hybrid_index_cache_and_search_tools(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    _make_video(video)
    detector = FixedDetector()
    transcriber = FixedTranscriber()
    keyframes = FakeKeyframes()
    captioner = FixedCaptioner()
    tagger = FixedTagger()
    embeddings = HashingEmbeddingProvider(dimension=128)
    indexer = HybridVideoIndexer(
        cache_dir=tmp_path / "index-cache",
        shot_detector=detector,
        scene_grouper=VisualSceneGrouper(
            FixedVisualEmbedder(), similarity_threshold=0.9, max_scene_duration_s=4
        ),
        transcriber=transcriber,
        keyframe_extractor=keyframes,
        captioner=captioner,
        tagger=tagger,
        embedding_provider=embeddings,
    )

    first = asyncio.run(indexer.build(video))
    assert len(first.scenes) == 3
    assert first.scenes[1].transcript == "我们必须找到秘密箱子"
    assert first.scenes[1].dense_caption == "主角在桌下发现秘密箱子"
    assert first.scenes[1].short_summary == "主角发现秘密箱子"
    assert first.scenes[1].objects == ["秘密箱子"]
    assert first.search_db_path and first.search_db_path.exists()

    asyncio.run(indexer.build(video))
    assert (
        detector.calls
        == transcriber.calls
        == keyframes.calls
        == captioner.calls
        == tagger.calls
        == 1
    )

    moved_video = tmp_path / "moved" / video.name
    moved_video.parent.mkdir()
    moved_video.write_bytes(video.read_bytes())
    third = asyncio.run(indexer.build(moved_video))
    assert third.video_path == moved_video.resolve()
    assert (
        detector.calls
        == transcriber.calls
        == keyframes.calls
        == captioner.calls
        == tagger.calls
        == 1
    )
    registry = json.loads((tmp_path / "index-cache" / "registry.json").read_text())
    assert registry[video.name]["index_dir"] == f"indexes/{video.name}"

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

    hierarchy = StorySummary(
        title="Indexed story",
        short_summary="A person searches for a box and escapes.",
        sequences=[
            StorySequence(
                id="sequence-0001",
                title="Search",
                short_summary="The group searches the warehouse.",
                scene_ids=["scene-00000", "scene-00001"],
            ),
            StorySequence(
                id="sequence-0002",
                title="Escape",
                short_summary="The group leaves before dawn.",
                scene_ids=["scene-00002"],
            ),
        ],
        acts=[
            StoryAct(
                id="act-0001",
                title="The operation",
                short_summary="The group searches and escapes.",
                sequence_ids=["sequence-0001", "sequence-0002"],
            )
        ],
    )
    hierarchy = validate_story_summary(first, hierarchy)
    asyncio.run(
        SceneSearchStore(first.search_db_path, embeddings).add_story_hierarchy(
            first, hierarchy
        )
    )
    hierarchy_hits = asyncio.run(
        tools.search_hierarchy("searches warehouse", node_type="sequence", limit=2)
    )
    assert hierarchy_hits[0].node_id == "sequence-0001"
