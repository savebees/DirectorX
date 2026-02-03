from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from directorx.agents import DirectorAgent, FootageAnalystAgent, GroundingAgent
from directorx.agents.grounding import SceneRetriever
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import (
    GroundingCandidate,
    GroundingDecision,
    GroundingFrame,
    GroundingManifest,
    NarrationManifest,
    NarrationSegment,
    Scene,
    ShotRequest,
    StoryAct,
    StoryBeat,
    Storyboard,
    StorySequence,
    StorySummary,
    TimeRange,
    VideoIndex,
)
from directorx.indexing import HashingEmbeddingProvider, SceneSearchStore
from directorx.indexing.store import scene_document
from directorx.services.grounding import (
    FFmpegGroundingFrameExtractor,
    OpenAICompatibleGroundingModel,
)
from tests.fakes import FixedGroundingModel


class UnusedIndexer:
    async def build(self, video_path: Path) -> VideoIndex:
        raise AssertionError("The footage agent is not used in this test")


class FakeFrameExtractor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def extract(
        self,
        video_path: Path,
        source_range: TimeRange,
        output_dir: Path,
        *,
        fps: float,
        max_frames: int,
        prefix: str,
    ) -> list[GroundingFrame]:
        self.calls.append(
            {
                "video_path": video_path,
                "source_range": source_range,
                "fps": fps,
                "max_frames": max_frames,
                "prefix": prefix,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamps = [
            source_range.start_s,
            source_range.start_s + source_range.duration_s / 2,
            source_range.end_s - 0.001,
        ]
        frames = []
        for number, timestamp_s in enumerate(timestamps, start=1):
            frame_id = f"{prefix}-{number:04d}"
            path = output_dir / f"{frame_id}.jpg"
            Image.new("RGB", (24, 16), "blue").save(path, format="JPEG")
            frames.append(
                GroundingFrame(id=frame_id, timestamp_s=timestamp_s, path=path)
            )
        return frames


def _source_artifacts(tmp_path: Path) -> dict[str, Path]:
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    movie_path = tmp_path / "movie.mp4"
    movie_path.write_bytes(b"fake movie")
    scenes = [
        Scene(
            id="scene-0001",
            source_range=TimeRange(start_s=0, end_s=10),
            caption="A traveler enters an empty station.",
            short_summary="A traveler arrives at the station.",
            tags=["traveler", "station"],
        ),
        Scene(
            id="scene-0002",
            source_range=TimeRange(start_s=10, end_s=20),
            caption="The station lights fail around the waiting traveler.",
            short_summary="The station falls into darkness.",
            tags=["station", "blackout", "darkness"],
        ),
    ]
    provider = HashingEmbeddingProvider(dimension=64)
    index = VideoIndex(video_path=movie_path, duration_s=20, scenes=scenes)
    embeddings = asyncio.run(
        provider.embed([scene_document(scene) for scene in scenes])
    )
    search_path = footage_dir / "search.sqlite3"
    SceneSearchStore(search_path, provider).build(index, embeddings)
    index.search_db_path = search_path
    index_path = footage_dir / "index.json"
    index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")

    summary = StorySummary(
        title="The Last Train",
        short_summary="A traveler waits through an unexpected station blackout.",
        sequences=[
            StorySequence(
                id="sequence-0001",
                title="Waiting",
                short_summary="An arrival turns into an uneasy wait.",
                scene_ids=["scene-0001", "scene-0002"],
            )
        ],
        acts=[
            StoryAct(
                id="act-0001",
                title="Blackout",
                short_summary="The empty station becomes threatening.",
                sequence_ids=["sequence-0001"],
            )
        ],
    )
    summary_path = footage_dir / "story-summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    storyboard = Storyboard(
        title="The Wait",
        logline="A routine wait becomes a confrontation with darkness.",
        narrative_angle="Build tension around uncertainty.",
        beats=[
            StoryBeat(
                id="beat-1",
                purpose="Escalate tension",
                story_content="The station lights fail while the traveler waits.",
                narration="Then every light went out.",
                visual_intent="A waiting traveler as the station falls into darkness",
                mood="tense",
                target_duration_s=5,
                source_sequence_ids=["sequence-0001"],
                evidence_scene_ids=["scene-0002"],
            )
        ],
        target_duration_s=5,
    )
    storyboard_path = tmp_path / "screenwriter" / "storyboard.json"
    storyboard_path.parent.mkdir()
    storyboard_path.write_text(storyboard.model_dump_json(indent=2), encoding="utf-8")

    narration = NarrationManifest(
        segments=[
            NarrationSegment(
                beat_id="beat-1",
                text="Then every light went out.",
                audio_path=tmp_path / "narration" / "beat-1.wav",
                target_duration_s=5,
                duration_s=12,
            )
        ],
        target_duration_s=5,
        duration_s=12,
    )
    narration_path = tmp_path / "narration" / "narration.json"
    narration_path.parent.mkdir()
    narration_path.write_text(narration.model_dump_json(indent=2), encoding="utf-8")
    return {
        "index": index_path,
        "search": search_path,
        "summary": summary_path,
        "storyboard": storyboard_path,
        "narration": narration_path,
    }


def _task(paths: dict[str, Path], task_id: str = "grounding-001") -> TaskContext:
    return TaskContext(
        task_id=task_id,
        assignee=AgentRole.GROUNDING,
        objective="Ground every approved visual beat in the source film.",
        input_artifacts=[
            ArtifactRef(name="video-index", path=paths["index"]),
            ArtifactRef(name="scene-search-database", path=paths["search"]),
            ArtifactRef(name="story-summary", path=paths["summary"]),
            ArtifactRef(name="storyboard", path=paths["storyboard"]),
            ArtifactRef(name="narration-manifest", path=paths["narration"]),
        ],
        expected_output="Exact visually verified source intervals.",
        acceptance_criteria=["Use the real narration duration as timing context"],
    )


def _agent(
    model: FixedGroundingModel | None = None,
    extractor: FakeFrameExtractor | None = None,
) -> GroundingAgent:
    return GroundingAgent(
        model or FixedGroundingModel(),
        SceneRetriever(HashingEmbeddingProvider(dimension=64)),
        extractor or FakeFrameExtractor(),
        candidate_limit=4,
        candidate_padding_s=1,
        coarse_fps=1,
        refine_fps=6,
        refine_margin_s=2,
        max_coarse_frames=24,
        max_refine_frames=24,
        max_parallel=1,
    )


def test_scene_retriever_combines_screenwriter_evidence_and_hybrid_search(
    tmp_path: Path,
) -> None:
    paths = _source_artifacts(tmp_path)
    index = VideoIndex.model_validate_json(paths["index"].read_text(encoding="utf-8"))
    summary = StorySummary.model_validate_json(
        paths["summary"].read_text(encoding="utf-8")
    )
    shot = ShotRequest(
        id="shot-beat-1",
        beat_id="beat-1",
        narration_text="Then every light went out.",
        story_content="The station lights fail.",
        visual_query="A station falling into darkness",
        mood="tense",
        target_duration_s=12,
        source_sequence_ids=["sequence-0001"],
        evidence_scene_ids=["scene-0002"],
    )

    candidates = asyncio.run(
        SceneRetriever(HashingEmbeddingProvider(dimension=64)).search(
            shot, index, summary, limit=4, padding_s=1
        )
    )

    assert candidates[0].anchor_scene_id == "scene-0002"
    assert candidates[0].retrieval_score == 1
    assert {candidate.anchor_scene_id for candidate in candidates} == {
        "scene-0001",
        "scene-0002",
    }


def test_director_delegates_grounding_and_agent_persists_its_result(
    tmp_path: Path,
) -> None:
    paths = _source_artifacts(tmp_path)
    model = FixedGroundingModel()
    extractor = FakeFrameExtractor()
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(
        runtime,
        FootageAnalystAgent(UnusedIndexer()),
        artifacts_dir=tmp_path / "artifacts",
        grounding_agent=_agent(model, extractor),
    )

    result = asyncio.run(director.run_grounding_task(_task(paths)))

    manifest_path = tmp_path / "artifacts" / "grounding.json"
    manifest = GroundingManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    assert result.status == "completed"
    assert result.agent == AgentRole.GROUNDING
    assert result.output_artifacts == [
        ArtifactRef(name="grounding-manifest", path=manifest_path)
    ]
    assert manifest.source_video == tmp_path / "movie.mp4"
    assert manifest.target_duration_s == 12
    assert manifest.source_duration_s == 2
    assert manifest.clips[0].source_range == TimeRange(start_s=12.5, end_s=14.5)
    assert manifest.clips[0].target_duration_s == 12
    assert manifest.clips[0].source_scene_ids == ["scene-0002"]
    assert len(manifest.clips[0].evidence_timestamps_s) == 2
    assert [call[0] for call in model.calls] == ["locate", "locate", "refine"]
    assert model.calls[0][1].target_duration_s == 12
    assert model.calls[0][1].evidence_scene_ids == ["scene-0002"]
    assert [call["fps"] for call in extractor.calls] == [1, 1, 6]
    assert all(call["video_path"] == tmp_path / "movie.mp4" for call in extractor.calls)
    assert list((tmp_path / "artifacts").glob(".grounding-frames.*")) == []
    assert director.read_result("grounding-001") == result
    result_payload = json.loads(
        (tmp_path / "coordination" / "tasks" / "grounding-001.result.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(result_payload) == {
        "task_id",
        "agent",
        "status",
        "summary",
        "output_artifacts",
    }


@pytest.mark.parametrize("stage", ["locate", "refine"])
def test_grounding_persists_blocked_result_when_vlm_fails(
    tmp_path: Path, stage: str
) -> None:
    paths = _source_artifacts(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(paths, f"grounding-{stage}-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        _agent(FixedGroundingModel(failing_stage=stage)).run_task(
            task, runtime, tmp_path / "artifacts"
        )
    )

    assert result.status == "blocked"
    assert result.agent == AgentRole.GROUNDING
    assert result.output_artifacts == []
    assert "Grounding blocked:" in result.summary
    assert not (tmp_path / "artifacts" / "grounding.json").exists()
    assert list((tmp_path / "artifacts").glob(".grounding-frames.*")) == []
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_grounding_blocks_when_required_artifact_cannot_load(tmp_path: Path) -> None:
    paths = _source_artifacts(tmp_path)
    paths["narration"] = tmp_path / "missing" / "narration.json"
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(paths, "grounding-load-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(_agent().run_task(task, runtime, tmp_path / "artifacts"))

    assert result.status == "blocked"
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_grounding_does_not_overwrite_existing_manifest(tmp_path: Path) -> None:
    paths = _source_artifacts(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    manifest_path = artifacts / "grounding.json"
    manifest_path.write_text("keep", encoding="utf-8")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(paths, "grounding-persistence-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)
    model = FixedGroundingModel()

    result = asyncio.run(_agent(model).run_task(task, runtime, artifacts))

    assert result.status == "blocked"
    assert manifest_path.read_text(encoding="utf-8") == "keep"
    assert model.calls == []
    assert list(artifacts.glob(".grounding-frames.*")) == []


def test_grounding_rejects_another_agents_task(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = TaskContext(
        task_id="narration-boundary",
        assignee=AgentRole.NARRATION,
        objective="Synthesize narration.",
        expected_output="Narration audio.",
        acceptance_criteria=["Return audio"],
    )

    with pytest.raises(ValueError):
        asyncio.run(_agent().run_task(task, runtime, tmp_path / "artifacts"))


def test_director_cannot_submit_a_grounding_result(tmp_path: Path) -> None:
    paths = _source_artifacts(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(paths, "grounding-submit-boundary")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = TaskResult(
        task_id=task.task_id,
        agent=AgentRole.GROUNDING,
        status="completed",
        summary="Grounded the approved visual intent.",
    )

    with pytest.raises(PermissionError):
        runtime.submit_result(AgentRole.DIRECTOR, result)


class GroundingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses = [
            GroundingDecision(
                matched=True,
                source_range=TimeRange(start_s=11, end_s=15),
                confidence=0.8,
                evidence_frame_ids=["frame-1"],
                rationale="The lights visibly fail.",
            ),
            GroundingDecision(
                matched=True,
                source_range=TimeRange(start_s=12, end_s=14),
                confidence=0.9,
                evidence_frame_ids=["frame-1"],
                rationale="Dense frames confirm the boundaries.",
            ),
        ]

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[len(self.calls) - 1]
        message = SimpleNamespace(content=response.model_dump_json(), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_qwen_grounding_vlm_uses_timestamped_frames_and_role_prompts(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (20, 12), "black").save(image_path, format="JPEG")
    frames = [GroundingFrame(id="frame-1", timestamp_s=12, path=image_path)]
    candidate = GroundingCandidate(
        id="candidate-0001",
        anchor_scene_id="scene-0002",
        scene_ids=["scene-0002"],
        source_range=TimeRange(start_s=10, end_s=20),
        retrieval_score=1,
    )
    shot = ShotRequest(
        id="shot-beat-1",
        beat_id="beat-1",
        narration_text="Then every light went out.",
        story_content="The station falls into darkness.",
        visual_query="A station falling into darkness",
        mood="tense",
        target_duration_s=12,
        source_sequence_ids=["sequence-0001"],
        evidence_scene_ids=["scene-0002"],
    )
    completions = GroundingCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAICompatibleGroundingModel(client=client)

    located = asyncio.run(model.locate(shot, candidate, frames))
    refined = asyncio.run(model.refine(shot, candidate, frames))

    assert located.source_range == TimeRange(start_s=11, end_s=15)
    assert refined.source_range == TimeRange(start_s=12, end_s=14)
    assert len(completions.calls) == 2
    assert all(
        call["model"] == "Qwen/Qwen3-VL-8B-Instruct" for call in completions.calls
    )
    assert (
        completions.calls[0]["messages"][0]["content"]
        == OpenAICompatibleGroundingModel.LOCATE_SYSTEM_PROMPT
    )
    assert (
        completions.calls[1]["messages"][0]["content"]
        == OpenAICompatibleGroundingModel.REFINE_SYSTEM_PROMPT
    )
    user_content = completions.calls[0]["messages"][1]["content"]
    assert "frame-1=12.000s" in user_content[0]["text"]
    assert user_content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


def test_ffmpeg_grounding_extractor_labels_exact_timestamps(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "movie.mp4"
    video_path.write_bytes(b"fake movie")
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        Image.new("RGB", (32, 20), "red").save(arguments[-1], format="JPEG")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("directorx.services.grounding.subprocess.run", fake_run)
    extractor = FFmpegGroundingFrameExtractor(max_image_dimension=24)

    frames = asyncio.run(
        extractor.extract(
            video_path,
            TimeRange(start_s=2, end_s=4),
            tmp_path / "frames",
            fps=1,
            max_frames=10,
            prefix="candidate-0001-coarse",
        )
    )

    assert [frame.timestamp_s for frame in frames] == [2, 3]
    assert all(frame.path.is_file() for frame in frames)
    assert [call[0][call[0].index("-ss") + 1] for call in calls] == [
        "2.000000",
        "3.000000",
    ]
    with Image.open(frames[0].path) as image:
        assert max(image.size) <= 24
