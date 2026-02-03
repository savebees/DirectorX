import asyncio
from pathlib import Path

import pytest

from directorx.agents import DirectorAgent, FootageAnalystAgent, NarrationAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import (
    NarrationManifest,
    StoryBeat,
    Storyboard,
    VideoIndex,
)


class UnusedIndexer:
    async def build(self, video_path: Path) -> VideoIndex:
        raise AssertionError("The footage agent is not used in this test")


class FakeTTS:
    def __init__(
        self,
        durations: dict[str, float] | None = None,
        failing_text: str | None = None,
        write_audio: bool = True,
    ) -> None:
        self.durations = durations or {}
        self.failing_text = failing_text
        self.write_audio = write_audio
        self.calls: list[tuple[str, Path]] = []

    async def synthesize(self, text: str, output_path: Path) -> float:
        self.calls.append((text, output_path))
        if text == self.failing_text:
            raise RuntimeError("tts unavailable")
        if self.write_audio:
            output_path.write_bytes(b"fake wav")
        return self.durations.get(text, 1.0)


def _storyboard() -> Storyboard:
    return Storyboard(
        title="The Wait",
        logline="A traveler waits through a station blackout.",
        narrative_angle="Build tension around uncertainty.",
        beats=[
            StoryBeat(
                id="beat-1",
                purpose="Establish uncertainty",
                story_content="The traveler arrives at an empty station.",
                narration="The station should have been busy.",
                visual_intent="An empty station platform",
                mood="uneasy",
                target_duration_s=5,
                source_sequence_ids=["sequence-0001"],
                evidence_scene_ids=["scene-0001"],
            ),
            StoryBeat(
                id="beat-2",
                purpose="Escalate tension",
                story_content="The station lights fail.",
                narration="Then every light went out.",
                visual_intent="A station falling into darkness",
                mood="tense",
                target_duration_s=7,
                source_sequence_ids=["sequence-0002"],
                evidence_scene_ids=["scene-0002"],
            ),
        ],
        target_duration_s=12,
    )


def _storyboard_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "screenwriter" / "storyboard.json"
    path.parent.mkdir()
    path.write_text(_storyboard().model_dump_json(indent=2), encoding="utf-8")
    return path


def _task(path: Path, task_id: str = "narration-001") -> TaskContext:
    return TaskContext(
        task_id=task_id,
        assignee=AgentRole.NARRATION,
        objective="Synthesize the approved voice-over.",
        input_artifacts=[ArtifactRef(name="storyboard", path=path)],
        expected_output="Measured narration audio and a narration manifest.",
        acceptance_criteria=["Preserve the approved narration text"],
    )


def test_director_delegates_and_narration_loads_storyboard_artifact(
    tmp_path: Path,
) -> None:
    storyboard_path = _storyboard_artifact(tmp_path)
    durations = {
        "The station should have been busy.": 8.0,
        "Then every light went out.": 9.5,
    }
    tts = FakeTTS(durations)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(
        runtime,
        FootageAnalystAgent(UnusedIndexer()),
        artifacts_dir=tmp_path / "artifacts",
        narration_agent=NarrationAgent(tts),
    )

    result = asyncio.run(director.run_narration_task(_task(storyboard_path)))

    manifest_path = tmp_path / "artifacts" / "narration" / "narration.json"
    manifest = NarrationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    assert result.status == "completed"
    assert result.agent == AgentRole.NARRATION
    assert result.output_artifacts == [
        ArtifactRef(name="narration-manifest", path=manifest_path)
    ]
    assert manifest.target_duration_s == 12
    assert manifest.duration_s == 17.5
    assert [segment.target_duration_s for segment in manifest.segments] == [5, 7]
    assert [segment.duration_s for segment in manifest.segments] == [8, 9.5]
    assert all(segment.audio_path.is_file() for segment in manifest.segments)
    assert "17.5s actual for 12.0s target" in result.summary
    assert director.read_result("narration-001") == result


def test_longer_audio_is_measured_without_blocking(tmp_path: Path) -> None:
    storyboard_path = _storyboard_artifact(tmp_path)
    tts = FakeTTS(
        {
            "The station should have been busy.": 20,
            "Then every light went out.": 15,
        }
    )
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(storyboard_path, "narration-longer-than-target")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        NarrationAgent(tts).run_task(task, runtime, tmp_path / "artifacts")
    )

    manifest = NarrationManifest.model_validate_json(
        (tmp_path / "artifacts" / "narration" / "narration.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.status == "completed"
    assert manifest.duration_s == 35
    assert manifest.target_duration_s == 12


def test_narration_persists_blocked_result_and_cleans_up_after_tts_failure(
    tmp_path: Path,
) -> None:
    storyboard_path = _storyboard_artifact(tmp_path)
    tts = FakeTTS(failing_text="Then every light went out.")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(storyboard_path, "narration-tts-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        NarrationAgent(tts).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "blocked"
    assert result.output_artifacts == []
    assert not (tmp_path / "artifacts" / "narration").exists()
    assert list((tmp_path / "artifacts").glob(".narration.*")) == []
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_narration_blocks_when_tts_does_not_write_audio(tmp_path: Path) -> None:
    storyboard_path = _storyboard_artifact(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(storyboard_path, "narration-empty-audio")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        NarrationAgent(FakeTTS(write_audio=False)).run_task(
            task, runtime, tmp_path / "artifacts"
        )
    )

    assert result.status == "blocked"
    assert "did not write usable audio" in result.summary
    assert not (tmp_path / "artifacts" / "narration").exists()


def test_narration_blocks_when_storyboard_cannot_load(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(
        tmp_path / "missing" / "storyboard.json",
        "narration-load-fail",
    )
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        NarrationAgent(FakeTTS()).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "blocked"
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_narration_does_not_overwrite_existing_artifacts(tmp_path: Path) -> None:
    storyboard_path = _storyboard_artifact(tmp_path)
    destination = tmp_path / "artifacts" / "narration"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(storyboard_path, "narration-persistence-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        NarrationAgent(FakeTTS()).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "blocked"
    assert marker.read_text(encoding="utf-8") == "keep"


def test_narration_rejects_another_agents_task(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = TaskContext(
        task_id="screenwriter-boundary",
        assignee=AgentRole.SCREENWRITER,
        objective="Write a storyboard.",
        expected_output="A storyboard.",
        acceptance_criteria=["Return a storyboard"],
    )

    with pytest.raises(ValueError):
        asyncio.run(
            NarrationAgent(FakeTTS()).run_task(task, runtime, tmp_path / "artifacts")
        )


def test_director_cannot_submit_a_narration_result(tmp_path: Path) -> None:
    storyboard_path = _storyboard_artifact(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(storyboard_path, "narration-submit-boundary")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = TaskResult(
        task_id=task.task_id,
        agent=AgentRole.NARRATION,
        status="completed",
        summary="Synthesized narration.",
    )

    with pytest.raises(PermissionError):
        runtime.submit_result(AgentRole.DIRECTOR, result)
