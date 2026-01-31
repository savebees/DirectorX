import asyncio
from pathlib import Path

import pytest

from directorx.agents import DirectorAgent, FootageAnalystAgent, ScreenwriterAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import (
    Scene,
    StoryBeat,
    Storyboard,
    TimeRange,
    VideoIndex,
)


class UnusedIndexer:
    async def build(self, video_path: Path) -> VideoIndex:
        raise AssertionError("The footage agent is not used in this test")


class FixedScreenwriterModel:
    def __init__(self, storyboard: Storyboard) -> None:
        self.storyboard = storyboard
        self.received_index: VideoIndex | None = None

    async def draft(
        self, prompt: str, video_index: VideoIndex, target_duration_s: float
    ) -> Storyboard:
        self.received_index = video_index
        assert prompt == "Write a concise suspenseful edit."
        assert target_duration_s == 12
        return self.storyboard


class FailingScreenwriterModel:
    async def draft(
        self, prompt: str, video_index: VideoIndex, target_duration_s: float
    ) -> Storyboard:
        raise RuntimeError("planner unavailable")


def _index_artifact(tmp_path: Path) -> tuple[Path, VideoIndex]:
    index_path = tmp_path / "footage" / "index.json"
    index_path.parent.mkdir()
    index = VideoIndex(
        video_path=tmp_path / "movie.mp4",
        duration_s=12,
        scenes=[
            Scene(
                id="scene-00000",
                source_range=TimeRange(start_s=0, end_s=12),
                caption="A person waits in a dark room.",
            )
        ],
    )
    index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return index_path, index


def _task(index_path: Path, task_id: str = "screenwriter-001") -> TaskContext:
    return TaskContext(
        task_id=task_id,
        assignee=AgentRole.SCREENWRITER,
        objective="Write a concise suspenseful edit.",
        input_artifacts=[ArtifactRef(name="video-index", path=index_path)],
        expected_output="A validated storyboard artifact.",
        acceptance_criteria=["Ground the story in the indexed scenes"],
    )


def _storyboard() -> Storyboard:
    return Storyboard(
        title="The Wait",
        logline="A person waits as tension builds.",
        beats=[
            StoryBeat(
                id="beat-1",
                purpose="Establish tension",
                narration="The room is still, but the silence feels deliberate.",
                visual_intent="A person waiting in a dark room",
                mood="tense",
                target_duration_s=12,
            )
        ],
        target_duration_s=12,
    )


def test_director_delegates_screenwriter_and_agent_loads_index_artifact(
    tmp_path: Path,
) -> None:
    index_path, index = _index_artifact(tmp_path)
    model = FixedScreenwriterModel(_storyboard())
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(
        runtime,
        FootageAnalystAgent(UnusedIndexer()),
        ScreenwriterAgent(model),
        artifacts_dir=tmp_path / "artifacts",
    )

    result = asyncio.run(
        director.run_screenwriter_task(_task(index_path), target_duration_s=12)
    )

    assert result.status == "completed"
    assert model.received_index == index
    assert result.output_artifacts == [
        ArtifactRef(name="storyboard", path=tmp_path / "artifacts" / "storyboard.json")
    ]
    assert director.read_result("screenwriter-001") == result


def test_screenwriter_persists_validated_storyboard(tmp_path: Path) -> None:
    index_path, _ = _index_artifact(tmp_path)
    storyboard = _storyboard()
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, "screenwriter-persist")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FixedScreenwriterModel(storyboard)).run_task(
            task,
            runtime,
            tmp_path / "configured-artifacts",
            target_duration_s=12,
        )
    )

    path = tmp_path / "configured-artifacts" / "storyboard.json"
    assert result.status == "completed"
    persisted = Storyboard.model_validate_json(path.read_text(encoding="utf-8"))
    assert persisted == storyboard


def test_screenwriter_persists_blocked_result_when_model_fails(tmp_path: Path) -> None:
    index_path, _ = _index_artifact(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, "screenwriter-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FailingScreenwriterModel()).run_task(
            task, runtime, tmp_path / "artifacts", target_duration_s=12
        )
    )

    assert result.status == "blocked"
    assert result.output_artifacts == []
    director_result = runtime.read_result(AgentRole.DIRECTOR, task.task_id)
    assert director_result == result


def test_screenwriter_persists_blocked_result_when_index_cannot_load(
    tmp_path: Path,
) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(tmp_path / "missing-index.json", "screenwriter-load-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FixedScreenwriterModel(_storyboard())).run_task(
            task, runtime, tmp_path / "artifacts", target_duration_s=12
        )
    )

    assert result.status == "blocked"
    assert "Screenwriting blocked:" in result.summary
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_director_cannot_run_or_submit_for_another_role(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(runtime, FootageAnalystAgent(UnusedIndexer()))
    task = TaskContext(
        task_id="footage-boundary",
        assignee=AgentRole.FOOTAGE_ANALYST,
        objective="Index footage.",
        expected_output="An index.",
        acceptance_criteria=["Return an index"],
    )
    with pytest.raises(ValueError):
        asyncio.run(director.run_screenwriter_task(task, target_duration_s=12))


def test_director_cannot_submit_a_screenwriter_result(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(tmp_path / "index.json", "screenwriter-submit-boundary")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = TaskResult(
        task_id=task.task_id,
        agent=AgentRole.SCREENWRITER,
        status="completed",
        summary="Drafted a storyboard.",
    )

    with pytest.raises(PermissionError):
        runtime.submit_result(AgentRole.DIRECTOR, result)
