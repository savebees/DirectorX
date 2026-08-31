from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskResult,
)
from directorx.workflow import DirectorWorkflow


class FakeDirector:
    def __init__(self, root: Path, blocked_stage: str | None = None) -> None:
        self.runtime = CoordinationRuntime(root)
        self.blocked_stage = blocked_stage
        self.calls: list[str] = []
        self.started: list[str] = []
        self.active = 0
        self.max_active = 0
        self.task_inputs: dict[str, list[str]] = {}

    def initialize_project(self, memory) -> Path:
        return self.runtime.initialize_project(AgentRole.DIRECTOR, memory)

    def update_project_memory(self, memory) -> Path:
        return self.runtime.update_project_memory(AgentRole.DIRECTOR, memory)

    async def _finish(self, stage: str, task, artifacts_dir: Path) -> TaskResult:
        self.calls.append(stage)
        self.task_inputs[stage] = [artifact.name for artifact in task.input_artifacts]
        self.started.append(stage)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.runtime.delegate(AgentRole.DIRECTOR, task)
        if stage in {"narration", "grounding"}:
            await asyncio.sleep(0.02)
        status = "blocked" if stage == self.blocked_stage else "completed"
        outputs: list[ArtifactRef] = []
        if status == "completed":
            names = {
                "footage": ("video-index", "index.json"),
                "screenwriter": ("storyboard", "storyboard.json"),
                "narration": ("narration-manifest", "narration.json"),
                "grounding": ("grounding-manifest", "grounding.json"),
                "sound": ("sound-plan", "sound-plan.json"),
                "editor": ("edit-timeline", "timeline.json"),
                "render": ("rendered-video", "final.mp4"),
                "review": ("review", "review.json"),
            }
            name, filename = names[stage]
            path = artifacts_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stage, encoding="utf-8")
            outputs = [ArtifactRef(name=name, path=path)]
            if stage == "footage":
                summary = artifacts_dir / "story-summary.json"
                database = artifacts_dir / "search.sqlite3"
                summary.write_text(stage, encoding="utf-8")
                database.write_text(stage, encoding="utf-8")
                outputs.extend(
                    [
                        ArtifactRef(name="story-summary", path=summary),
                        ArtifactRef(name="scene-search-database", path=database),
                    ]
                )
        result = TaskResult(
            task_id=task.task_id,
            agent=task.assignee,
            status=status,
            summary=f"{stage} {status}",
            output_artifacts=outputs,
        )
        self.runtime.submit_result(task.assignee, result)
        self.active -= 1
        return result

    async def run_footage_task(self, task, video_path, artifacts_dir):
        return await self._finish("footage", task, artifacts_dir)

    async def run_screenwriter_task(self, task, **kwargs):
        return await self._finish("screenwriter", task, kwargs["artifacts_dir"])

    async def run_narration_task(self, task, artifacts_dir):
        return await self._finish("narration", task, artifacts_dir)

    async def run_grounding_task(self, task, artifacts_dir):
        return await self._finish("grounding", task, artifacts_dir)

    async def run_sound_task(self, task, artifacts_dir):
        return await self._finish("sound", task, artifacts_dir)

    async def run_editor_task(self, task, artifacts_dir):
        return await self._finish("editor", task, artifacts_dir)

    async def run_render_task(self, task, artifacts_dir):
        return await self._finish("render", task, artifacts_dir)

    async def run_review_task(self, task, artifacts_dir):
        return await self._finish("review", task, artifacts_dir)


def test_workflow_runs_in_order_and_passes_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    director = FakeDirector(tmp_path / "coordination")
    workflow = DirectorWorkflow(
        director,
        artifacts_dir=tmp_path / "artifacts",
        checkpoint_path=tmp_path / "checkpoints.sqlite3",
    )

    state = asyncio.run(
        workflow.run(
            project_id="workflow-order",
            video_path=video,
            brief="Make a concise story",
            target_duration_s=4,
        )
    )

    assert set(director.calls) == {
        "footage",
        "screenwriter",
        "narration",
        "grounding",
        "sound",
        "editor",
        "render",
        "review",
    }
    assert director.max_active >= 2
    assert "narration-manifest" not in director.task_inputs["grounding"]
    assert "narration-manifest" in director.task_inputs["sound"]
    assert director.started.index("narration") < director.started.index("sound")
    assert director.started.index("grounding") < director.started.index("render")
    assert "edit-timeline" in director.task_inputs["render"]
    assert state["review_passed"] is True
    assert state["artifacts"]["rendered-video"].endswith("final.mp4")
    assert state["statuses"]["review"] == "completed"
    memory = director.runtime.store.read_project_memory()
    assert "rendered-video" in memory.approved_artifacts


def test_workflow_stops_after_blocked_stage(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    director = FakeDirector(tmp_path / "coordination", blocked_stage="grounding")
    workflow = DirectorWorkflow(director, artifacts_dir=tmp_path / "artifacts")

    state = asyncio.run(
        workflow.run(
            project_id="workflow-blocked",
            video_path=video,
            brief="Make a concise story",
        )
    )

    assert set(director.calls) == {
        "footage",
        "screenwriter",
        "narration",
        "grounding",
        "sound",
    }
    assert state["current_stage"] == "media_join"
    assert state["error"] == "grounding blocked"
    assert state["statuses"]["sound"] == "completed"
    assert "render" not in state["statuses"]


def test_workflow_rejects_mismatched_project_memory(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    director = FakeDirector(tmp_path / "coordination")
    workflow = DirectorWorkflow(director, artifacts_dir=tmp_path / "artifacts")
    asyncio.run(workflow.run(project_id="first", video_path=video, brief="first brief"))

    with pytest.raises(ValueError, match="belongs to project"):
        asyncio.run(
            workflow.run(project_id="second", video_path=video, brief="second brief")
        )
