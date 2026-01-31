from __future__ import annotations

import os
import tempfile
from pathlib import Path

from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import Storyboard, VideoIndex
from directorx.core.ports import ScreenwriterModel


class ScreenwriterAgent:
    role = AgentRole.SCREENWRITER

    def __init__(
        self, model: ScreenwriterModel, artifacts_dir: Path | None = None
    ) -> None:
        self.model = model
        self.artifacts_dir = artifacts_dir

    async def run(
        self, prompt: str, index: VideoIndex, target_duration_s: float
    ) -> Storyboard:
        storyboard = await self.model.draft(prompt, index, target_duration_s)
        if not storyboard.beats:
            raise ValueError("Screenwriter agent returned no beats")
        beat_ids = [beat.id for beat in storyboard.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be unique")
        return storyboard

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
        prompt: str | None = None,
        target_duration_s: float | None = None,
    ) -> TaskResult:
        """Execute a Director-delegated task using only declared artifacts."""
        if task.assignee != self.role:
            raise ValueError("Screenwriter can only execute its own tasks")

        try:
            index = self._load_video_index(task)
            storyboard = await self.run(
                prompt if prompt is not None else task.objective,
                index,
                target_duration_s
                if target_duration_s is not None
                else self._target_duration(task),
            )
            storyboard = Storyboard.model_validate(storyboard)
            destination = artifacts_dir or self.artifacts_dir
            if destination is None:
                raise ValueError(
                    "Screenwriter requires a configured artifacts directory"
                )
            storyboard_path = self._persist_storyboard(storyboard, destination)
        except Exception as exc:
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Screenwriting blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result

        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Drafted storyboard {storyboard.title!r} with "
                f"{len(storyboard.beats)} beats."
            ),
            output_artifacts=[ArtifactRef(name="storyboard", path=storyboard_path)],
        )
        runtime.submit_result(self.role, result)
        return result

    @staticmethod
    def _load_video_index(task: TaskContext) -> VideoIndex:
        references = [
            artifact
            for artifact in task.input_artifacts
            if artifact.name == "video-index"
        ]
        if len(references) != 1 or references[0].path.name != "index.json":
            raise ValueError(
                "Screenwriter task must declare exactly one video-index index.json "
                "input artifact"
            )
        path = references[0].path
        return VideoIndex.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _target_duration(task: TaskContext) -> float:
        raise ValueError("Screenwriter task requires target_duration_s")

    @staticmethod
    def _persist_storyboard(storyboard: Storyboard, artifacts_dir: Path) -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / "storyboard.json"
        temporary: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=artifacts_dir, prefix=".storyboard.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary = Path(raw_path)
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(storyboard.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
            temporary = None
            return path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
