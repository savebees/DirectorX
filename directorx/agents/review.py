from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import ReviewReport
from directorx.core.ports import ReviewFrameExtractor, ReviewModel
from directorx.services.review import _probe_duration


class ReviewAgent:
    """Review a finished video once for obvious continuity and quality defects."""

    role = AgentRole.REVIEW

    def __init__(
        self,
        model: ReviewModel,
        frame_extractor: ReviewFrameExtractor,
        *,
        artifacts_dir: Path | None = None,
        max_frames: int = 12,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("Review max_frames must be positive")
        self.model = model
        self.frame_extractor = frame_extractor
        self.artifacts_dir = artifacts_dir
        self.max_frames = max_frames

    async def run(self, video_path: Path, work_dir: Path) -> ReviewReport:
        duration_s = await asyncio.to_thread(_probe_duration, video_path)
        frames = await self.frame_extractor.extract(
            video_path, work_dir, max_frames=self.max_frames
        )
        report = ReviewReport.model_validate(
            await self.model.inspect(duration_s, frames)
        )
        return report

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != self.role:
            raise ValueError("Review can only execute its own tasks")
        staging_dir: Path | None = None
        try:
            video_path = self._load_video(task)
            destination = artifacts_dir or self.artifacts_dir
            if destination is None:
                raise ValueError("Review requires a configured artifacts directory")
            destination.mkdir(parents=True, exist_ok=True)
            report_path = destination / "review.json"
            if report_path.exists():
                raise FileExistsError(report_path)
            staging_dir = Path(tempfile.mkdtemp(dir=destination, prefix=".review."))
            report = await self.run(video_path, staging_dir)
            temporary = staging_dir / "review.json"
            temporary.write_text(
                report.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, report_path)
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir = None
        except Exception as exc:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Video review blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result

        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed" if report.passed else "blocked",
            summary=report.summary,
            output_artifacts=[ArtifactRef(name="review", path=report_path)],
        )
        runtime.submit_result(self.role, result)
        return result

    @staticmethod
    def _load_video(task: TaskContext) -> Path:
        references = [
            artifact
            for artifact in task.input_artifacts
            if artifact.name == "rendered-video"
        ]
        if len(references) != 1:
            raise ValueError(
                "Review task must declare exactly one rendered-video input artifact"
            )
        path = references[0].path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
