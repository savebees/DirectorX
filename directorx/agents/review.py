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
from directorx.core.models import (
    EditTimeline,
    NarrationManifest,
    ReviewReport,
    Storyboard,
)
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
        min_voice_coverage: float = 0.65,
        max_voice_coverage: float = 0.88,
        max_freeze_per_clip_s: float = 0.3,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("Review max_frames must be positive")
        self.model = model
        self.frame_extractor = frame_extractor
        self.artifacts_dir = artifacts_dir
        self.max_frames = max_frames
        self.min_voice_coverage = min_voice_coverage
        self.max_voice_coverage = max_voice_coverage
        self.max_freeze_per_clip_s = max_freeze_per_clip_s

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
            context = self._load_edit_context(task)
            if context is not None:
                self._validate_edit_context(*context)
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

    @staticmethod
    def _load_edit_context(
        task: TaskContext,
    ) -> tuple[Storyboard, NarrationManifest, EditTimeline] | None:
        names = {"storyboard", "narration-manifest", "edit-timeline"}
        references = {
            artifact.name: artifact.path
            for artifact in task.input_artifacts
            if artifact.name in names
        }
        if not references:
            return None
        if set(references) != names:
            raise ValueError(
                "Narrative review requires storyboard, narration, and edit timeline"
            )
        return (
            Storyboard.model_validate_json(
                references["storyboard"].read_text(encoding="utf-8")
            ),
            NarrationManifest.model_validate_json(
                references["narration-manifest"].read_text(encoding="utf-8")
            ),
            EditTimeline.model_validate_json(
                references["edit-timeline"].read_text(encoding="utf-8")
            ),
        )

    def _validate_edit_context(
        self,
        storyboard: Storyboard,
        narration: NarrationManifest,
        timeline: EditTimeline,
    ) -> None:
        if not storyboard.full_narration.strip():
            raise ValueError("Narrative review requires continuous full narration")
        joined = "".join(beat.narration for beat in storyboard.beats)
        if "".join(joined.split()) != "".join(storyboard.full_narration.split()):
            raise ValueError("Narrative passages do not reconstruct full narration")
        coverage = narration.duration_s / timeline.duration_s
        if not self.min_voice_coverage <= coverage <= self.max_voice_coverage:
            raise ValueError(
                f"Narrative voice coverage {coverage:.1%} is outside "
                f"{self.min_voice_coverage:.1%}-{self.max_voice_coverage:.1%}"
            )
        freezes = [
            max(0.0, clip.target_duration_s - clip.source_range.duration_s)
            for clip in timeline.clips
        ]
        if any(value > self.max_freeze_per_clip_s + 1e-6 for value in freezes):
            raise ValueError("Timeline contains an excessive frozen frame hold")
        if abs(sum(freezes) - timeline.freeze_duration_s) > 0.01:
            raise ValueError("Timeline freeze duration is inconsistent")
