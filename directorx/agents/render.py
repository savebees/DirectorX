from __future__ import annotations

import math
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
    GroundingManifest,
    NarrationManifest,
    RenderPlan,
    SoundPlan,
    TimelineBeat,
)
from directorx.core.ports import RenderEngine


class RenderAgent:
    """Compile validated media artifacts into one playable video."""

    role = AgentRole.RENDER

    def __init__(
        self,
        renderer: RenderEngine,
        *,
        artifacts_dir: Path | None = None,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Render dimensions must be positive")
        self.renderer = renderer
        self.artifacts_dir = artifacts_dir
        self.width = width
        self.height = height

    async def run(self, plan: RenderPlan) -> Path:
        plan = RenderPlan.model_validate(plan)
        if not plan.clips:
            raise ValueError("Render requires at least one grounded clip")
        if not plan.narration:
            raise ValueError("Render requires at least one narration segment")
        return await self.renderer.render(plan)

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != self.role:
            raise ValueError("Render can only execute its own tasks")

        staging_dir: Path | None = None
        try:
            timeline, narration, sound = self._load_artifacts(task)
            destination = artifacts_dir or self.artifacts_dir
            if destination is None:
                raise ValueError("Render requires a configured artifacts directory")
            destination.mkdir(parents=True, exist_ok=True)
            output_path = destination / "final.mp4"
            subtitle_path = destination / "subtitles.srt"
            if output_path.exists():
                raise FileExistsError(output_path)
            if subtitle_path.exists():
                raise FileExistsError(subtitle_path)
            staging_dir = Path(tempfile.mkdtemp(dir=destination, prefix=".render."))
            staging_output = staging_dir / "final.mp4"
            staging_subtitles = staging_dir / "subtitles.srt"
            plan = self._build_plan(
                timeline,
                narration,
                sound,
                staging_output,
            )
            self._write_subtitles(plan, staging_subtitles)
            plan = plan.model_copy(update={"subtitle_path": staging_subtitles})
            rendered_path = await self.run(plan)
            if rendered_path != staging_output:
                raise ValueError("Render engine returned an unexpected output path")
            if not rendered_path.is_file() or rendered_path.stat().st_size == 0:
                raise ValueError("Render engine did not produce a usable video")
            os.replace(rendered_path, output_path)
            os.replace(staging_subtitles, subtitle_path)
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir = None
        except Exception as exc:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Rendering blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result

        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Rendered {len(plan.clips)} clips into {output_path.name} "
                f"({plan.duration_s:.1f}s)."
            ),
            output_artifacts=[
                ArtifactRef(name="rendered-video", path=output_path),
                ArtifactRef(name="subtitles", path=subtitle_path),
            ],
        )
        runtime.submit_result(self.role, result)
        return result

    @classmethod
    def _build_plan(
        cls,
        timeline: EditTimeline | GroundingManifest,
        narration: NarrationManifest,
        sound: SoundPlan,
        output_path: Path,
    ) -> RenderPlan:
        if not timeline.source_video.is_file():
            raise FileNotFoundError(timeline.source_video)
        if not sound.track.path.is_file():
            raise FileNotFoundError(sound.track.path)
        segments_by_beat = cls._unique_by_beat(narration.segments, "narration")
        segments = list(narration.segments)
        for segment in segments:
            if (
                not segment.audio_path.is_file()
                or segment.audio_path.stat().st_size == 0
            ):
                raise FileNotFoundError(segment.audio_path)
        narration_duration = sum(segment.duration_s for segment in segments)
        if not math.isclose(narration_duration, narration.duration_s, abs_tol=0.1):
            raise ValueError("Narration duration does not match its segments")
        if isinstance(timeline, GroundingManifest):
            clips_by_beat = cls._unique_by_beat(timeline.clips, "grounded clips")
            if set(clips_by_beat) != set(segments_by_beat):
                raise ValueError(
                    "Grounded clips and narration must cover the same beats"
                )
            clips = []
            beats = []
            cursor_s = 0.0
            for segment in segments:
                clip = clips_by_beat[segment.beat_id]
                duration_s = max(clip.target_duration_s, segment.duration_s)
                clip = clip.model_copy(update={"target_duration_s": duration_s})
                clips.append(clip)
                beats.append(
                    TimelineBeat(
                        beat_id=segment.beat_id,
                        clip_ids=[clip.shot_id],
                        start_s=cursor_s,
                        duration_s=duration_s,
                        narration_duration_s=segment.duration_s,
                    )
                )
                cursor_s += duration_s
            duration = cursor_s
        else:
            clips = list(timeline.clips)
            beats = list(timeline.beats)
            timeline_beat_ids = [beat.beat_id for beat in beats]
            if not set(segments_by_beat).issubset(timeline_beat_ids):
                raise ValueError("Narration references an unknown timeline beat")
            if [
                beat_id for beat_id in timeline_beat_ids if beat_id in segments_by_beat
            ] != [segment.beat_id for segment in segments]:
                raise ValueError(
                    "Narration must preserve the order of spoken timeline beats"
                )
            if [clip.shot_id for clip in clips] != [
                clip_id for beat in beats for clip_id in beat.clip_ids
            ]:
                raise ValueError("Timeline beat clip ids do not match timeline clips")
            clips_by_id = {clip.shot_id: clip for clip in clips}
            if len(clips_by_id) != len(clips):
                raise ValueError("Timeline contains duplicate clip ids")
            for beat in beats:
                beat_clips = [clips_by_id[clip_id] for clip_id in beat.clip_ids]
                if any(clip.beat_id != beat.beat_id for clip in beat_clips):
                    raise ValueError("Timeline assigns a clip to the wrong beat")
                if not math.isclose(
                    sum(clip.target_duration_s for clip in beat_clips),
                    beat.duration_s,
                    abs_tol=0.01,
                ):
                    raise ValueError("Timeline beat duration does not match its clips")
            duration = timeline.duration_s
            if not math.isclose(
                duration,
                sum(beat.duration_s for beat in beats),
                abs_tol=0.01,
            ):
                raise ValueError("Timeline duration does not match its beats")
        return RenderPlan(
            source_video=timeline.source_video,
            clips=clips,
            beats=beats,
            narration=segments,
            sound=sound,
            output_path=output_path,
            target_duration_s=duration,
        )

    @classmethod
    def _write_subtitles(cls, plan: RenderPlan, path: Path) -> None:
        cursor_s = 0.0
        entries: list[str] = []
        narration_by_beat = cls._unique_by_beat(plan.narration, "narration")
        for beat in plan.beats:
            narration = narration_by_beat.get(beat.beat_id)
            if narration is None:
                cursor_s += beat.duration_s
                continue
            start_s = cursor_s
            end_s = start_s + min(narration.duration_s, beat.duration_s)
            text = narration.text.strip().replace("\r", " ").replace("\n", " ")
            if not text:
                raise ValueError(f"Narration text is empty for {narration.beat_id}")
            entries.append(
                f"{len(entries) + 1}\n{cls._srt_timestamp(start_s)} --> "
                f"{cls._srt_timestamp(end_s)}\n{text}\n"
            )
            cursor_s += beat.duration_s
        path.write_text("\n".join(entries), encoding="utf-8")

    @staticmethod
    def _srt_timestamp(seconds: float) -> str:
        milliseconds = round(seconds * 1000)
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"

    @staticmethod
    def _unique_by_beat(items: list, label: str) -> dict[str, object]:
        result: dict[str, object] = {}
        for item in items:
            beat_id = item.beat_id
            if beat_id in result:
                raise ValueError(f"{label} contain duplicate beat {beat_id}")
            result[beat_id] = item
        if not result:
            raise ValueError(f"{label} cannot be empty")
        return result

    @staticmethod
    def _load_artifacts(
        task: TaskContext,
    ) -> tuple[EditTimeline | GroundingManifest, NarrationManifest, SoundPlan]:
        timeline_refs = [
            artifact
            for artifact in task.input_artifacts
            if artifact.name == "edit-timeline"
        ]
        if timeline_refs:
            if len(timeline_refs) != 1 or timeline_refs[0].path.name != "timeline.json":
                raise ValueError(
                    "Render task must declare exactly one edit-timeline timeline.json"
                )
            timeline: EditTimeline | GroundingManifest = (
                EditTimeline.model_validate_json(
                    timeline_refs[0].path.read_text(encoding="utf-8")
                )
            )
        else:
            grounding_path = RenderAgent._artifact_path(
                task, "grounding-manifest", "grounding.json"
            )
            timeline = GroundingManifest.model_validate_json(
                grounding_path.read_text(encoding="utf-8")
            )
        narration_path = RenderAgent._artifact_path(
            task, "narration-manifest", "narration.json"
        )
        sound_path = RenderAgent._artifact_path(task, "sound-plan", "sound-plan.json")
        return (
            timeline,
            NarrationManifest.model_validate_json(
                narration_path.read_text(encoding="utf-8")
            ),
            SoundPlan.model_validate_json(sound_path.read_text(encoding="utf-8")),
        )

    @staticmethod
    def _artifact_path(task: TaskContext, name: str, filename: str) -> Path:
        references = [
            artifact for artifact in task.input_artifacts if artifact.name == name
        ]
        if len(references) != 1 or references[0].path.name != filename:
            raise ValueError(
                f"Render task must declare exactly one {name} {filename} input artifact"
            )
        path = references[0].path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
