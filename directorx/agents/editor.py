from __future__ import annotations

import math
import os
import tempfile
from collections import defaultdict
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
    GroundedClip,
    GroundingManifest,
    NarrationManifest,
    SoundPlan,
    Storyboard,
    TimelineBeat,
)


class EditorAgent:
    """Turn parallel specialist outputs into one duration-safe edit timeline."""

    role = AgentRole.EDITOR

    def __init__(
        self,
        *,
        artifacts_dir: Path | None = None,
        min_voice_coverage: float = 0.65,
        max_voice_coverage: float = 0.88,
        breathing_room_s: float = 0.35,
        max_freeze_per_clip_s: float = 0.3,
        max_title_duration_s: float = 8.0,
    ) -> None:
        if not 0 <= min_voice_coverage <= max_voice_coverage <= 1:
            raise ValueError("Editor voice coverage bounds are invalid")
        if breathing_room_s < 0 or max_freeze_per_clip_s < 0:
            raise ValueError("Editor timing margins must be non-negative")
        if max_title_duration_s <= 0:
            raise ValueError("Editor title duration must be positive")
        self.artifacts_dir = artifacts_dir
        self.min_voice_coverage = min_voice_coverage
        self.max_voice_coverage = max_voice_coverage
        self.breathing_room_s = breathing_room_s
        self.max_freeze_per_clip_s = max_freeze_per_clip_s
        self.max_title_duration_s = max_title_duration_s

    def run(
        self,
        storyboard: Storyboard,
        grounding: GroundingManifest,
        narration: NarrationManifest,
        sound: SoundPlan,
    ) -> EditTimeline:
        storyboard = Storyboard.model_validate(storyboard)
        grounding = GroundingManifest.model_validate(grounding)
        narration = NarrationManifest.model_validate(narration)
        SoundPlan.model_validate(sound)
        if not grounding.source_video.is_file():
            raise FileNotFoundError(grounding.source_video)

        beat_ids = [beat.id for beat in storyboard.beats]
        if not beat_ids or len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be non-empty and unique")
        segments = {segment.beat_id: segment for segment in narration.segments}
        if len(segments) != len(narration.segments) or set(segments) != set(beat_ids):
            raise ValueError("Narration must contain exactly one segment per beat")

        clips_by_beat: dict[str, list[GroundedClip]] = defaultdict(list)
        shot_ids: set[str] = set()
        starts: list[float] = []
        for clip in grounding.clips:
            if clip.shot_id in shot_ids:
                raise ValueError(f"Grounding repeats shot id {clip.shot_id}")
            if clip.beat_id not in segments:
                raise ValueError(f"Grounding references unknown beat {clip.beat_id}")
            shot_ids.add(clip.shot_id)
            starts.append(clip.source_range.start_s)
            clips_by_beat[clip.beat_id].append(clip)
        if starts != sorted(starts):
            raise ValueError("Grounded clips must preserve source chronology")
        if set(clips_by_beat) != set(beat_ids):
            raise ValueError("Grounding must provide at least one clip per beat")

        target_s = storyboard.target_duration_s
        voice_coverage = narration.duration_s / target_s
        if voice_coverage < self.min_voice_coverage:
            raise ValueError(
                f"Narration coverage {voice_coverage:.1%} is below "
                f"{self.min_voice_coverage:.1%}"
            )
        if voice_coverage > self.max_voice_coverage:
            raise ValueError(
                f"Narration coverage {voice_coverage:.1%} exceeds "
                f"{self.max_voice_coverage:.1%}"
            )

        capacities: list[float] = []
        lower_bounds: list[float] = []
        desired: list[float] = []
        for position, beat in enumerate(storyboard.beats):
            beat_clips = clips_by_beat[beat.id]
            capacity = sum(
                clip.source_range.duration_s + self.max_freeze_per_clip_s
                for clip in beat_clips
            )
            lower = segments[beat.id].duration_s + self.breathing_room_s
            if lower > capacity + 1e-6:
                raise ValueError(
                    f"Beat {beat.id} needs {lower:.2f}s for narration but its "
                    f"verified footage can provide only {capacity:.2f}s"
                )
            value = min(capacity, max(lower, beat.target_duration_s))
            if position == len(storyboard.beats) - 1:
                capacity = min(capacity, self.max_title_duration_s)
                if lower > capacity + 1e-6:
                    raise ValueError(
                        f"Final beat {beat.id} needs {lower:.2f}s but the title "
                        f"limit is {self.max_title_duration_s:.2f}s"
                    )
                value = max(lower, min(value, capacity))
            capacities.append(capacity)
            lower_bounds.append(lower)
            desired.append(value)

        durations = self._fit_total(
            desired,
            lower_bounds,
            capacities,
            target_s,
        )
        timeline_clips: list[GroundedClip] = []
        timeline_beats: list[TimelineBeat] = []
        cursor_s = 0.0
        freeze_duration_s = 0.0
        for beat, duration_s in zip(storyboard.beats, durations, strict=True):
            source_clips = clips_by_beat[beat.id]
            clip_capacities = [
                clip.source_range.duration_s + self.max_freeze_per_clip_s
                for clip in source_clips
            ]
            clip_durations = self._allocate_with_caps(
                duration_s,
                clip_capacities,
            )
            edited_clips = [
                clip.model_copy(update={"target_duration_s": clip_duration})
                for clip, clip_duration in zip(
                    source_clips, clip_durations, strict=True
                )
            ]
            freeze_duration_s += sum(
                max(0.0, clip.target_duration_s - clip.source_range.duration_s)
                for clip in edited_clips
            )
            timeline_clips.extend(edited_clips)
            timeline_beats.append(
                TimelineBeat(
                    beat_id=beat.id,
                    clip_ids=[clip.shot_id for clip in edited_clips],
                    start_s=cursor_s,
                    duration_s=duration_s,
                    narration_duration_s=segments[beat.id].duration_s,
                )
            )
            cursor_s += duration_s

        return EditTimeline(
            source_video=grounding.source_video,
            clips=timeline_clips,
            beats=timeline_beats,
            target_duration_s=target_s,
            duration_s=cursor_s,
            voice_coverage=narration.duration_s / cursor_s,
            freeze_duration_s=freeze_duration_s,
        )

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != self.role:
            raise ValueError("Editor can only execute its own tasks")
        try:
            storyboard, grounding, narration, sound = self._load_artifacts(task)
            timeline = self.run(storyboard, grounding, narration, sound)
            destination = artifacts_dir or self.artifacts_dir
            if destination is None:
                raise ValueError("Editor requires a configured artifacts directory")
            path = self._persist_timeline(timeline, destination)
        except Exception as exc:
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Editing blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result
        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Built a {timeline.duration_s:.1f}s timeline from "
                f"{len(timeline.clips)} clips with {timeline.voice_coverage:.1%} "
                f"voice coverage and {timeline.freeze_duration_s:.2f}s freeze."
            ),
            output_artifacts=[ArtifactRef(name="edit-timeline", path=path)],
        )
        runtime.submit_result(self.role, result)
        return result

    @staticmethod
    def _fit_total(
        desired: list[float],
        lower: list[float],
        upper: list[float],
        target: float,
    ) -> list[float]:
        if sum(lower) > target + 1e-6:
            raise ValueError("Narration and breathing room exceed target duration")
        if sum(upper) < target - 1e-6:
            raise ValueError("Verified footage cannot fill the target duration")
        values = list(desired)
        difference = target - sum(values)
        if difference > 0:
            room = [
                maximum - value for maximum, value in zip(upper, values, strict=True)
            ]
            EditorAgent._distribute(values, room, difference, direction=1)
        elif difference < 0:
            room = [
                value - minimum for value, minimum in zip(values, lower, strict=True)
            ]
            EditorAgent._distribute(values, room, -difference, direction=-1)
        if not math.isclose(sum(values), target, abs_tol=1e-6):
            raise ValueError("Editor could not reconcile the target duration")
        return values

    @staticmethod
    def _distribute(
        values: list[float],
        room: list[float],
        amount: float,
        *,
        direction: int,
    ) -> None:
        remaining = amount
        for index in sorted(range(len(room)), key=lambda item: -room[item]):
            change = min(room[index], remaining)
            values[index] += direction * change
            remaining -= change
            if remaining <= 1e-6:
                break
        if remaining > 1e-6:
            raise ValueError("Editor has insufficient timing slack")

    @staticmethod
    def _allocate_with_caps(total: float, capacities: list[float]) -> list[float]:
        capacity = sum(capacities)
        if total > capacity + 1e-6:
            raise ValueError("Beat duration exceeds its clip capacities")
        if len(capacities) == 1:
            return [total]
        allocations = [total * item / capacity for item in capacities]
        allocations[-1] += total - sum(allocations)
        return allocations

    @staticmethod
    def _load_artifacts(
        task: TaskContext,
    ) -> tuple[Storyboard, GroundingManifest, NarrationManifest, SoundPlan]:
        paths = {
            "storyboard": EditorAgent._artifact_path(
                task, "storyboard", "storyboard.json"
            ),
            "grounding": EditorAgent._artifact_path(
                task, "grounding-manifest", "grounding.json"
            ),
            "narration": EditorAgent._artifact_path(
                task, "narration-manifest", "narration.json"
            ),
            "sound": EditorAgent._artifact_path(task, "sound-plan", "sound-plan.json"),
        }
        return (
            Storyboard.model_validate_json(paths["storyboard"].read_text()),
            GroundingManifest.model_validate_json(paths["grounding"].read_text()),
            NarrationManifest.model_validate_json(paths["narration"].read_text()),
            SoundPlan.model_validate_json(paths["sound"].read_text()),
        )

    @staticmethod
    def _artifact_path(task: TaskContext, name: str, filename: str) -> Path:
        references = [
            artifact for artifact in task.input_artifacts if artifact.name == name
        ]
        if len(references) != 1 or references[0].path.name != filename:
            raise ValueError(
                f"Editor task must declare exactly one {name} {filename} artifact"
            )
        if not references[0].path.is_file():
            raise FileNotFoundError(references[0].path)
        return references[0].path

    @staticmethod
    def _persist_timeline(timeline: EditTimeline, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "timeline.json"
        temporary: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=destination, prefix=".timeline.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary = Path(raw_path)
            temporary.write_text(
                timeline.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            if path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
            temporary = None
            return path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
