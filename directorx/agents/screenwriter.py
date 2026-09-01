from __future__ import annotations

import math
import os
import re
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
    NarrationDraft,
    Screenplay,
    ScreenwriterSceneEvidence,
    StoryBeat,
    Storyboard,
    StorySummary,
    VideoIndex,
)
from directorx.core.ports import ScreenwriterModel
from directorx.indexing import validate_story_summary


class ScreenwriterAgent:
    role = AgentRole.SCREENWRITER
    _STAGE_DIRECTION = re.compile(r"[（(\[]([^（）()\[\]]{1,60})[）)\]]")
    _STAGE_DIRECTION_MARKERS = (
        "片名",
        "字幕",
        "画面",
        "镜头",
        "淡入",
        "淡出",
        "浮现",
        "出现",
        "音乐",
        "音效",
        "title",
        "shot",
        "visual",
        "music",
        "sfx",
    )

    def __init__(
        self, model: ScreenwriterModel, artifacts_dir: Path | None = None
    ) -> None:
        self.model = model
        self.artifacts_dir = artifacts_dir

    async def run(
        self,
        objective: str,
        constraints: list[str],
        index: VideoIndex,
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> Storyboard:
        screenplay = Screenplay.model_validate(
            await self.model.draft_screenplay(
                objective,
                constraints,
                story_summary,
                target_duration_s,
            )
        )
        self._validate_screenplay(screenplay, story_summary, target_duration_s)
        evidence_by_beat = self._expand_evidence(screenplay, story_summary, index)
        narration = NarrationDraft.model_validate(
            await self.model.draft_narration(
                objective,
                constraints,
                screenplay,
                evidence_by_beat,
            )
        )
        screenplay = self._rebalance_for_narration(screenplay, narration, story_summary)
        storyboard = self._merge_storyboard(screenplay, narration, evidence_by_beat)
        self._validate_storyboard(storyboard, target_duration_s)
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
            index, story_summary = self._load_source_artifacts(task)
            duration = (
                target_duration_s
                if target_duration_s is not None
                else self._target_duration(task)
            )
            if duration <= 0:
                raise ValueError("target_duration_s must be positive")
            constraints = [*task.constraints, *task.acceptance_criteria]
            storyboard = await self.run(
                prompt if prompt is not None else task.objective,
                constraints,
                index,
                story_summary,
                duration,
            )
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
    def _load_source_artifacts(
        task: TaskContext,
    ) -> tuple[VideoIndex, StorySummary]:
        index_path = ScreenwriterAgent._artifact_path(task, "video-index", "index.json")
        summary_path = ScreenwriterAgent._artifact_path(
            task, "story-summary", "story-summary.json"
        )
        index = VideoIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
        summary = StorySummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        return index, validate_story_summary(index, summary)

    @staticmethod
    def _artifact_path(task: TaskContext, name: str, filename: str) -> Path:
        references = [
            artifact for artifact in task.input_artifacts if artifact.name == name
        ]
        if len(references) != 1 or references[0].path.name != filename:
            raise ValueError(
                f"Screenwriter task must declare exactly one {name} {filename} "
                "input artifact"
            )
        return references[0].path

    @staticmethod
    def _target_duration(task: TaskContext) -> float:
        raise ValueError("Screenwriter task requires target_duration_s")

    @staticmethod
    def _validate_screenplay(
        screenplay: Screenplay,
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> None:
        ScreenwriterAgent._require_text(
            screenplay.title, screenplay.logline, screenplay.narrative_angle
        )
        if not screenplay.beats:
            raise ValueError("Screenwriter model returned no screenplay beats")
        beat_ids = [beat.id for beat in screenplay.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Screenplay beat ids must be unique")
        sequence_ids = {sequence.id for sequence in story_summary.sequences}
        for beat in screenplay.beats:
            ScreenwriterAgent._require_text(
                beat.id,
                beat.purpose,
                beat.story_content,
                beat.visual_intent,
                beat.mood,
            )
            if not beat.source_sequence_ids:
                raise ValueError(f"Screenplay beat {beat.id} has no source sequences")
            if len(beat.source_sequence_ids) != len(set(beat.source_sequence_ids)):
                raise ValueError(f"Screenplay beat {beat.id} repeats a source sequence")
            unknown = set(beat.source_sequence_ids) - sequence_ids
            if unknown:
                raise ValueError(
                    f"Screenplay beat {beat.id} references unknown sequences: "
                    + ", ".join(sorted(unknown))
                )
        ScreenwriterAgent._validate_duration(
            screenplay.target_duration_s,
            [beat.target_duration_s for beat in screenplay.beats],
            target_duration_s,
        )

    @staticmethod
    def _expand_evidence(
        screenplay: Screenplay,
        story_summary: StorySummary,
        index: VideoIndex,
    ) -> dict[str, list[ScreenwriterSceneEvidence]]:
        sequences = {sequence.id: sequence for sequence in story_summary.sequences}
        scenes = {scene.id: scene for scene in index.scenes}
        scene_order = {
            scene.id: position for position, scene in enumerate(index.scenes)
        }
        evidence_by_beat: dict[str, list[ScreenwriterSceneEvidence]] = {}
        for beat in screenplay.beats:
            selected_ids = {
                scene_id
                for sequence_id in beat.source_sequence_ids
                for scene_id in sequences[sequence_id].scene_ids
            }
            ordered_ids = sorted(selected_ids, key=scene_order.__getitem__)
            evidence: list[ScreenwriterSceneEvidence] = []
            for scene_id in ordered_ids:
                scene = scenes[scene_id]
                ScreenwriterAgent._require_text(scene.short_summary, scene.caption)
                evidence.append(
                    ScreenwriterSceneEvidence(
                        scene_id=scene_id,
                        short_summary=scene.short_summary,
                        caption=scene.caption,
                        tags=scene.tags,
                    )
                )
            evidence_by_beat[beat.id] = evidence
        return evidence_by_beat

    @staticmethod
    def _rebalance_for_narration(
        screenplay: Screenplay,
        narration: NarrationDraft,
        story_summary: StorySummary,
        breathing_room_s: float = 0.35,
    ) -> Screenplay:
        """Fit planning durations to the written story without rewriting its prose."""
        narration_by_id = {beat.beat_id: beat.narration for beat in narration.beats}
        if set(narration_by_id) != {beat.id for beat in screenplay.beats}:
            return screenplay
        weights = [
            ScreenwriterAgent._spoken_units(
                ScreenwriterAgent._spoken_narration(narration_by_id[beat.id])
            )
            for beat in screenplay.beats
        ]
        available_s = screenplay.target_duration_s - breathing_room_s * len(weights)
        if available_s <= 0:
            return screenplay
        weight_sum = sum(weights)
        if weight_sum == 0:
            return screenplay
        desired = [
            breathing_room_s + available_s * weight / weight_sum for weight in weights
        ]
        sequence_durations = {
            sequence.id: sequence.source_range.duration_s
            for sequence in story_summary.sequences
            if sequence.source_range is not None
        }
        capacities = [
            (
                sum(sequence_durations[item] for item in beat.source_sequence_ids)
                if all(item in sequence_durations for item in beat.source_sequence_ids)
                else float("inf")
            )
            for beat in screenplay.beats
        ]
        if screenplay.target_duration_s < 30:
            capacities = [float("inf") for _ in screenplay.beats]
        else:
            capacities[-1] = min(capacities[-1], 8.0)
        values = [
            min(value, capacity)
            for value, capacity in zip(desired, capacities, strict=True)
        ]
        remaining_s = screenplay.target_duration_s - sum(values)
        while remaining_s > 1e-6:
            active = [
                index
                for index, (value, capacity) in enumerate(
                    zip(values, capacities, strict=True)
                )
                if capacity - value > 1e-6
            ]
            if not active:
                raise ValueError(
                    "Selected source sequences cannot fit narration timing"
                )
            total_room = sum(capacities[index] - values[index] for index in active)
            applied_s = 0.0
            for index in active:
                room_s = capacities[index] - values[index]
                change_s = min(room_s, remaining_s * room_s / total_room)
                values[index] += change_s
                applied_s += change_s
            if applied_s <= 1e-9:
                raise ValueError("Could not rebalance narration timing")
            remaining_s -= applied_s
        values[-1] += screenplay.target_duration_s - sum(values)
        return screenplay.model_copy(
            update={
                "beats": [
                    beat.model_copy(update={"target_duration_s": duration_s})
                    for beat, duration_s in zip(screenplay.beats, values, strict=True)
                ]
            }
        )

    @staticmethod
    def _spoken_units(text: str) -> int:
        han = len(re.findall(r"[\u4e00-\u9fff]", text))
        if han:
            return han
        return max(1, len(re.findall(r"\b\w+\b", text)))

    @staticmethod
    def _merge_storyboard(
        screenplay: Screenplay,
        narration: NarrationDraft,
        evidence_by_beat: dict[str, list[ScreenwriterSceneEvidence]],
    ) -> Storyboard:
        narration_ids = [beat.beat_id for beat in narration.beats]
        if len(narration_ids) != len(set(narration_ids)):
            raise ValueError("Narration beat ids must be unique")
        screenplay_ids = [beat.id for beat in screenplay.beats]
        if set(narration_ids) != set(screenplay_ids):
            raise ValueError(
                "Narration must contain exactly one entry per screenplay beat"
            )

        narration_by_id = {beat.beat_id: beat for beat in narration.beats}
        beats: list[StoryBeat] = []
        for screenplay_beat in screenplay.beats:
            narration_beat = narration_by_id[screenplay_beat.id]
            narration_text = ScreenwriterAgent._spoken_narration(
                narration_beat.narration
            )
            evidence_ids = narration_beat.evidence_scene_ids
            if not evidence_ids:
                raise ValueError(
                    f"Narration beat {screenplay_beat.id} has no evidence scenes"
                )
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError(
                    f"Narration beat {screenplay_beat.id} repeats an evidence scene"
                )
            allowed_ids = {
                scene.scene_id for scene in evidence_by_beat[screenplay_beat.id]
            }
            unknown = set(evidence_ids) - allowed_ids
            if unknown:
                raise ValueError(
                    f"Narration beat {screenplay_beat.id} references scenes outside "
                    "its source sequences: " + ", ".join(sorted(unknown))
                )
            beats.append(
                StoryBeat(
                    id=screenplay_beat.id,
                    purpose=screenplay_beat.purpose,
                    story_content=screenplay_beat.story_content,
                    narration=narration_text,
                    visual_intent=screenplay_beat.visual_intent,
                    mood=screenplay_beat.mood,
                    target_duration_s=screenplay_beat.target_duration_s,
                    source_sequence_ids=screenplay_beat.source_sequence_ids,
                    evidence_scene_ids=evidence_ids,
                )
            )
        return Storyboard(
            title=screenplay.title,
            logline=screenplay.logline,
            narrative_angle=screenplay.narrative_angle,
            full_narration="".join(beat.narration for beat in beats),
            beats=beats,
            target_duration_s=screenplay.target_duration_s,
        )

    @staticmethod
    def _validate_storyboard(storyboard: Storyboard, target_duration_s: float) -> None:
        Storyboard.model_validate(storyboard)
        ScreenwriterAgent._require_text(
            storyboard.title, storyboard.logline, storyboard.narrative_angle
        )
        if not storyboard.beats:
            raise ValueError("Screenwriter agent returned no storyboard beats")
        if storyboard.full_narration:
            joined = "".join(beat.narration for beat in storyboard.beats)
            if "".join(joined.split()) != "".join(storyboard.full_narration.split()):
                raise ValueError(
                    "Storyboard full narration must match its ordered beat passages"
                )
        ScreenwriterAgent._validate_duration(
            storyboard.target_duration_s,
            [beat.target_duration_s for beat in storyboard.beats],
            target_duration_s,
        )

    @staticmethod
    def _validate_duration(
        declared_duration_s: float,
        beat_durations_s: list[float],
        target_duration_s: float,
    ) -> None:
        if not math.isclose(declared_duration_s, target_duration_s, abs_tol=0.1):
            raise ValueError("Screenplay target duration does not match the task")
        if not math.isclose(sum(beat_durations_s), declared_duration_s, abs_tol=0.1):
            raise ValueError(
                "Screenplay beat durations do not match its target duration"
            )

    @staticmethod
    def _require_text(*values: str) -> None:
        if any(not value.strip() for value in values):
            raise ValueError(
                "Screenwriter output contains an empty required text field"
            )

    @classmethod
    def _spoken_narration(cls, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            content = match.group(1).strip().casefold()
            if any(marker in content for marker in cls._STAGE_DIRECTION_MARKERS):
                return ""
            return match.group(0)

        cleaned = cls._STAGE_DIRECTION.sub(replace, text)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s+([，。！？；：,.!?;:])", r"\1", cleaned)
        return cleaned.strip()

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
