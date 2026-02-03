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
    GroundedClip,
    GroundingCandidate,
    GroundingDecision,
    GroundingFrame,
    GroundingManifest,
    NarrationManifest,
    ShotRequest,
    Storyboard,
    StorySummary,
    TimeRange,
    VideoIndex,
)
from directorx.core.ports import (
    EmbeddingProvider,
    GroundingFrameExtractor,
    GroundingModel,
)
from directorx.indexing.hierarchy import validate_story_summary
from directorx.indexing.store import SceneSearchStore


class SceneRetriever:
    """Combine Screenwriter evidence with persisted hybrid scene search."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    async def search(
        self,
        shot: ShotRequest,
        index: VideoIndex,
        story_summary: StorySummary,
        *,
        limit: int,
        padding_s: float,
    ) -> list[GroundingCandidate]:
        if index.search_db_path is None or not index.search_db_path.is_file():
            raise RuntimeError("Grounding requires a persisted scene search index")
        if limit <= 0:
            raise ValueError("Grounding candidate limit must be positive")
        if padding_s < 0:
            raise ValueError("Grounding candidate padding must be non-negative")

        scenes = {scene.id: scene for scene in index.scenes}
        sequences = {sequence.id: sequence for sequence in story_summary.sequences}
        preferred_scene_ids = list(shot.evidence_scene_ids)
        for sequence_id in shot.source_sequence_ids:
            preferred_scene_ids.extend(sequences[sequence_id].scene_ids)

        scores: dict[str, float] = {}
        for scene_id in preferred_scene_ids:
            if scene_id not in scenes:
                raise ValueError(f"Grounding references unknown scene {scene_id}")
            score = 1.0 if scene_id in shot.evidence_scene_ids else 0.8
            scores[scene_id] = max(scores.get(scene_id, 0.0), score)

        query = " ".join(
            value
            for value in (
                shot.visual_query,
                shot.story_content,
                shot.mood,
                shot.narration_text,
            )
            if value.strip()
        )
        hits = await SceneSearchStore(
            index.search_db_path, self.embedding_provider
        ).search(query, limit=limit * 2)
        for hit in hits:
            scores[hit.scene_id] = max(
                scores.get(hit.scene_id, 0.0),
                0.5 + hit.score * 0.4,
            )

        ranked = sorted(
            scores,
            key=lambda scene_id: (
                -scores[scene_id],
                scenes[scene_id].source_range.start_s,
            ),
        )[:limit]
        candidates = [
            self._candidate(
                scene_id,
                scores[scene_id],
                index,
                padding_s,
                number,
            )
            for number, scene_id in enumerate(ranked, start=1)
        ]
        if not candidates:
            raise ValueError(f"No coarse grounding candidates found for {shot.id}")
        return candidates

    @staticmethod
    def _candidate(
        scene_id: str,
        score: float,
        index: VideoIndex,
        padding_s: float,
        number: int,
    ) -> GroundingCandidate:
        scenes = {scene.id: scene for scene in index.scenes}
        scene = scenes[scene_id]
        source_range = TimeRange(
            start_s=max(0.0, scene.source_range.start_s - padding_s),
            end_s=min(index.duration_s, scene.source_range.end_s + padding_s),
        )
        overlapping_scene_ids = [
            candidate.id
            for candidate in index.scenes
            if candidate.source_range.end_s > source_range.start_s
            and candidate.source_range.start_s < source_range.end_s
        ]
        return GroundingCandidate(
            id=f"candidate-{number:04d}",
            anchor_scene_id=scene_id,
            scene_ids=overlapping_scene_ids,
            source_range=source_range,
            retrieval_score=score,
        )


class GroundingAgent:
    """Ground approved visual intent in exact, visually verified source ranges."""

    role = AgentRole.GROUNDING

    def __init__(
        self,
        model: GroundingModel,
        retriever: SceneRetriever,
        frame_extractor: GroundingFrameExtractor,
        *,
        artifacts_dir: Path | None = None,
        candidate_limit: int = 4,
        candidate_padding_s: float = 1.0,
        coarse_fps: float = 1.0,
        refine_fps: float = 6.0,
        refine_margin_s: float = 2.0,
        max_coarse_frames: int = 24,
        max_refine_frames: int = 24,
        max_parallel: int = 2,
    ) -> None:
        positive_values = {
            "candidate_limit": candidate_limit,
            "coarse_fps": coarse_fps,
            "refine_fps": refine_fps,
            "max_coarse_frames": max_coarse_frames,
            "max_refine_frames": max_refine_frames,
            "max_parallel": max_parallel,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if candidate_padding_s < 0 or refine_margin_s < 0:
            raise ValueError("Grounding margins must be non-negative")
        self.model = model
        self.retriever = retriever
        self.frame_extractor = frame_extractor
        self.artifacts_dir = artifacts_dir
        self.candidate_limit = candidate_limit
        self.candidate_padding_s = candidate_padding_s
        self.coarse_fps = coarse_fps
        self.refine_fps = refine_fps
        self.refine_margin_s = refine_margin_s
        self.max_coarse_frames = max_coarse_frames
        self.max_refine_frames = max_refine_frames
        self.max_parallel = max_parallel

    async def run(
        self,
        shot: ShotRequest,
        index: VideoIndex,
        story_summary: StorySummary,
        work_dir: Path,
    ) -> GroundedClip:
        candidates = await self.retriever.search(
            shot,
            index,
            story_summary,
            limit=self.candidate_limit,
            padding_s=self.candidate_padding_s,
        )
        localized: list[tuple[GroundingCandidate, GroundingDecision]] = []
        for candidate in candidates:
            frames = await self.frame_extractor.extract(
                index.video_path,
                candidate.source_range,
                work_dir / candidate.id / "coarse",
                fps=self.coarse_fps,
                max_frames=self.max_coarse_frames,
                prefix=f"{candidate.id}-coarse",
            )
            decision = GroundingDecision.model_validate(
                await self.model.locate(shot, candidate, frames)
            )
            self._validate_decision(decision, candidate, frames)
            if decision.matched:
                localized.append((candidate, decision))

        localized.sort(
            key=lambda item: (
                -item[1].confidence,
                -item[0].retrieval_score,
                item[0].source_range.start_s,
            )
        )
        for candidate, coarse_decision in localized:
            refinement_candidate = candidate.model_copy(
                update={
                    "source_range": self._refinement_range(
                        candidate.source_range,
                        coarse_decision.source_range,
                    ),
                    "proposal_range": coarse_decision.source_range,
                }
            )
            frames = await self.frame_extractor.extract(
                index.video_path,
                refinement_candidate.source_range,
                work_dir / candidate.id / "refine",
                fps=self.refine_fps,
                max_frames=self.max_refine_frames,
                prefix=f"{candidate.id}-refine",
            )
            decision = GroundingDecision.model_validate(
                await self.model.refine(shot, refinement_candidate, frames)
            )
            self._validate_decision(decision, refinement_candidate, frames)
            if decision.matched:
                source_range = decision.source_range
                if source_range is None:
                    raise ValueError("Matched refinement has no source range")
                source_scene_ids = [
                    scene.id
                    for scene in index.scenes
                    if scene.source_range.end_s > source_range.start_s
                    and scene.source_range.start_s < source_range.end_s
                ]
                if not source_scene_ids:
                    raise ValueError("Grounded range does not overlap a source scene")
                frame_timestamps = {frame.id: frame.timestamp_s for frame in frames}
                return GroundedClip(
                    shot_id=shot.id,
                    beat_id=shot.beat_id,
                    source_scene_ids=source_scene_ids,
                    source_range=source_range,
                    target_duration_s=shot.target_duration_s,
                    confidence=decision.confidence,
                    evidence_frame_ids=decision.evidence_frame_ids,
                    evidence_timestamps_s=[
                        frame_timestamps[frame_id]
                        for frame_id in decision.evidence_frame_ids
                    ],
                    rationale=decision.rationale,
                )
        if not localized:
            raise ValueError(f"No candidate visually matches {shot.id}")
        raise ValueError(f"No candidate boundary could be refined for {shot.id}")

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        """Ground only the source artifacts declared by the Director's task."""
        if task.assignee != self.role:
            raise ValueError("Grounding can only execute its own tasks")

        staging_dir: Path | None = None
        try:
            index, story_summary, storyboard, narration = self._load_artifacts(task)
            shots = self._shot_requests(storyboard, narration, story_summary, index)
            destination = artifacts_dir or self.artifacts_dir
            if destination is None:
                raise ValueError("Grounding requires a configured artifacts directory")
            destination.mkdir(parents=True, exist_ok=True)
            if (destination / "grounding.json").exists():
                raise FileExistsError(destination / "grounding.json")
            staging_dir = Path(
                tempfile.mkdtemp(dir=destination, prefix=".grounding-frames.")
            )
            clips = await GroundingBatchProcessor(
                self, max_parallel=self.max_parallel
            ).run(shots, index, story_summary, staging_dir)
            manifest = GroundingManifest(
                source_video=index.video_path,
                clips=clips,
                target_duration_s=sum(clip.target_duration_s for clip in clips),
                source_duration_s=sum(clip.source_range.duration_s for clip in clips),
            )
            manifest_path = self._persist_manifest(manifest, destination)
        except Exception as exc:
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Grounding blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)

        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Grounded {len(manifest.clips)} beats to "
                f"{manifest.source_duration_s:.1f}s of visually verified footage "
                f"for {manifest.target_duration_s:.1f}s of narration."
            ),
            output_artifacts=[
                ArtifactRef(name="grounding-manifest", path=manifest_path)
            ],
        )
        runtime.submit_result(self.role, result)
        return result

    def _refinement_range(
        self,
        candidate_range: TimeRange,
        coarse_range: TimeRange | None,
    ) -> TimeRange:
        if coarse_range is None:
            raise ValueError("Matched coarse grounding has no source range")
        return TimeRange(
            start_s=max(
                candidate_range.start_s,
                coarse_range.start_s - self.refine_margin_s,
            ),
            end_s=min(
                candidate_range.end_s,
                coarse_range.end_s + self.refine_margin_s,
            ),
        )

    @staticmethod
    def _validate_decision(
        decision: GroundingDecision,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> None:
        if not decision.rationale.strip():
            raise ValueError("Grounding VLM returned an empty rationale")
        if len(decision.evidence_frame_ids) != len(set(decision.evidence_frame_ids)):
            raise ValueError("Grounding VLM repeated an evidence frame")
        known_frame_ids = {frame.id for frame in frames}
        unknown_frames = set(decision.evidence_frame_ids) - known_frame_ids
        if unknown_frames:
            raise ValueError(
                "Grounding VLM cited unknown frames: "
                + ", ".join(sorted(unknown_frames))
            )
        if not decision.matched:
            return
        if not decision.evidence_frame_ids:
            raise ValueError("Matched grounding decision has no visual evidence")
        source_range = decision.source_range
        if source_range is None:
            raise ValueError("Matched grounding decision has no source range")
        if (
            source_range.start_s < candidate.source_range.start_s
            or source_range.end_s > candidate.source_range.end_s
        ):
            raise ValueError("Grounding VLM returned a range outside its candidate")

    @staticmethod
    def _load_artifacts(
        task: TaskContext,
    ) -> tuple[VideoIndex, StorySummary, Storyboard, NarrationManifest]:
        index_path = GroundingAgent._artifact_path(task, "video-index", "index.json")
        search_path = GroundingAgent._artifact_path(
            task, "scene-search-database", "search.sqlite3"
        )
        summary_path = GroundingAgent._artifact_path(
            task, "story-summary", "story-summary.json"
        )
        storyboard_path = GroundingAgent._artifact_path(
            task, "storyboard", "storyboard.json"
        )
        narration_path = GroundingAgent._artifact_path(
            task, "narration-manifest", "narration.json"
        )
        if not search_path.is_file():
            raise FileNotFoundError(search_path)
        index = VideoIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
        index.search_db_path = search_path
        summary = StorySummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        storyboard = Storyboard.model_validate_json(
            storyboard_path.read_text(encoding="utf-8")
        )
        narration = NarrationManifest.model_validate_json(
            narration_path.read_text(encoding="utf-8")
        )
        return index, validate_story_summary(index, summary), storyboard, narration

    @staticmethod
    def _artifact_path(task: TaskContext, name: str, filename: str) -> Path:
        references = [
            artifact for artifact in task.input_artifacts if artifact.name == name
        ]
        if len(references) != 1 or references[0].path.name != filename:
            raise ValueError(
                f"Grounding task must declare exactly one {name} {filename} "
                "input artifact"
            )
        return references[0].path

    @staticmethod
    def _shot_requests(
        storyboard: Storyboard,
        narration: NarrationManifest,
        story_summary: StorySummary,
        index: VideoIndex,
    ) -> list[ShotRequest]:
        if not storyboard.beats:
            raise ValueError("Grounding requires at least one storyboard beat")
        beat_ids = [beat.id for beat in storyboard.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be unique")
        segments = {segment.beat_id: segment for segment in narration.segments}
        if len(segments) != len(narration.segments) or set(segments) != set(beat_ids):
            raise ValueError("Narration must contain exactly one segment per beat")
        sequences = {sequence.id: sequence for sequence in story_summary.sequences}
        scene_ids = {scene.id for scene in index.scenes}
        requests = []
        for beat in storyboard.beats:
            if any(
                not value.strip()
                for value in (
                    beat.id,
                    beat.story_content,
                    beat.narration,
                    beat.visual_intent,
                    beat.mood,
                )
            ):
                raise ValueError("Storyboard contains empty grounding input")
            unknown_sequences = set(beat.source_sequence_ids) - set(sequences)
            if unknown_sequences:
                raise ValueError(
                    f"Storyboard beat {beat.id} references unknown sequences: "
                    + ", ".join(sorted(unknown_sequences))
                )
            allowed_scene_ids = {
                scene_id
                for sequence_id in beat.source_sequence_ids
                for scene_id in sequences[sequence_id].scene_ids
            }
            unknown_evidence = set(beat.evidence_scene_ids) - allowed_scene_ids
            if unknown_evidence or not set(beat.evidence_scene_ids) <= scene_ids:
                raise ValueError(
                    f"Storyboard beat {beat.id} has invalid evidence scenes"
                )
            segment = segments[beat.id]
            if segment.text != beat.narration:
                raise ValueError(
                    f"Narration text does not match storyboard beat {beat.id}"
                )
            requests.append(
                ShotRequest(
                    id=f"shot-{beat.id}",
                    beat_id=beat.id,
                    narration_text=segment.text,
                    story_content=beat.story_content,
                    visual_query=beat.visual_intent,
                    mood=beat.mood,
                    target_duration_s=segment.duration_s,
                    source_sequence_ids=beat.source_sequence_ids,
                    evidence_scene_ids=beat.evidence_scene_ids,
                )
            )
        return requests

    @staticmethod
    def _persist_manifest(manifest: GroundingManifest, artifacts_dir: Path) -> Path:
        path = artifacts_dir / "grounding.json"
        temporary: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=artifacts_dir, prefix=".grounding.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary = Path(raw_path)
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(manifest.model_dump_json(indent=2) + "\n")
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


class GroundingBatchProcessor:
    """Ground storyboard beats concurrently without imposing duration rejection."""

    def __init__(self, agent: GroundingAgent, max_parallel: int = 2) -> None:
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        self.agent = agent
        self.max_parallel = max_parallel

    async def run(
        self,
        shots: list[ShotRequest],
        index: VideoIndex,
        story_summary: StorySummary,
        work_dir: Path,
    ) -> list[GroundedClip]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def ground(shot: ShotRequest) -> GroundedClip:
            async with semaphore:
                return await self.agent.run(
                    shot,
                    index,
                    story_summary,
                    work_dir / shot.id,
                )

        return list(await asyncio.gather(*(ground(shot) for shot in shots)))
