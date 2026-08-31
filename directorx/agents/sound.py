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
    MusicIndex,
    MusicIndexEntry,
    MusicTrack,
    NarrationManifest,
    SoundPlan,
    Storyboard,
    TimeRange,
)
from directorx.core.ports import AudioTextEmbeddingProvider, MusicLibrary


class SoundAgent:
    """Select one semantically matched background track for the complete edit."""

    role = AgentRole.SOUND

    def __init__(
        self,
        music_source: MusicLibrary | AudioTextEmbeddingProvider,
        embedding_provider: AudioTextEmbeddingProvider | None = None,
        *,
        artifacts_dir: Path | None = None,
        analysis_window_s: float = 10.0,
        analysis_windows_per_track: int = 3,
        gain_db: float = -23.0,
        duck_under_voice_db: float = -9.0,
        require_music_index: bool = False,
    ) -> None:
        if analysis_window_s <= 0:
            raise ValueError("analysis_window_s must be positive")
        if analysis_windows_per_track <= 0:
            raise ValueError("analysis_windows_per_track must be positive")
        # ``music_source`` accepts the former library-first shape for callers
        # that use ``run`` directly. Task execution always reads music-index.json.
        self.music_library = music_source if embedding_provider is not None else None
        self.embedding_provider = embedding_provider or music_source
        self.artifacts_dir = artifacts_dir
        self.analysis_window_s = analysis_window_s
        self.analysis_windows_per_track = analysis_windows_per_track
        self.gain_db = gain_db
        self.duck_under_voice_db = duck_under_voice_db
        self.require_music_index = require_music_index

    async def run(
        self,
        storyboard: Storyboard,
        narration: NarrationManifest,
        music_index: MusicIndex | None = None,
    ) -> SoundPlan:
        storyboard = Storyboard.model_validate(storyboard)
        narration = NarrationManifest.model_validate(narration)
        mood_weights = self._mood_weights(storyboard, narration)
        query = self._music_query(storyboard, mood_weights)
        query_embedding = self._normalize(
            await self.embedding_provider.embed_text(query), "music query"
        )

        if music_index is not None:
            return self._select_from_index(
                music_index, query_embedding, query, narration.duration_s, mood_weights
            )

        if self.music_library is None:
            raise ValueError("Sound requires a music index")
        tracks = sorted(
            await self.music_library.tracks(), key=lambda item: str(item.path)
        )
        self._validate_tracks(tracks)

        scored: list[tuple[float, float, MusicTrack, list[TimeRange]]] = []
        for track in tracks:
            windows = self._analysis_windows(track.duration_s)
            window_embeddings = await self.embedding_provider.embed_audio(
                track.path, windows
            )
            if len(window_embeddings) != len(windows):
                raise ValueError(
                    f"Music embedding count does not match windows for {track.path}"
                )
            audio_embedding = self._mean_embedding(
                window_embeddings, f"music track {track.path}"
            )
            similarity = self._cosine(query_embedding, audio_embedding)
            folded_tags = {tag.casefold() for tag in track.tags}
            tag_score = sum(
                weight
                for mood, weight in mood_weights.items()
                if mood.casefold() in folded_tags
            )
            scored.append((similarity, tag_score, track, windows))

        similarity, _, track, windows = min(
            scored,
            key=lambda item: (-item[0], -item[1], str(item[2].path)),
        )
        return SoundPlan(
            track=track,
            target_duration_s=narration.duration_s,
            match_score=similarity,
            gain_db=self.gain_db,
            duck_under_voice_db=self.duck_under_voice_db,
            selection_rationale=(
                "Selected the highest-scoring whole-edit music match from "
                f"{len(tracks)} tracks using {len(windows)} sampled audio windows. "
                f"Query: {query}"
            ),
        )

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        """Select music using only the artifacts declared by the Director."""
        if task.assignee != self.role:
            raise ValueError("Sound can only execute its own tasks")

        try:
            storyboard, narration, music_index = self._load_artifacts(task)
            plan = await self.run(storyboard, narration, music_index)
            destination = artifacts_dir or self.artifacts_dir
            if destination is None:
                raise ValueError("Sound requires a configured artifacts directory")
            plan_path = self._persist_plan(plan, destination)
        except Exception as exc:
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Sound selection blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result

        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Selected one background track, {plan.track.title!r}, for the "
                f"complete {plan.target_duration_s:.1f}s edit."
            ),
            output_artifacts=[ArtifactRef(name="sound-plan", path=plan_path)],
        )
        runtime.submit_result(self.role, result)
        return result

    def _load_artifacts(
        self,
        task: TaskContext,
    ) -> tuple[Storyboard, NarrationManifest, MusicIndex | None]:
        storyboard_path = SoundAgent._artifact_path(
            task, "storyboard", "storyboard.json"
        )
        narration_path = SoundAgent._artifact_path(
            task, "narration-manifest", "narration.json"
        )
        index_refs = [
            artifact
            for artifact in task.input_artifacts
            if artifact.name == "music-index"
        ]
        if len(index_refs) == 1 and index_refs[0].path.name == "music-index.json":
            index_path: Path | None = index_refs[0].path
        elif self.music_library is not None and not self.require_music_index:
            index_path = None
        else:
            raise ValueError(
                "Sound task must declare exactly one music-index "
                "music-index.json input artifact"
            )
        storyboard = Storyboard.model_validate_json(
            storyboard_path.read_text(encoding="utf-8")
        )
        narration = NarrationManifest.model_validate_json(
            narration_path.read_text(encoding="utf-8")
        )
        music_index = (
            MusicIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
            if index_path is not None
            else None
        )
        return storyboard, narration, music_index

    def _select_from_index(
        self,
        music_index: MusicIndex,
        query_embedding: list[float],
        query: str,
        target_duration_s: float,
        mood_weights: dict[str, float],
    ) -> SoundPlan:
        if not music_index.entries:
            raise ValueError("Music index contains no tracks")
        provider_model = getattr(self.embedding_provider, "model_name", None)
        if provider_model and music_index.model_name != provider_model:
            raise ValueError(
                "Music index model does not match the configured CLAP model"
            )
        if music_index.embedding_dimension != len(query_embedding):
            raise ValueError("Music index and query embedding dimensions do not match")
        paths = [entry.track.path for entry in music_index.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("Music index contains duplicate track paths")
        scored: list[tuple[float, float, MusicIndexEntry]] = []
        entries = self._production_entries(music_index.entries)
        for entry in entries:
            if entry.model_name != music_index.model_name:
                raise ValueError("Music index entry model does not match the index")
            embedding = self._normalize(
                entry.embedding, f"music track {entry.track.path}"
            )
            if len(embedding) != music_index.embedding_dimension:
                raise ValueError(
                    "Music index contains inconsistent embedding dimensions"
                )
            similarity = self._cosine(query_embedding, embedding)
            folded_tags = {tag.casefold() for tag in entry.track.tags}
            tag_score = sum(
                weight
                for mood, weight in mood_weights.items()
                if mood.casefold() in folded_tags
            )
            scored.append((similarity, tag_score, entry))
        similarity, _, entry = min(
            scored,
            key=lambda item: (-item[0], -item[1], str(item[2].track.path)),
        )
        return SoundPlan(
            track=entry.track,
            target_duration_s=target_duration_s,
            match_score=similarity,
            gain_db=self.gain_db,
            duck_under_voice_db=self.duck_under_voice_db,
            selection_rationale=(
                "Selected the highest-scoring whole-edit music match from "
                f"{len(entries)} eligible indexed tracks using "
                f"{music_index.model_name}. "
                f"Query: {query}"
            ),
        )

    @staticmethod
    def _production_entries(entries: list[MusicIndexEntry]) -> list[MusicIndexEntry]:
        placeholder_tags = {"test", "tone", "placeholder"}
        production = [
            entry
            for entry in entries
            if not placeholder_tags.intersection(
                tag.casefold() for tag in entry.track.tags
            )
        ]
        return production or entries

    @staticmethod
    def _artifact_path(task: TaskContext, name: str, filename: str) -> Path:
        references = [
            artifact for artifact in task.input_artifacts if artifact.name == name
        ]
        if len(references) != 1 or references[0].path.name != filename:
            raise ValueError(
                f"Sound task must declare exactly one {name} {filename} input artifact"
            )
        return references[0].path

    @staticmethod
    def _mood_weights(
        storyboard: Storyboard, narration: NarrationManifest
    ) -> dict[str, float]:
        if not storyboard.narrative_angle.strip():
            raise ValueError("Storyboard narrative angle is empty")
        if not storyboard.beats:
            raise ValueError("Sound selection requires at least one storyboard beat")
        beat_ids = [beat.id for beat in storyboard.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be unique")
        segments = {segment.beat_id: segment for segment in narration.segments}
        if len(segments) != len(narration.segments) or set(segments) != set(beat_ids):
            raise ValueError("Narration must contain exactly one segment per beat")
        if not math.isclose(
            narration.duration_s,
            sum(segment.duration_s for segment in narration.segments),
            abs_tol=0.1,
        ):
            raise ValueError("Narration duration does not match its segments")

        weights: dict[str, float] = defaultdict(float)
        for beat in storyboard.beats:
            mood = beat.mood.strip()
            if not mood:
                raise ValueError(f"Storyboard beat {beat.id} has an empty mood")
            weights[mood] += segments[beat.id].duration_s
        return dict(weights)

    @staticmethod
    def _music_query(storyboard: Storyboard, mood_weights: dict[str, float]) -> str:
        moods = sorted(mood_weights, key=lambda mood: (-mood_weights[mood], mood))[:3]
        return (
            "Cinematic background music for a video edit. "
            f"Narrative angle: {storyboard.narrative_angle.strip()} "
            f"Dominant moods: {', '.join(moods)}."
        )

    def _analysis_windows(self, duration_s: float) -> list[TimeRange]:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Music track duration must be positive and finite")
        window_s = min(self.analysis_window_s, duration_s)
        if duration_s <= window_s or self.analysis_windows_per_track == 1:
            start_s = max(0.0, (duration_s - window_s) / 2)
            return [TimeRange(start_s=start_s, end_s=start_s + window_s)]
        last_start = duration_s - window_s
        starts = [
            last_start * index / (self.analysis_windows_per_track - 1)
            for index in range(self.analysis_windows_per_track)
        ]
        return [TimeRange(start_s=start, end_s=start + window_s) for start in starts]

    @staticmethod
    def _validate_tracks(tracks: list[MusicTrack]) -> None:
        if not tracks:
            raise ValueError("Music library returned no tracks")
        paths = [track.path for track in tracks]
        if len(paths) != len(set(paths)):
            raise ValueError("Music library returned duplicate track paths")

    @classmethod
    def _mean_embedding(cls, embeddings: list[list[float]], label: str) -> list[float]:
        if not embeddings:
            raise ValueError(f"{label} returned no embeddings")
        normalized = [cls._normalize(embedding, label) for embedding in embeddings]
        dimension = len(normalized[0])
        if any(len(embedding) != dimension for embedding in normalized):
            raise ValueError(f"{label} returned inconsistent embedding dimensions")
        mean = [
            sum(embedding[index] for embedding in normalized) / len(normalized)
            for index in range(dimension)
        ]
        return cls._normalize(mean, label)

    @staticmethod
    def _normalize(embedding: list[float], label: str) -> list[float]:
        if not embedding or any(not math.isfinite(value) for value in embedding):
            raise ValueError(f"{label} returned an invalid embedding")
        norm = math.sqrt(sum(value * value for value in embedding))
        if norm == 0:
            raise ValueError(f"{label} returned a zero embedding")
        return [value / norm for value in embedding]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Text and audio embedding dimensions do not match")
        score = sum(a * b for a, b in zip(left, right, strict=True))
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _persist_plan(plan: SoundPlan, artifacts_dir: Path) -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / "sound-plan.json"
        temporary: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=artifacts_dir, prefix=".sound-plan.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary = Path(raw_path)
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(plan.model_dump_json(indent=2) + "\n")
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
