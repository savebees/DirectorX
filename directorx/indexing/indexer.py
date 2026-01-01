from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from directorx.core.models import (
    DialogueLine,
    Scene,
    SceneAnnotation,
    TimeRange,
    VideoIndex,
)
from directorx.core.ports import (
    EmbeddingProvider,
    KeyframeExtractor,
    SceneAnnotator,
    SceneDetector,
    Transcriber,
)

from .cache import VideoIndexCache
from .store import SceneSearchStore, scene_document


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"Video duration must be positive: {path}")
    return duration


class HybridVideoIndexer:
    """Scene detection -> dialogue -> keyframes -> VLM labels -> search index."""

    INDEX_VERSION = 6

    def __init__(
        self,
        *,
        cache_dir: Path,
        scene_detector: SceneDetector,
        transcriber: Transcriber,
        keyframe_extractor: KeyframeExtractor,
        annotator: SceneAnnotator,
        embedding_provider: EmbeddingProvider,
        max_scene_duration_s: float = 15.0,
        annotation_batch_size: int = 32,
    ) -> None:
        self.cache = VideoIndexCache(cache_dir)
        self.scene_detector = scene_detector
        self.transcriber = transcriber
        self.keyframe_extractor = keyframe_extractor
        self.annotator = annotator
        self.embedding_provider = embedding_provider
        if max_scene_duration_s <= 0:
            raise ValueError("max_scene_duration_s must be positive")
        if annotation_batch_size <= 0:
            raise ValueError("annotation_batch_size must be positive")
        self.max_scene_duration_s = max_scene_duration_s
        self.annotation_batch_size = annotation_batch_size

    async def build(self, video_path: Path) -> VideoIndex:
        source = video_path.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        decision = await asyncio.to_thread(self.cache.resolve, source)
        if (
            decision.cached_index is not None
            and decision.cached_index.index_version == self.INDEX_VERSION
            and decision.cached_index.search_db_path is not None
            and decision.cached_index.search_db_path.exists()
        ):
            return decision.cached_index

        duration = await asyncio.to_thread(_probe_duration, source)
        # Index schema changes invalidate semantic labels, but deterministic
        # scene boundaries from an earlier index remain valid for the same
        # content fingerprint. Reuse them so a resumed online annotation run
        # does not decode a feature film a second time.
        cached_ranges = (
            [scene.source_range for scene in decision.cached_index.scenes]
            if decision.cached_index is not None
            else None
        )
        detected = (
            cached_ranges
            if cached_ranges
            else await self.scene_detector.detect(source, duration)
        )
        ranges, dialogue = await asyncio.gather(
            asyncio.sleep(0, result=detected),
            self.transcriber.transcribe(source),
        )
        ranges = self._normalize_ranges(
            ranges, duration, max_scene_duration_s=self.max_scene_duration_s
        )
        scenes = [
            Scene(
                id=f"scene-{index:05d}",
                source_range=time_range,
                caption="",
                dialogue=self._dialogue_for_range(dialogue, time_range),
                keyframes=(
                    decision.cached_index.scenes[index].keyframes
                    if decision.cached_index is not None
                    and index < len(decision.cached_index.scenes)
                    else []
                ),
            )
            for index, time_range in enumerate(ranges)
        ]
        for scene in scenes:
            scene.transcript = " ".join(line.text for line in scene.dialogue)

        if not all(scene.keyframes for scene in scenes):
            frames = await self.keyframe_extractor.extract(
                source, scenes, decision.index_dir / "keyframes"
            )
            for scene in scenes:
                scene.keyframes = frames.get(scene.id, [])

        annotations = await self._annotate_with_checkpoint(
            scenes, decision.index_dir / f"annotations-v{self.INDEX_VERSION}.json"
        )
        for scene in scenes:
            annotation = annotations.get(scene.id)
            if annotation is None:
                raise ValueError(f"Scene annotator omitted {scene.id}")
            scene.caption = annotation.caption
            scene.tags = annotation.tags
            scene.characters = annotation.characters
            scene.actions = annotation.actions
            scene.location = annotation.location
            scene.objects = annotation.objects
            scene.mood_scores = annotation.mood_scores
            scene.plot_event = annotation.plot_event
            scene.annotation_confidence = annotation.confidence
            if annotation.mood_scores:
                scene.mood = max(annotation.mood_scores, key=annotation.mood_scores.get)

        documents = [scene_document(scene) for scene in scenes]
        embeddings = await self.embedding_provider.embed(documents)
        if len(embeddings) != len(scenes):
            raise ValueError("Embedding provider returned the wrong number of vectors")

        search_path = decision.index_dir / "search.sqlite3"
        index = VideoIndex(
            video_path=source,
            duration_s=duration,
            scenes=scenes,
            content_fingerprint=decision.fingerprint,
            index_version=self.INDEX_VERSION,
            search_db_path=search_path,
        )
        await asyncio.to_thread(
            SceneSearchStore(search_path, self.embedding_provider).build,
            index,
            embeddings,
        )
        await asyncio.to_thread(self.cache.commit, source, index)
        return index

    async def _annotate_with_checkpoint(
        self, scenes: list[Scene], checkpoint_path: Path
    ) -> dict[str, SceneAnnotation]:
        cached: dict[str, SceneAnnotation] = {}
        raw: dict[str, object] = {}
        if checkpoint_path.exists():
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a JSON object in {checkpoint_path}")

        pending: list[Scene] = []
        for scene in scenes:
            entry = raw.get(scene.id)
            if isinstance(entry, dict) and entry.get(
                "signature"
            ) == self._scene_signature(scene):
                cached[scene.id] = SceneAnnotation.model_validate(entry["annotation"])
                continue
            pending.append(scene)

        for offset in range(0, len(pending), self.annotation_batch_size):
            batch = pending[offset : offset + self.annotation_batch_size]
            annotations = await self.annotator.annotate_batch(batch)
            missing = [scene.id for scene in batch if scene.id not in annotations]
            if missing:
                raise ValueError(
                    f"Scene annotator omitted scene ids: {', '.join(missing[:8])}"
                )
            cached.update(annotations)
            checkpoint = {
                scene.id: {
                    "signature": self._scene_signature(scene),
                    "annotation": cached[scene.id].model_dump(mode="json"),
                }
                for scene in scenes
                if scene.id in cached
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(checkpoint_path)
        return cached

    @staticmethod
    def _scene_signature(scene: Scene) -> str:
        frames = ",".join(
            f"{frame.timestamp_s:.6f}:{frame.path.name}" for frame in scene.keyframes
        )
        return (
            f"{scene.source_range.start_s:.6f}:{scene.source_range.end_s:.6f}:{frames}"
        )

    @staticmethod
    def _dialogue_for_range(
        dialogue: list[DialogueLine], time_range: TimeRange
    ) -> list[DialogueLine]:
        return [
            line
            for line in dialogue
            if line.end_s > time_range.start_s and line.start_s < time_range.end_s
        ]

    @staticmethod
    def _normalize_ranges(
        ranges: list[TimeRange],
        duration_s: float,
        *,
        max_scene_duration_s: float = 15.0,
    ) -> list[TimeRange]:
        if max_scene_duration_s <= 0:
            raise ValueError("max_scene_duration_s must be positive")
        ordered = sorted(ranges, key=lambda value: value.start_s)
        # First make the timeline continuous. Detector rounding can otherwise
        # leave tiny holes that are never searchable or rendered.
        continuous: list[TimeRange] = []
        cursor = 0.0
        for value in ordered:
            start = min(duration_s, max(cursor, value.start_s))
            end = min(duration_s, max(start, value.end_s))
            if end - start <= 1e-3:
                continue
            if start > cursor + 1e-3:
                continuous.append(TimeRange(start_s=cursor, end_s=start))
            continuous.append(TimeRange(start_s=start, end_s=end))
            cursor = end
        if cursor < duration_s - 1e-3:
            continuous.append(TimeRange(start_s=cursor, end_s=duration_s))
        if not continuous:
            continuous = [TimeRange(start_s=0, end_s=duration_s)]

        # Scene detectors can emit tiny tail fragments around a timestamp or
        # encoder boundary. They cannot yield a decodable representative frame,
        # so absorb them into the preceding searchable window.
        merged: list[TimeRange] = []
        for value in continuous:
            if merged and value.duration_s < 0.25:
                previous = merged.pop()
                merged.append(TimeRange(start_s=previous.start_s, end_s=value.end_s))
            else:
                merged.append(value)
        continuous = merged

        normalized: list[TimeRange] = []
        for value in continuous:
            start = value.start_s
            while start < value.end_s - 1e-3:
                end = min(value.end_s, start + max_scene_duration_s)
                normalized.append(TimeRange(start_s=start, end_s=end))
                start = end
        return normalized
