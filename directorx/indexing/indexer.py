from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from directorx.core.models import (
    DialogueLine,
    Scene,
    SceneTags,
    Shot,
    TimeRange,
    VideoIndex,
)
from directorx.core.ports import (
    DenseCaptioner,
    EmbeddingProvider,
    KeyframeExtractor,
    SceneTagger,
    ShotDetector,
    Transcriber,
)

from .cache import VideoIndexCache
from .grouping import VisualSceneGrouper
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
    """Shot detection -> visual scene grouping -> captions -> tags -> index."""

    INDEX_VERSION = 10

    def __init__(
        self,
        *,
        cache_dir: Path,
        shot_detector: ShotDetector,
        scene_grouper: VisualSceneGrouper,
        transcriber: Transcriber,
        keyframe_extractor: KeyframeExtractor,
        captioner: DenseCaptioner,
        tagger: SceneTagger,
        embedding_provider: EmbeddingProvider,
        batch_size: int = 32,
    ) -> None:
        self.cache = VideoIndexCache(cache_dir)
        self.shot_detector = shot_detector
        self.scene_grouper = scene_grouper
        self.transcriber = transcriber
        self.keyframe_extractor = keyframe_extractor
        self.captioner = captioner
        self.tagger = tagger
        self.embedding_provider = embedding_provider
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size

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
        detected_shots, dialogue = await asyncio.gather(
            self.shot_detector.detect(source, duration),
            self.transcriber.transcribe(source),
        )
        shots = self._normalize_shots(detected_shots, duration)
        for shot in shots:
            shot.dialogue = self._dialogue_for_range(dialogue, shot.source_range)

        if not all(shot.keyframes for shot in shots):
            frames = await self.keyframe_extractor.extract(
                source, shots, decision.index_dir / "keyframes"
            )
            for shot in shots:
                shot.keyframes = frames.get(shot.id, [])

        scenes = await self.scene_grouper.group(shots)
        for scene in scenes:
            scene.transcript = " ".join(line.text for line in scene.dialogue)

        dense_captions = await self._caption_with_checkpoint(
            scenes, decision.index_dir / "dense-captions-v1.json"
        )
        for scene in scenes:
            if scene.id not in dense_captions:
                raise ValueError(f"Dense captioner omitted {scene.id}")
            scene.dense_caption = dense_captions[scene.id]

        scene_tags = await self._tag_with_checkpoint(
            scenes, decision.index_dir / f"scene-tags-v{self.INDEX_VERSION}.json"
        )
        for scene in scenes:
            tags = scene_tags.get(scene.id)
            if tags is None:
                raise ValueError(f"Scene tagger omitted {scene.id}")
            scene.caption = tags.caption
            scene.short_summary = tags.short_summary
            scene.tags = tags.tags
            scene.characters = tags.characters
            scene.actions = tags.actions
            scene.location = tags.location
            scene.objects = tags.objects

        documents = [scene_document(scene) for scene in scenes]
        embeddings = await self.embedding_provider.embed(documents)
        if len(embeddings) != len(scenes):
            raise ValueError("Embedding provider returned the wrong number of vectors")

        search_path = decision.index_dir / "search.sqlite3"
        index = VideoIndex(
            video_path=source,
            duration_s=duration,
            scenes=scenes,
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

    async def _caption_with_checkpoint(
        self, scenes: list[Scene], checkpoint_path: Path
    ) -> dict[str, str]:
        cached: dict[str, str] = {}
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
                caption = entry.get("dense_caption")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(
                        f"Invalid dense caption checkpoint entry for {scene.id}"
                    )
                cached[scene.id] = caption
                continue
            pending.append(scene)

        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset : offset + self.batch_size]
            captions = await self.captioner.caption_batch(batch)
            missing = [scene.id for scene in batch if scene.id not in captions]
            if missing:
                raise ValueError(
                    f"Dense captioner omitted scene ids: {', '.join(missing[:8])}"
                )
            cached.update(captions)
            checkpoint = {
                scene.id: {
                    "signature": self._scene_signature(scene),
                    "dense_caption": cached[scene.id],
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

    async def _tag_with_checkpoint(
        self, scenes: list[Scene], checkpoint_path: Path
    ) -> dict[str, SceneTags]:
        cached: dict[str, SceneTags] = {}
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
            ) == self._tag_signature(scene):
                cached[scene.id] = SceneTags.model_validate(entry["tags"])
                continue
            pending.append(scene)

        for offset in range(0, len(pending), self.batch_size):
            batch = pending[offset : offset + self.batch_size]
            tags = await self.tagger.tag_batch(batch)
            missing = [scene.id for scene in batch if scene.id not in tags]
            if missing:
                raise ValueError(
                    f"Scene tagger omitted scene ids: {', '.join(missing[:8])}"
                )
            cached.update(tags)
            checkpoint = {
                scene.id: {
                    "signature": self._tag_signature(scene),
                    "tags": cached[scene.id].model_dump(mode="json"),
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
        shots = ",".join(
            f"{shot.id}:{shot.source_range.start_s:.6f}:{shot.source_range.end_s:.6f}"
            for shot in scene.shots
        )
        frames = ",".join(
            f"{frame.timestamp_s:.6f}:{frame.path.name}" for frame in scene.keyframes
        )
        return (
            f"{scene.source_range.start_s:.6f}:{scene.source_range.end_s:.6f}:"
            f"{shots}:{frames}"
        )

    @classmethod
    def _tag_signature(cls, scene: Scene) -> str:
        return f"{cls._scene_signature(scene)}:{scene.transcript}:{scene.dense_caption}"

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
    def _normalize_shots(shots: list[Shot], duration_s: float) -> list[Shot]:
        ordered = sorted(shots, key=lambda value: value.source_range.start_s)
        # First make the timeline continuous. Detector rounding can otherwise
        # leave tiny holes that are never searchable or rendered.
        continuous: list[TimeRange] = []
        cursor = 0.0
        for shot in ordered:
            value = shot.source_range
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

        # Shot detectors can emit tiny tail fragments around a timestamp or
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

        return [
            Shot(
                id=f"shot-{index:05d}",
                source_range=value,
            )
            for index, value in enumerate(continuous)
        ]
