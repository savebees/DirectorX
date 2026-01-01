from __future__ import annotations

import asyncio
from collections import Counter

from directorx.core.models import (
    GroundedClip,
    Scene,
    ShotRequest,
    TimeRange,
    VideoIndex,
)
from directorx.core.ports import EmbeddingProvider, GroundingModel
from directorx.indexing.store import SceneSearchStore


class SceneRetriever:
    """Retrieve grounding candidates from the persisted hybrid index."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    async def search(
        self, shot: ShotRequest, index: VideoIndex, limit: int = 8
    ) -> list[Scene]:
        if index.search_db_path is None or not index.search_db_path.exists():
            raise RuntimeError("Grounding requires a persisted scene search index")
        hits = await SceneSearchStore(
            index.search_db_path, self.embedding_provider
        ).search(
            f"{shot.visual_query} {shot.mood} {shot.narration_text}",
            limit=limit,
        )
        scenes = {scene.id: scene for scene in index.scenes}
        return [scenes[hit.scene_id] for hit in hits]


class GroundingAgent:
    def __init__(self, model: GroundingModel, retriever: SceneRetriever) -> None:
        self.model = model
        self.retriever = retriever

    async def run(
        self, shot: ShotRequest, index: VideoIndex
    ) -> list[tuple[Scene, float, str]]:
        candidates = await self.retriever.search(shot, index)
        scores = await self.model.score(shot, candidates)
        scenes = {scene.id: scene for scene in candidates}
        return [
            (scenes[item.scene_id], item.score, item.rationale)
            for item in sorted(scores, key=lambda item: item.score, reverse=True)
        ]


class GroundingBatchProcessor:
    """Ground a shot batch concurrently while enforcing scene reuse limits."""

    def __init__(
        self, agent: GroundingAgent, max_parallel: int = 6, reuse_limit: int = 1
    ) -> None:
        self.agent = agent
        self.max_parallel = max_parallel
        self.reuse_limit = reuse_limit

    async def run(
        self, shots: list[ShotRequest], index: VideoIndex
    ) -> list[GroundedClip]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def localize(shot: ShotRequest) -> list[tuple[Scene, float, str]]:
            async with semaphore:
                return await self.agent.run(shot, index)

        rankings = await asyncio.gather(*(localize(shot) for shot in shots))
        use_count: Counter[str] = Counter()
        clips: list[GroundedClip] = []

        for shot, ranked in zip(shots, rankings, strict=True):
            candidates = [
                item
                for item in ranked
                if use_count[item[0].id] < self.reuse_limit
                and item[0].source_range.duration_s >= shot.target_duration_s
            ]
            if not candidates:
                raise ValueError(
                    f"No unused scene can cover {shot.id} for "
                    f"{shot.target_duration_s:.2f}s"
                )
            scene, confidence, rationale = candidates[0]
            use_count[scene.id] += 1
            source_range = TimeRange(
                start_s=scene.source_range.start_s,
                end_s=scene.source_range.start_s + shot.target_duration_s,
            )
            clips.append(
                GroundedClip(
                    shot_id=shot.id,
                    beat_id=shot.beat_id,
                    source_range=source_range,
                    target_duration_s=shot.target_duration_s,
                    confidence=confidence,
                    rationale=rationale,
                )
            )
        return clips
