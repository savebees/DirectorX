from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from directorx.agents.grounding import (
    GroundingAgent,
    GroundingBatchProcessor,
    SceneRetriever,
)
from directorx.core.models import Scene, ShotRequest, TimeRange, VideoIndex
from directorx.indexing import HashingEmbeddingProvider, SceneSearchStore
from directorx.indexing.store import scene_document
from tests.fakes import FixedGroundingModel


def _persist_search_index(
    tmp_path: Path, index: VideoIndex, provider: HashingEmbeddingProvider
) -> None:
    embeddings = asyncio.run(
        provider.embed([scene_document(scene) for scene in index.scenes])
    )
    search_path = tmp_path / "search.sqlite3"
    SceneSearchStore(search_path, provider).build(index, embeddings)
    index.search_db_path = search_path


def test_grounding_batch_avoids_scene_reuse(tmp_path: Path) -> None:
    index = VideoIndex(
        video_path=Path("movie.mp4"),
        duration_s=30,
        scenes=[
            Scene(
                id=f"scene-{idx}",
                source_range=TimeRange(start_s=idx * 10, end_s=(idx + 1) * 10),
                caption="hero enters the city",
                tags=["hero", "city"],
            )
            for idx in range(3)
        ],
    )
    shots = [
        ShotRequest(
            id=f"shot-{idx}",
            beat_id=f"beat-{idx}",
            narration_text="hero",
            visual_query="hero enters the city",
            mood="neutral",
            target_duration_s=3,
        )
        for idx in range(3)
    ]
    provider = HashingEmbeddingProvider(dimension=64)
    _persist_search_index(tmp_path, index, provider)
    processor = GroundingBatchProcessor(
        GroundingAgent(FixedGroundingModel(), SceneRetriever(provider)),
        max_parallel=3,
        reuse_limit=1,
    )
    clips = asyncio.run(processor.run(shots, index))
    assert len({clip.source_range.start_s for clip in clips}) == 3


def test_grounding_rejects_a_scene_shorter_than_the_requested_shot(
    tmp_path: Path,
) -> None:
    index = VideoIndex(
        video_path=tmp_path / "movie.mp4",
        duration_s=20,
        scenes=[
            Scene(
                id="short",
                source_range=TimeRange(start_s=8, end_s=9),
                caption="door opens",
                tags=["door"],
            ),
        ],
    )
    shot = ShotRequest(
        id="shot-short",
        beat_id="beat-1",
        narration_text="door",
        visual_query="door",
        mood="neutral",
        target_duration_s=5,
    )
    provider = HashingEmbeddingProvider(dimension=64)
    _persist_search_index(tmp_path, index, provider)
    processor = GroundingBatchProcessor(
        GroundingAgent(FixedGroundingModel(), SceneRetriever(provider)),
        reuse_limit=1,
    )
    with pytest.raises(ValueError, match="No unused scene can cover"):
        asyncio.run(processor.run([shot], index))


def test_scene_retriever_prefers_persisted_hybrid_index(tmp_path: Path) -> None:
    scenes = [
        Scene(
            id="a",
            source_range=TimeRange(start_s=0, end_s=5),
            caption="主角发现秘密箱子",
            transcript="秘密箱子在桌下",
            tags=["仓库"],
        ),
        Scene(
            id="b",
            source_range=TimeRange(start_s=5, end_s=10),
            caption="人物走出房间",
            tags=["街道"],
        ),
    ]
    provider = HashingEmbeddingProvider(dimension=64)
    index = VideoIndex(video_path=tmp_path / "movie.mp4", duration_s=10, scenes=scenes)
    embeddings = asyncio.run(
        provider.embed([scene_document(scene) for scene in scenes])
    )
    search_path = tmp_path / "search.sqlite3"
    SceneSearchStore(search_path, provider).build(index, embeddings)
    index.search_db_path = search_path
    shot = ShotRequest(
        id="shot",
        beat_id="beat",
        narration_text="秘密箱子",
        visual_query="发现线索",
        mood="neutral",
        target_duration_s=3,
    )
    result = asyncio.run(SceneRetriever(provider).search(shot, index))
    assert result[0].id == "a"
