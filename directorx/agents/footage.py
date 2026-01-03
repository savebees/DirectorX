from __future__ import annotations

from pathlib import Path

from directorx.coordination import AgentRole
from directorx.core.models import VideoIndex
from directorx.core.ports import VideoIndexer


class FootageAnalystAgent:
    """Understand a source video and build its searchable scene index."""

    role = AgentRole.FOOTAGE_ANALYST

    def __init__(self, indexer: VideoIndexer) -> None:
        self.indexer = indexer

    async def run(self, video_path: Path) -> VideoIndex:
        return await self.indexer.build(video_path)
