from __future__ import annotations

from pathlib import Path

from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import VideoIndex
from directorx.core.ports import VideoIndexer


class FootageAnalystAgent:
    """Understand a source video and build its searchable scene index."""

    role = AgentRole.FOOTAGE_ANALYST

    def __init__(self, indexer: VideoIndexer) -> None:
        self.indexer = indexer

    async def run(self, video_path: Path) -> VideoIndex:
        return await self.indexer.build(video_path)

    async def run_task(
        self,
        task: TaskContext,
        video_path: Path,
        runtime: CoordinationRuntime,
    ) -> TaskResult:
        if task.assignee != self.role:
            raise ValueError("Footage Analyst can only execute its own tasks")
        try:
            index = await self.run(video_path)
        except Exception as exc:
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Footage indexing blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result
        index_path = index.search_db_path.with_name("index.json")
        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Indexed {index.duration_s:.1f} seconds into "
                f"{len(index.scenes)} scenes."
            ),
            output_artifacts=[
                ArtifactRef(name="video-index", path=index_path),
                ArtifactRef(name="scene-search-database", path=index.search_db_path),
            ],
        )
        runtime.submit_result(self.role, result)
        return result
