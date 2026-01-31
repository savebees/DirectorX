from __future__ import annotations

import os
import tempfile
from pathlib import Path

from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import StorySummary, VideoIndex
from directorx.core.ports import StoryStructureModel, VideoIndexer
from directorx.indexing.hierarchy import validate_story_summary
from directorx.indexing.store import SceneSearchStore


class FootageAnalystAgent:
    """Understand a source video and build its searchable scene index."""

    role = AgentRole.FOOTAGE_ANALYST

    def __init__(
        self,
        indexer: VideoIndexer,
        story_structure_model: StoryStructureModel | None = None,
        artifacts_dir: Path | None = None,
        hierarchy_store: SceneSearchStore | None = None,
    ) -> None:
        self.indexer = indexer
        self.story_structure_model = story_structure_model
        self.artifacts_dir = artifacts_dir
        self.hierarchy_store = hierarchy_store

    async def run(self, video_path: Path) -> VideoIndex:
        return await self.indexer.build(video_path)

    async def run_task(
        self,
        task: TaskContext,
        video_path: Path,
        runtime: CoordinationRuntime,
        artifacts_dir: Path | None = None,
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
        if index.search_db_path is None:
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary="Footage indexing blocked: index has no search database path",
            )
            runtime.submit_result(self.role, result)
            return result
        index_path = index.search_db_path.with_name("index.json")
        output_artifacts = [
            ArtifactRef(name="video-index", path=index_path),
            ArtifactRef(name="scene-search-database", path=index.search_db_path),
        ]
        if self.story_structure_model is not None:
            try:
                summary = validate_story_summary(
                    index, await self.story_structure_model.build(index)
                )
                hierarchy_store = self.hierarchy_store
                embedding_provider = getattr(self.indexer, "embedding_provider", None)
                if hierarchy_store is None and embedding_provider is not None:
                    hierarchy_store = SceneSearchStore(
                        index.search_db_path, embedding_provider
                    )
                if hierarchy_store is not None:
                    await hierarchy_store.add_story_hierarchy(index, summary)
                summary_path = self._persist_story_summary(
                    summary,
                    artifacts_dir
                    or self.artifacts_dir
                    or runtime.store.root / "artifacts",
                )
            except Exception as exc:
                result = TaskResult(
                    task_id=task.task_id,
                    agent=self.role,
                    status="blocked",
                    summary=f"Footage hierarchy blocked: {exc}",
                )
                runtime.submit_result(self.role, result)
                return result
            output_artifacts.append(
                ArtifactRef(name="story-summary", path=summary_path)
            )
        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Indexed {index.duration_s:.1f} seconds into "
                f"{len(index.scenes)} scenes."
                + (
                    f" Built {len(summary.sequences)} sequences and "
                    f"{len(summary.acts)} acts."
                    if self.story_structure_model is not None
                    else ""
                )
            ),
            output_artifacts=output_artifacts,
        )
        runtime.submit_result(self.role, result)
        return result

    @staticmethod
    def _persist_story_summary(summary: StorySummary, artifacts_dir: Path) -> Path:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / "story-summary.json"
        if path.exists():
            raise FileExistsError(path)
        descriptor, raw_path = tempfile.mkstemp(
            dir=artifacts_dir, prefix=".story-summary.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(raw_path)
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(summary.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            return path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
