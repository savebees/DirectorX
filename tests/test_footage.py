import asyncio
from pathlib import Path

from directorx.agents import DirectorAgent, FootageAnalystAgent
from directorx.coordination import ArtifactRef, CoordinationRuntime, TaskContext
from directorx.core.models import Scene, TimeRange, VideoIndex


class FixedIndexer:
    def __init__(self, index: VideoIndex) -> None:
        self.index = index
        self.video_path: Path | None = None

    async def build(self, video_path: Path) -> VideoIndex:
        self.video_path = video_path
        return self.index


class FailingIndexer:
    async def build(self, video_path: Path) -> VideoIndex:
        raise RuntimeError("ffprobe failed")


def test_director_delegates_footage_task_and_receives_result(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    search_db = index_dir / "search.sqlite3"
    search_db.write_bytes(b"sqlite")
    (index_dir / "index.json").write_text("{}", encoding="utf-8")
    index = VideoIndex(
        video_path=video,
        duration_s=10,
        scenes=[
            Scene(
                id="scene-00000",
                source_range=TimeRange(start_s=0, end_s=10),
                caption="A person enters a room.",
            )
        ],
        search_db_path=search_db,
    )
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(runtime, FootageAnalystAgent(FixedIndexer(index)))
    task = TaskContext(
        task_id="footage-001",
        assignee="footage_analyst",
        objective="Index the source footage.",
        input_artifacts=[ArtifactRef(name="source-video", path=video)],
        expected_output="A searchable VideoIndex artifact.",
        acceptance_criteria=["Every detected scene has searchable metadata"],
    )

    result = asyncio.run(director.run_footage_task(task, video))

    assert result.status == "completed"
    assert result.summary == "Indexed 10.0 seconds into 1 scenes."
    assert [artifact.name for artifact in result.output_artifacts] == [
        "video-index",
        "scene-search-database",
    ]
    assert all(
        "version" not in artifact.model_dump() for artifact in result.output_artifacts
    )
    assert set(result.model_dump()) == {
        "task_id",
        "agent",
        "status",
        "summary",
        "output_artifacts",
    }
    assert director.read_result(task.task_id) == result


def test_footage_task_persists_blocked_result(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(runtime, FootageAnalystAgent(FailingIndexer()))
    task = TaskContext(
        task_id="footage-002",
        assignee="footage_analyst",
        objective="Index the source footage.",
        expected_output="A searchable VideoIndex artifact.",
        acceptance_criteria=["Return a task result"],
    )

    result = asyncio.run(director.run_footage_task(task, tmp_path / "movie.mp4"))

    assert result.status == "blocked"
    assert result.summary == "Footage indexing blocked: ffprobe failed"
    assert result.output_artifacts == []
    assert director.read_result(task.task_id) == result


def test_footage_analyst_builds_a_video_index() -> None:
    index = VideoIndex(
        video_path=Path("movie.mp4"),
        duration_s=10,
        scenes=[
            Scene(
                id="scene-00000",
                source_range=TimeRange(start_s=0, end_s=10),
                caption="A person enters a room.",
            )
        ],
    )
    indexer = FixedIndexer(index)
    agent = FootageAnalystAgent(indexer)

    result = asyncio.run(agent.run(Path("movie.mp4")))

    assert result == index
    assert indexer.video_path == Path("movie.mp4")
