import asyncio
from pathlib import Path

from directorx.agents import DirectorAgent, FootageAnalystAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
)
from directorx.core.models import (
    Scene,
    StoryAct,
    StorySequence,
    StorySummary,
    TimeRange,
    VideoIndex,
)


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


class FixedStoryStructureModel:
    async def build(self, video_index: VideoIndex) -> StorySummary:
        scene_ids = [scene.id for scene in video_index.scenes]
        return StorySummary(
            title="Indexed story",
            short_summary="A person enters and leaves a room.",
            sequences=[
                StorySequence(
                    id="model-sequence",
                    title="The entrance",
                    short_summary="A person enters the room.",
                    scene_ids=scene_ids,
                )
            ],
            acts=[
                StoryAct(
                    id="model-act",
                    title="Opening",
                    short_summary="The story begins with an entrance.",
                    sequence_ids=["model-sequence"],
                )
            ],
        )


class FailingStoryStructureModel:
    async def build(self, video_index: VideoIndex) -> StorySummary:
        raise RuntimeError("story model unavailable")


class InvalidStoryStructureModel:
    async def build(self, video_index: VideoIndex) -> StorySummary:
        return StorySummary(
            title="Invalid",
            short_summary="Invalid summary",
            sequences=[
                StorySequence(
                    id="sequence-1",
                    title="Unknown source",
                    short_summary="This references a missing scene.",
                    scene_ids=["scene-missing"],
                )
            ],
            acts=[
                StoryAct(
                    id="act-1",
                    title="Invalid act",
                    short_summary="Invalid",
                    sequence_ids=["sequence-1"],
                )
            ],
        )


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


def test_footage_task_builds_and_persists_story_summary(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    search_db = index_dir / "search.sqlite3"
    search_db.write_bytes(b"sqlite")
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
    agent = FootageAnalystAgent(
        FixedIndexer(index),
        story_structure_model=FixedStoryStructureModel(),
        artifacts_dir=tmp_path / "artifacts",
    )
    task = TaskContext(
        task_id="footage-story-001",
        assignee=AgentRole.FOOTAGE_ANALYST,
        objective="Index the source footage and summarize its story.",
        expected_output="A layered video index and story summary.",
        acceptance_criteria=["Every scene has a hierarchy parent"],
    )
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(agent.run_task(task, video, runtime))

    assert result.status == "completed"
    assert [artifact.name for artifact in result.output_artifacts] == [
        "video-index",
        "scene-search-database",
        "story-summary",
    ]
    summary_path = tmp_path / "artifacts" / "story-summary.json"
    summary = StorySummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    assert summary.acts[0].source_range == TimeRange(start_s=0, end_s=10)
    assert summary.sequences[0].source_range == TimeRange(start_s=0, end_s=10)


def test_footage_task_persists_blocked_result_when_story_model_fails(
    tmp_path: Path,
) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    search_db = tmp_path / "index" / "search.sqlite3"
    search_db.parent.mkdir()
    search_db.write_bytes(b"sqlite")
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
    agent = FootageAnalystAgent(FixedIndexer(index), FailingStoryStructureModel())
    task = TaskContext(
        task_id="footage-story-002",
        assignee=AgentRole.FOOTAGE_ANALYST,
        objective="Index and summarize.",
        expected_output="A story summary.",
        acceptance_criteria=["Return a task result"],
    )
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(agent.run_task(task, video, runtime))

    assert result.status == "blocked"
    assert "story model unavailable" in result.summary
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_footage_task_blocks_invalid_story_references(tmp_path: Path) -> None:
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    search_db = tmp_path / "index" / "search.sqlite3"
    search_db.parent.mkdir()
    search_db.write_bytes(b"sqlite")
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
    agent = FootageAnalystAgent(FixedIndexer(index), InvalidStoryStructureModel())
    task = TaskContext(
        task_id="footage-story-003",
        assignee=AgentRole.FOOTAGE_ANALYST,
        objective="Index and summarize.",
        expected_output="A story summary.",
        acceptance_criteria=["Return a task result"],
    )
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(agent.run_task(task, video, runtime))

    assert result.status == "blocked"
    assert "unknown scene" in result.summary
