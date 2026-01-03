import asyncio
from pathlib import Path

from directorx.agents import FootageAnalystAgent
from directorx.core.models import Scene, TimeRange, VideoIndex


class FixedIndexer:
    def __init__(self, index: VideoIndex) -> None:
        self.index = index
        self.video_path: Path | None = None

    async def build(self, video_path: Path) -> VideoIndex:
        self.video_path = video_path
        return self.index


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
