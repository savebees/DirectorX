from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    CandidateScore,
    DialogueLine,
    Keyframe,
    MusicTrack,
    RenderPlan,
    Scene,
    SceneTags,
    ShotRequest,
    Storyboard,
    TimeRange,
    VideoIndex,
)


class VideoIndexer(Protocol):
    async def build(self, video_path: Path) -> VideoIndex: ...


class SceneDetector(Protocol):
    async def detect(self, video_path: Path, duration_s: float) -> list[TimeRange]: ...


class Transcriber(Protocol):
    async def transcribe(self, video_path: Path) -> list[DialogueLine]: ...


class KeyframeExtractor(Protocol):
    async def extract(
        self, video_path: Path, scenes: list[Scene], output_dir: Path
    ) -> dict[str, list[Keyframe]]: ...


class DenseCaptioner(Protocol):
    async def caption_batch(self, scenes: list[Scene]) -> dict[str, str]: ...


class SceneTagger(Protocol):
    async def tag_batch(self, scenes: list[Scene]) -> dict[str, SceneTags]: ...


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ScreenwriterModel(Protocol):
    async def draft(
        self, prompt: str, video_index: VideoIndex, target_duration_s: float
    ) -> Storyboard: ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str, output_path: Path) -> float:
        """Write audio and return its measured duration in seconds."""
        ...


class GroundingModel(Protocol):
    async def score(
        self, shot: ShotRequest, candidates: list[Scene]
    ) -> list[CandidateScore]: ...


class MusicLibrary(Protocol):
    async def tracks(self) -> list[MusicTrack]: ...


class RenderEngine(Protocol):
    async def render(self, plan: RenderPlan) -> Path: ...
