from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import (
    DialogueLine,
    GroundingCandidate,
    GroundingDecision,
    GroundingFrame,
    Keyframe,
    MusicTrack,
    NarrationDraft,
    RenderPlan,
    Scene,
    SceneTags,
    Screenplay,
    ScreenwriterSceneEvidence,
    Shot,
    ShotRequest,
    StorySummary,
    TimeRange,
    VideoIndex,
)


class VideoIndexer(Protocol):
    async def build(self, video_path: Path) -> VideoIndex: ...


class ShotDetector(Protocol):
    async def detect(self, video_path: Path, duration_s: float) -> list[Shot]: ...


class Transcriber(Protocol):
    async def transcribe(self, video_path: Path) -> list[DialogueLine]: ...


class KeyframeExtractor(Protocol):
    async def extract(
        self, video_path: Path, shots: list[Shot], output_dir: Path
    ) -> dict[str, list[Keyframe]]: ...


class ShotVisualEmbedder(Protocol):
    async def embed(self, shots: list[Shot]) -> list[list[float]]: ...


class DenseCaptioner(Protocol):
    async def caption_batch(self, scenes: list[Scene]) -> dict[str, str]: ...


class SceneTagger(Protocol):
    async def tag_batch(self, scenes: list[Scene]) -> dict[str, SceneTags]: ...


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class ScreenwriterModel(Protocol):
    async def draft_screenplay(
        self,
        objective: str,
        constraints: list[str],
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> Screenplay: ...

    async def draft_narration(
        self,
        objective: str,
        constraints: list[str],
        screenplay: Screenplay,
        evidence_by_beat: dict[str, list[ScreenwriterSceneEvidence]],
    ) -> NarrationDraft: ...


class StoryStructureModel(Protocol):
    async def build(self, video_index: VideoIndex) -> StorySummary: ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str, output_path: Path) -> float:
        """Write audio and return its measured duration in seconds."""
        ...


class GroundingModel(Protocol):
    async def locate(
        self,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> GroundingDecision: ...

    async def refine(
        self,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> GroundingDecision: ...


class GroundingFrameExtractor(Protocol):
    async def extract(
        self,
        video_path: Path,
        source_range: TimeRange,
        output_dir: Path,
        *,
        fps: float,
        max_frames: int,
        prefix: str,
    ) -> list[GroundingFrame]: ...


class MusicLibrary(Protocol):
    async def tracks(self) -> list[MusicTrack]: ...


class RenderEngine(Protocol):
    async def render(self, plan: RenderPlan) -> Path: ...
