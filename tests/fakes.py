from __future__ import annotations

from pathlib import Path

from directorx.core.models import (
    GroundingCandidate,
    GroundingDecision,
    GroundingFrame,
    MusicTrack,
    ReviewReport,
    ShotRequest,
    TimeRange,
)


class FixedReviewFrameExtractor:
    def __init__(self, frames: list[GroundingFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[Path, Path, int]] = []

    async def extract(
        self, video_path: Path, output_dir: Path, *, max_frames: int
    ) -> list[GroundingFrame]:
        self.calls.append((video_path, output_dir, max_frames))
        return self.frames


class FixedReviewModel:
    def __init__(
        self, report: ReviewReport | None = None, failure: Exception | None = None
    ) -> None:
        self.report = report or ReviewReport(
            passed=True, summary="Video is complete and coherent."
        )
        self.failure = failure
        self.calls: list[tuple[float, list[GroundingFrame]]] = []

    async def inspect(
        self, duration_s: float, frames: list[GroundingFrame]
    ) -> ReviewReport:
        self.calls.append((duration_s, frames))
        if self.failure is not None:
            raise self.failure
        return self.report


class FixedMusicLibrary:
    def __init__(self, tracks: list[MusicTrack]) -> None:
        self._tracks = tracks
        self.calls = 0

    async def tracks(self) -> list[MusicTrack]:
        self.calls += 1
        return self._tracks


class FixedAudioTextEmbeddingProvider:
    def __init__(
        self,
        audio_embeddings: dict[str, list[list[float]]],
        *,
        text_embedding: list[float] | None = None,
        failing_path: Path | None = None,
        model_name: str | None = None,
    ) -> None:
        self.audio_embeddings = audio_embeddings
        self.text_embedding = text_embedding or [1.0, 0.0]
        self.failing_path = failing_path
        self.model_name = model_name
        self.text_calls: list[str] = []
        self.audio_calls: list[tuple[Path, list[TimeRange]]] = []

    async def embed_text(self, text: str) -> list[float]:
        self.text_calls.append(text)
        return self.text_embedding

    async def embed_audio(
        self, path: Path, windows: list[TimeRange]
    ) -> list[list[float]]:
        self.audio_calls.append((path, windows))
        if path == self.failing_path:
            raise RuntimeError("audio embedding unavailable")
        values = self.audio_embeddings[path.name]
        if len(values) == 1:
            return values * len(windows)
        return values


class FixedGroundingModel:
    def __init__(
        self,
        *,
        anchor_scene_id: str = "scene-0002",
        coarse_range: TimeRange | None = None,
        refined_range: TimeRange | None = None,
        failing_stage: str | None = None,
    ) -> None:
        self.anchor_scene_id = anchor_scene_id
        self.coarse_range = coarse_range or TimeRange(start_s=12, end_s=15)
        self.refined_range = refined_range or TimeRange(start_s=12.5, end_s=14.5)
        self.failing_stage = failing_stage
        self.calls: list[
            tuple[str, ShotRequest, GroundingCandidate, list[GroundingFrame]]
        ] = []

    async def locate(
        self,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> GroundingDecision:
        self.calls.append(("locate", shot, candidate, frames))
        if self.failing_stage == "locate":
            raise RuntimeError("grounding model unavailable")
        if candidate.anchor_scene_id != self.anchor_scene_id:
            return GroundingDecision(
                matched=False,
                source_range=None,
                confidence=0.1,
                evidence_frame_ids=[],
                rationale="The requested action is not visible.",
            )
        return GroundingDecision(
            matched=True,
            source_range=self.coarse_range,
            confidence=0.84,
            evidence_frame_ids=[frames[len(frames) // 2].id],
            rationale="The timestamped frames show the requested action.",
        )

    async def refine(
        self,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> GroundingDecision:
        self.calls.append(("refine", shot, candidate, frames))
        if self.failing_stage == "refine":
            raise RuntimeError("grounding refinement unavailable")
        return GroundingDecision(
            matched=True,
            source_range=self.refined_range,
            confidence=0.92,
            evidence_frame_ids=[frames[0].id, frames[-1].id],
            rationale="Dense frames confirm the precise action boundaries.",
        )
