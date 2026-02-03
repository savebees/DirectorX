from __future__ import annotations

from directorx.core.models import (
    GroundingCandidate,
    GroundingDecision,
    GroundingFrame,
    ShotRequest,
    TimeRange,
)


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
