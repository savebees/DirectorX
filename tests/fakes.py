from __future__ import annotations

from directorx.core.models import CandidateScore, Scene, ShotRequest


class FixedGroundingModel:
    async def score(
        self, shot: ShotRequest, candidates: list[Scene]
    ) -> list[CandidateScore]:
        return [
            CandidateScore(
                scene_id=scene.id,
                score=max(0.0, 1.0 - rank * 0.1),
                rationale="fixed test score",
            )
            for rank, scene in enumerate(candidates)
        ]
