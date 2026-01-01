from __future__ import annotations

from directorx.core.models import Storyboard, VideoIndex
from directorx.core.ports import ScreenwriterModel


class ScreenwriterAgent:
    def __init__(self, model: ScreenwriterModel) -> None:
        self.model = model

    async def run(
        self, prompt: str, index: VideoIndex, target_duration_s: float
    ) -> Storyboard:
        storyboard = await self.model.draft(prompt, index, target_duration_s)
        if not storyboard.beats:
            raise ValueError("Screenwriter agent returned no beats")
        beat_ids = [beat.id for beat in storyboard.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be unique")
        return storyboard
