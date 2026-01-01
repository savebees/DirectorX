from __future__ import annotations

import asyncio
from pathlib import Path

from directorx.core.models import NarrationSegment, Storyboard
from directorx.core.ports import TextToSpeech


class NarrationAgent:
    def __init__(self, tts: TextToSpeech, max_parallel: int = 4) -> None:
        self.tts = tts
        self.max_parallel = max_parallel

    async def run(
        self, storyboard: Storyboard, audio_dir: Path
    ) -> list[NarrationSegment]:
        audio_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def synthesize(beat_id: str, text: str) -> NarrationSegment:
            path = audio_dir / f"{beat_id}.wav"
            async with semaphore:
                duration = await self.tts.synthesize(text, path)
            return NarrationSegment(
                beat_id=beat_id,
                text=text,
                audio_path=path,
                duration_s=duration,
            )

        return list(
            await asyncio.gather(
                *(synthesize(beat.id, beat.narration) for beat in storyboard.beats)
            )
        )
