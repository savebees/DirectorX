from __future__ import annotations

from collections import Counter

from directorx.core.models import MusicTrack, SoundPlan, Storyboard


class SoundAgent:
    async def run(self, storyboard: Storyboard, tracks: list[MusicTrack]) -> SoundPlan:
        if not tracks:
            raise ValueError("Music library returned no tracks")
        desired_moods = Counter(beat.mood for beat in storyboard.beats)

        def score(track: MusicTrack) -> tuple[int, float]:
            return sum(desired_moods[tag] for tag in track.tags), track.duration_s

        return SoundPlan(track=max(tracks, key=score))
