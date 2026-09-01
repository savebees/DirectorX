from pathlib import Path

import pytest

from directorx.agents import EditorAgent
from directorx.core.models import (
    GroundedClip,
    GroundingManifest,
    MusicTrack,
    NarrationManifest,
    NarrationSegment,
    SoundPlan,
    StoryBeat,
    Storyboard,
    TimeRange,
)


def _inputs(tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    music = tmp_path / "music.wav"
    music.write_bytes(b"music")
    beat_durations = [17.5, 17.5, 17.5, 7.5]
    voice_durations = [10.75, 10.75, 10.75, 7.0]
    beats = []
    narration_segments = []
    clips = []
    source_cursor = 0.0
    for index, (beat_duration, voice_duration) in enumerate(
        zip(beat_durations, voice_durations, strict=True),
        start=1,
    ):
        beat_id = f"beat-{index:02d}"
        beats.append(
            StoryBeat(
                id=beat_id,
                purpose="Advance the story",
                story_content=f"Story passage {index}",
                narration=f"连续叙事第{index}段。",
                visual_intent=f"Visible event {index}",
                mood="tense",
                target_duration_s=beat_duration,
                source_sequence_ids=[f"sequence-{index:04d}"],
                evidence_scene_ids=[f"scene-{index:04d}"],
            )
        )
        audio = tmp_path / f"{beat_id}.wav"
        audio.write_bytes(b"wav")
        narration_segments.append(
            NarrationSegment(
                beat_id=beat_id,
                text=f"连续叙事第{index}段。",
                audio_path=audio,
                target_duration_s=beat_duration,
                duration_s=voice_duration,
            )
        )
        clip_count = 2 if index == 1 else 1
        clip_duration = beat_duration / clip_count
        for shot in range(clip_count):
            clips.append(
                GroundedClip(
                    shot_id=f"shot-{index:02d}-{shot + 1:02d}",
                    beat_id=beat_id,
                    source_scene_ids=[f"scene-{index:04d}"],
                    source_range=TimeRange(
                        start_s=source_cursor,
                        end_s=source_cursor + clip_duration,
                    ),
                    target_duration_s=clip_duration,
                    confidence=0.9,
                    evidence_frame_ids=[f"frame-{index}-{shot}"],
                    evidence_timestamps_s=[source_cursor + 0.5],
                    rationale="Visually verified.",
                )
            )
            source_cursor += clip_duration
    full_narration = "".join(beat.narration for beat in beats)
    storyboard = Storyboard(
        title="Test",
        logline="A continuous story.",
        narrative_angle="Escalation and payoff.",
        full_narration=full_narration,
        beats=beats,
        target_duration_s=60,
    )
    narration = NarrationManifest(
        segments=narration_segments,
        target_duration_s=60,
        duration_s=sum(voice_durations),
    )
    grounding = GroundingManifest(
        source_video=source,
        clips=clips,
        target_duration_s=60,
        source_duration_s=source_cursor,
    )
    sound = SoundPlan(
        track=MusicTrack(path=music, title="Suspense", duration_s=60),
        target_duration_s=60,
        match_score=0.8,
        selection_rationale="Matched.",
    )
    return storyboard, grounding, narration, sound


def test_editor_builds_exact_multi_shot_timeline(tmp_path: Path) -> None:
    storyboard, grounding, narration, sound = _inputs(tmp_path)

    timeline = EditorAgent().run(storyboard, grounding, narration, sound)

    assert timeline.duration_s == pytest.approx(60)
    assert timeline.voice_coverage == pytest.approx(39.25 / 60)
    assert timeline.beats[0].clip_ids == ["shot-01-01", "shot-01-02"]
    assert [beat.start_s for beat in timeline.beats] == [0, 17.5, 35, 52.5]
    assert timeline.beats[-1].duration_s <= 8
    assert all(
        clip.target_duration_s - clip.source_range.duration_s <= 0.3 + 1e-6
        for clip in timeline.clips
    )


def test_editor_blocks_short_narration(tmp_path: Path) -> None:
    storyboard, grounding, narration, sound = _inputs(tmp_path)
    narration.duration_s = 20

    with pytest.raises(ValueError, match="coverage"):
        EditorAgent().run(storyboard, grounding, narration, sound)


def test_editor_preserves_a_silent_visual_beat(tmp_path: Path) -> None:
    storyboard, grounding, narration, sound = _inputs(tmp_path)
    storyboard.beats[-1].narration = ""
    removed = narration.segments.pop()
    narration.duration_s -= removed.duration_s

    timeline = EditorAgent(min_voice_coverage=0).run(
        storyboard, grounding, narration, sound
    )

    assert timeline.beats[-1].beat_id == "beat-04"
    assert timeline.beats[-1].narration_duration_s == 0
    assert timeline.duration_s == pytest.approx(60)


def test_editor_blocks_when_verified_footage_cannot_cover_voice(
    tmp_path: Path,
) -> None:
    storyboard, grounding, narration, sound = _inputs(tmp_path)
    grounding.clips[0].source_range = TimeRange(start_s=0, end_s=1)

    with pytest.raises(ValueError, match="verified footage"):
        EditorAgent().run(storyboard, grounding, narration, sound)
