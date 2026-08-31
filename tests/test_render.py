from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from directorx.agents import RenderAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
)
from directorx.core.models import (
    GroundedClip,
    GroundingManifest,
    MusicTrack,
    NarrationManifest,
    NarrationSegment,
    RenderPlan,
    SoundPlan,
    TimelineBeat,
    TimeRange,
)
from directorx.rendering.ffmpeg import FFmpegRenderer


class FakeRenderEngine:
    async def render(self, plan):
        plan.output_path.write_bytes(b"fake mp4")
        return plan.output_path


def _inputs(tmp_path: Path, beats: tuple[str, ...] = ("beat-1", "beat-2")):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    music = tmp_path / "music.wav"
    music.write_bytes(b"music")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    segments = []
    clips = []
    for index, beat_id in enumerate(beats):
        audio = audio_dir / f"{beat_id}.wav"
        audio.write_bytes(b"wav")
        segments.append(
            NarrationSegment(
                beat_id=beat_id,
                text=f"Narration {beat_id}",
                audio_path=audio,
                target_duration_s=1,
                duration_s=1,
            )
        )
        clips.append(
            GroundedClip(
                shot_id=f"shot-{index + 1}",
                beat_id=beat_id,
                source_scene_ids=[f"scene-{index + 1}"],
                source_range=TimeRange(start_s=index, end_s=index + 1),
                target_duration_s=1,
                confidence=0.9,
                evidence_frame_ids=[f"frame-{index + 1}"],
                evidence_timestamps_s=[index + 0.5],
                rationale="test",
            )
        )
    grounding_path = tmp_path / "grounding.json"
    grounding_path.write_text(
        GroundingManifest(
            source_video=source,
            clips=clips,
            target_duration_s=len(beats),
            source_duration_s=10,
        ).model_dump_json(),
        encoding="utf-8",
    )
    narration_path = tmp_path / "narration.json"
    narration_path.write_text(
        NarrationManifest(
            segments=segments,
            target_duration_s=len(beats),
            duration_s=len(beats),
        ).model_dump_json(),
        encoding="utf-8",
    )
    sound_path = tmp_path / "sound-plan.json"
    sound_path.write_text(
        SoundPlan(
            track=MusicTrack(path=music, title="Test", duration_s=2),
            target_duration_s=len(beats),
            match_score=0.5,
            selection_rationale="test",
        ).model_dump_json(),
        encoding="utf-8",
    )
    task = TaskContext(
        task_id="render-test",
        assignee=AgentRole.RENDER,
        objective="Render the video",
        input_artifacts=[
            ArtifactRef(name="grounding-manifest", path=grounding_path),
            ArtifactRef(name="narration-manifest", path=narration_path),
            ArtifactRef(name="sound-plan", path=sound_path),
        ],
        expected_output="final.mp4",
        acceptance_criteria=["A playable video exists"],
    )
    return task, source, music, segments, clips


def test_render_agent_publishes_video_atomically(tmp_path: Path) -> None:
    task, *_ = _inputs(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)
    output_dir = tmp_path / "artifacts"

    result = asyncio.run(
        RenderAgent(FakeRenderEngine()).run_task(task, runtime, output_dir)
    )

    assert result.status == "completed"
    assert result.output_artifacts == [
        ArtifactRef(name="rendered-video", path=output_dir / "final.mp4"),
        ArtifactRef(name="subtitles", path=output_dir / "subtitles.srt"),
    ]
    assert (output_dir / "final.mp4").read_bytes() == b"fake mp4"
    assert "Narration beat-1" in (output_dir / "subtitles.srt").read_text()
    assert not list(output_dir.glob(".render.*"))


def test_render_preserves_planned_timing_and_extends_for_long_narration(
    tmp_path: Path,
) -> None:
    task, *_ = _inputs(tmp_path, ("beat-1",))
    narration_path = task.input_artifacts[1].path
    manifest = NarrationManifest.model_validate_json(narration_path.read_text())
    manifest.segments[0].duration_s = 1.4
    manifest.duration_s = 1.4
    narration_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    seen: list[float] = []

    class InspectingEngine(FakeRenderEngine):
        async def render(self, plan):
            seen.append(plan.clips[0].target_duration_s)
            return await super().render(plan)

    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = asyncio.run(
        RenderAgent(InspectingEngine()).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "completed"
    assert seen == [1.4]


def test_render_preserves_longer_visual_target_and_pads_short_narration(
    tmp_path: Path,
) -> None:
    task, *_ = _inputs(tmp_path, ("beat-1",))
    grounding = GroundingManifest.model_validate_json(
        task.input_artifacts[0].path.read_text()
    )
    grounding.clips[0].target_duration_s = 2
    grounding.target_duration_s = 2
    task.input_artifacts[0].path.write_text(
        grounding.model_dump_json(), encoding="utf-8"
    )
    narration = NarrationManifest.model_validate_json(
        task.input_artifacts[1].path.read_text()
    )
    narration.segments[0].target_duration_s = 2
    narration.target_duration_s = 2
    task.input_artifacts[1].path.write_text(
        narration.model_dump_json(), encoding="utf-8"
    )

    plan = RenderAgent._build_plan(
        grounding,
        narration,
        SoundPlan.model_validate_json(task.input_artifacts[2].path.read_text()),
        tmp_path / "final.mp4",
    )
    command = FFmpegRenderer().command(plan)

    assert plan.duration_s == 2
    assert "apad=pad_dur=1.000000" in " ".join(command)
    assert "atrim=duration=2.000000[n0]" in " ".join(command)


def test_render_agent_blocks_mismatched_beats(tmp_path: Path) -> None:
    task, *_ = _inputs(tmp_path, ("beat-1",))
    narration_path = task.input_artifacts[1].path
    manifest = NarrationManifest.model_validate_json(narration_path.read_text())
    manifest.segments.append(
        manifest.segments[0].model_copy(update={"beat_id": "beat-2"})
    )
    manifest.duration_s = 2
    manifest.target_duration_s = 2
    narration_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        RenderAgent(FakeRenderEngine()).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "blocked"
    assert "same beats" in result.summary


def test_render_agent_blocks_existing_output(tmp_path: Path) -> None:
    task, *_ = _inputs(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)
    output_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    (output_dir / "final.mp4").write_bytes(b"keep")

    result = asyncio.run(
        RenderAgent(FakeRenderEngine()).run_task(task, runtime, output_dir)
    )

    assert result.status == "blocked"
    assert (output_dir / "final.mp4").read_bytes() == b"keep"


def test_render_writes_subtitles_on_the_visual_timeline(tmp_path: Path) -> None:
    task, *_ = _inputs(tmp_path)
    grounding = GroundingManifest.model_validate_json(
        task.input_artifacts[0].path.read_text()
    )
    narration = NarrationManifest.model_validate_json(
        task.input_artifacts[1].path.read_text()
    )
    sound = SoundPlan.model_validate_json(task.input_artifacts[2].path.read_text())
    plan = RenderAgent._build_plan(grounding, narration, sound, tmp_path / "final.mp4")
    subtitle_path = tmp_path / "subtitles.srt"

    RenderAgent._write_subtitles(plan, subtitle_path)

    assert subtitle_path.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,000\nNarration beat-1\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nNarration beat-2\n"
    )


def test_render_agent_blocks_missing_audio(tmp_path: Path) -> None:
    task, *_ = _inputs(tmp_path, ("beat-1",))
    narration_path = task.input_artifacts[1].path
    manifest = NarrationManifest.model_validate_json(narration_path.read_text())
    manifest.segments[0].audio_path.unlink()
    narration_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        RenderAgent(FakeRenderEngine()).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "blocked"
    assert "wav" in result.summary


def test_ffmpeg_command_pads_short_source_clips(tmp_path: Path) -> None:
    task, *_ = _inputs(tmp_path, ("beat-1",))
    plan = RenderAgent._build_plan(
        GroundingManifest.model_validate_json(task.input_artifacts[0].path.read_text()),
        NarrationManifest.model_validate_json(task.input_artifacts[1].path.read_text()),
        SoundPlan.model_validate_json(task.input_artifacts[2].path.read_text()),
        tmp_path / "final.mp4",
    )
    plan.clips[0].source_range = TimeRange(start_s=0, end_s=0.25)
    plan.clips[0].target_duration_s = 1
    command = FFmpegRenderer().command(plan)
    joined = " ".join(command)
    assert "tpad=stop_mode=clone" in joined
    assert "trim=duration=1.000000" in joined


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required",
)
def test_ffmpeg_renderer_produces_video_with_target_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    music = tmp_path / "music.wav"
    audio = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:r=10:d=0.5",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    for path, frequency in ((music, 220), (audio, 440)):
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:duration=1",
                "-c:a",
                "pcm_s16le",
                str(path),
            ],
            check=True,
        )
    clip = GroundedClip(
        shot_id="shot-1",
        beat_id="beat-1",
        source_scene_ids=["scene-1"],
        source_range=TimeRange(start_s=0, end_s=0.5),
        target_duration_s=1,
        confidence=1,
        evidence_frame_ids=["frame-1"],
        evidence_timestamps_s=[0.25],
        rationale="test",
    )
    segment = NarrationSegment(
        beat_id="beat-1",
        text="test",
        audio_path=audio,
        target_duration_s=1,
        duration_s=1,
    )
    plan = RenderPlan(
        source_video=source,
        clips=[clip],
        beats=[
            TimelineBeat(
                beat_id="beat-1",
                clip_ids=["shot-1"],
                start_s=0,
                duration_s=1,
                narration_duration_s=1,
            )
        ],
        narration=[segment],
        sound=SoundPlan(
            track=MusicTrack(path=music, title="Test", duration_s=1),
            target_duration_s=1,
            match_score=0,
            selection_rationale="test",
        ),
        output_path=tmp_path / "final.mp4",
        target_duration_s=1,
    )
    asyncio.run(FFmpegRenderer(width=320, height=240, fps=10).render(plan))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(plan.output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(1, abs=0.15)
