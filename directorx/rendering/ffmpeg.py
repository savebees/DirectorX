from __future__ import annotations

import asyncio
from pathlib import Path

from directorx.core.models import RenderPlan


class FFmpegRenderer:
    """Deterministic executor for an already validated render plan."""

    def __init__(
        self,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        video_codec: str = "libx264",
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.video_codec = video_codec

    async def render(self, plan: RenderPlan) -> Path:
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.command(plan)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            tail = stderr.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"FFmpeg render failed:\n{tail}")
        return plan.output_path

    def command(self, plan: RenderPlan) -> list[str]:
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(plan.source_video),
        ]
        for narration in plan.narration:
            args.extend(["-i", str(narration.audio_path)])

        music_input = 1 + len(plan.narration)
        args.extend(["-stream_loop", "-1", "-i", str(plan.sound.track.path)])

        filters: list[str] = []
        video_labels: list[str] = []
        for idx, clip in enumerate(plan.clips):
            label = f"v{idx}"
            source_duration = clip.source_range.duration_s
            pad_duration = max(0.0, clip.target_duration_s - source_duration)
            padding = (
                f",tpad=stop_mode=clone:stop_duration={pad_duration:.6f}"
                if pad_duration > 1e-6
                else ""
            )
            filters.append(
                f"[0:v]trim=start={clip.source_range.start_s:.6f}:"
                f"end={clip.source_range.end_s:.6f},setpts=PTS-STARTPTS,"
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,"
                f"fps={self.fps}{padding},"
                f"trim=duration={clip.target_duration_s:.6f},setpts=PTS-STARTPTS,"
                f"format=yuv420p[{label}]"
            )
            video_labels.append(f"[{label}]")
        filters.append(
            f"{''.join(video_labels)}concat=n={len(video_labels)}:v=1:a=0[vout]"
        )
        if plan.subtitle_path is not None:
            subtitle_file = str(plan.subtitle_path.resolve()).replace("\\", "/")
            subtitle_file = subtitle_file.replace(":", "\\:").replace("'", "\\'")
            filters.append(
                "[vout]subtitles=filename='"
                + subtitle_file
                + "':force_style='FontName=Hiragino Sans GB,FontSize=22,"
                "Outline=2,Shadow=0,Alignment=2,MarginV=55'[vsub]"
            )
            video_map = "[vsub]"
        else:
            video_map = "[vout]"

        narration_inputs = {
            narration.beat_id: (idx, narration)
            for idx, narration in enumerate(plan.narration, start=1)
        }
        narration_labels: list[str] = []
        for beat_index, beat in enumerate(plan.beats):
            label = f"n{beat_index}"
            item = narration_inputs.get(beat.beat_id)
            if item is None:
                filters.append(
                    f"anullsrc=r=48000:cl=mono,atrim=duration={beat.duration_s:.6f},"
                    f"asetpts=PTS-STARTPTS[{label}]"
                )
                narration_labels.append(f"[{label}]")
                continue
            input_index, narration = item
            silence_duration = max(
                0.0,
                beat.duration_s - narration.duration_s,
            )
            padding = (
                f",apad=pad_dur={silence_duration:.6f}"
                if silence_duration > 1e-6
                else ""
            )
            filters.append(
                f"[{input_index}:a]aresample=48000,"
                f"atrim=duration={narration.duration_s:.6f},"
                f"asetpts=PTS-STARTPTS{padding},"
                f"atrim=duration={beat.duration_s:.6f}[{label}]"
            )
            narration_labels.append(f"[{label}]")
        filters.append(
            f"{''.join(narration_labels)}concat=n={len(narration_labels)}:v=0:a=1[voice_raw]"
        )

        filters.append("[voice_raw]asplit=2[voice_mix][voice_key]")
        filters.append(
            f"[{music_input}:a]aresample=48000,atrim=duration={plan.duration_s:.6f},"
            f"asetpts=PTS-STARTPTS,volume={plan.sound.gain_db}dB[music]"
        )
        filters.append(
            "[music][voice_key]sidechaincompress=threshold=0.015:ratio=10:"
            "attack=20:release=500[ducked]"
        )
        filters.append(
            "[voice_mix][ducked]amix=inputs=2:duration=first:normalize=0[aout]"
        )

        args.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                video_map,
                "-map",
                "[aout]",
                "-c:v",
                self.video_codec,
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                "-shortest",
                str(plan.output_path),
            ]
        )
        return args
