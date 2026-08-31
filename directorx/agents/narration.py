from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import NarrationManifest, NarrationSegment, Storyboard
from directorx.core.ports import TextToSpeech


class NarrationAgent:
    role = AgentRole.NARRATION

    def __init__(
        self,
        tts: TextToSpeech,
        max_parallel: int = 4,
        min_voice_coverage: float = 0.0,
        max_voice_coverage: float = float("inf"),
        breathing_room_s: float = 0.35,
    ) -> None:
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if min_voice_coverage < 0 or max_voice_coverage < min_voice_coverage:
            raise ValueError("Narration coverage bounds are invalid")
        if breathing_room_s < 0:
            raise ValueError("Narration breathing room must be non-negative")
        self.tts = tts
        self.max_parallel = max_parallel
        self.min_voice_coverage = min_voice_coverage
        self.max_voice_coverage = max_voice_coverage
        self.breathing_room_s = breathing_room_s

    async def run(
        self, storyboard: Storyboard, audio_dir: Path
    ) -> list[NarrationSegment]:
        storyboard = Storyboard.model_validate(storyboard)
        if not storyboard.beats:
            raise ValueError("Narration requires at least one storyboard beat")
        beat_ids = [beat.id for beat in storyboard.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be unique")

        audio_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def synthesize(beat) -> NarrationSegment:
            if not beat.narration.strip():
                raise ValueError(f"Storyboard beat {beat.id} has empty narration")
            path = audio_dir / self._audio_filename(beat.id)
            async with semaphore:
                duration = await self.tts.synthesize(beat.narration, path)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"TTS returned an invalid duration for {beat.id}")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"TTS did not write usable audio for {beat.id}")
            return NarrationSegment(
                beat_id=beat.id,
                text=beat.narration,
                audio_path=path,
                target_duration_s=beat.target_duration_s,
                duration_s=duration,
            )

        results = await asyncio.gather(
            *(synthesize(beat) for beat in storyboard.beats),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                raise result
        return [result for result in results if isinstance(result, NarrationSegment)]

    async def run_task(
        self,
        task: TaskContext,
        runtime: CoordinationRuntime,
        artifacts_dir: Path,
    ) -> TaskResult:
        """Synthesize narration from the declared storyboard artifact."""
        if task.assignee != self.role:
            raise ValueError("Narration can only execute its own tasks")

        staging_dir: Path | None = None
        try:
            storyboard = self._load_storyboard(task)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            destination = artifacts_dir / "narration"
            if destination.exists():
                raise FileExistsError(destination)
            staging_dir = Path(
                tempfile.mkdtemp(dir=artifacts_dir, prefix=".narration.")
            )
            staging_segments = await self.run(storyboard, staging_dir)
            staging_segments = await asyncio.to_thread(
                self._normalize_coverage,
                staging_segments,
                storyboard.target_duration_s,
            )
            staging_segments = await asyncio.to_thread(
                self._fit_segment_windows,
                staging_segments,
            )
            segments = [
                segment.model_copy(
                    update={
                        "audio_path": destination / segment.audio_path.name,
                    }
                )
                for segment in staging_segments
            ]
            manifest = NarrationManifest(
                segments=segments,
                target_duration_s=storyboard.target_duration_s,
                duration_s=sum(segment.duration_s for segment in segments),
            )
            coverage = manifest.duration_s / manifest.target_duration_s
            if coverage < self.min_voice_coverage:
                raise ValueError(
                    f"Voice coverage {coverage:.1%} is below "
                    f"{self.min_voice_coverage:.1%}"
                )
            if coverage > self.max_voice_coverage:
                raise ValueError(
                    f"Voice coverage {coverage:.1%} exceeds "
                    f"{self.max_voice_coverage:.1%}"
                )
            self._write_manifest(manifest, staging_dir / "narration.json")
            os.replace(staging_dir, destination)
            staging_dir = None
            manifest_path = destination / "narration.json"
        except Exception as exc:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            result = TaskResult(
                task_id=task.task_id,
                agent=self.role,
                status="blocked",
                summary=f"Narration blocked: {exc}",
            )
            runtime.submit_result(self.role, result)
            return result

        result = TaskResult(
            task_id=task.task_id,
            agent=self.role,
            status="completed",
            summary=(
                f"Synthesized {len(manifest.segments)} narration segments: "
                f"{manifest.duration_s:.1f}s actual for "
                f"{manifest.target_duration_s:.1f}s target."
            ),
            output_artifacts=[
                ArtifactRef(name="narration-manifest", path=manifest_path)
            ],
        )
        runtime.submit_result(self.role, result)
        return result

    @staticmethod
    def _load_storyboard(task: TaskContext) -> Storyboard:
        references = [
            artifact
            for artifact in task.input_artifacts
            if artifact.name == "storyboard"
        ]
        if len(references) != 1 or references[0].path.name != "storyboard.json":
            raise ValueError(
                "Narration task must declare exactly one storyboard "
                "storyboard.json input artifact"
            )
        return Storyboard.model_validate_json(
            references[0].path.read_text(encoding="utf-8")
        )

    @staticmethod
    def _audio_filename(beat_id: str) -> str:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", beat_id) is None:
            raise ValueError(f"Storyboard beat id is not a safe filename: {beat_id}")
        return f"{beat_id}.wav"

    @staticmethod
    def _write_manifest(manifest: NarrationManifest, path: Path) -> None:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(manifest.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _normalize_coverage(
        self,
        segments: list[NarrationSegment],
        target_duration_s: float,
    ) -> list[NarrationSegment]:
        duration_s = sum(segment.duration_s for segment in segments)
        coverage = duration_s / target_duration_s
        if self.min_voice_coverage <= coverage <= self.max_voice_coverage:
            return segments
        desired_coverage = (
            self.min_voice_coverage
            if coverage < self.min_voice_coverage
            else self.max_voice_coverage
        )
        tempo = coverage / desired_coverage
        if not 0.85 <= tempo <= 1.30:
            raise ValueError(
                f"Voice coverage {coverage:.1%} requires an unsafe "
                f"{tempo:.2f}x tempo adjustment"
            )
        normalized = []
        for segment in segments:
            temporary = segment.audio_path.with_suffix(".retimed.wav")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(segment.audio_path),
                        "-filter:a",
                        f"atempo={tempo:.6f}",
                        str(temporary),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                os.replace(temporary, segment.audio_path)
            finally:
                temporary.unlink(missing_ok=True)
            normalized.append(
                segment.model_copy(
                    update={
                        "duration_s": self._probe_duration(segment.audio_path),
                    }
                )
            )
        return normalized

    def _fit_segment_windows(
        self, segments: list[NarrationSegment]
    ) -> list[NarrationSegment]:
        if not math.isfinite(self.max_voice_coverage):
            return segments
        fitted = []
        for segment in segments:
            maximum_s = max(0.1, segment.target_duration_s - self.breathing_room_s)
            if segment.duration_s <= maximum_s + 1e-6:
                fitted.append(segment)
                continue
            tempo = segment.duration_s / maximum_s
            if tempo > 1.30:
                raise ValueError(
                    f"Narration segment {segment.beat_id} requires an unsafe "
                    f"{tempo:.2f}x tempo adjustment"
                )
            temporary = segment.audio_path.with_suffix(".fitted.wav")
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(segment.audio_path),
                        "-filter:a",
                        f"atempo={tempo:.6f}",
                        str(temporary),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                os.replace(temporary, segment.audio_path)
            finally:
                temporary.unlink(missing_ok=True)
            fitted.append(
                segment.model_copy(
                    update={"duration_s": self._probe_duration(segment.audio_path)}
                )
            )
        return fitted

    @staticmethod
    def _probe_duration(path: Path) -> float:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
