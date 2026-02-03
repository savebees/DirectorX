from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
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

    def __init__(self, tts: TextToSpeech, max_parallel: int = 4) -> None:
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        self.tts = tts
        self.max_parallel = max_parallel

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
