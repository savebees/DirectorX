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
from directorx.core.models import (
    NarrationDelivery,
    NarrationManifest,
    NarrationSegment,
    Storyboard,
    VoiceProfile,
)
from directorx.core.ports import TextToSpeech


class NarrationAgent:
    role = AgentRole.NARRATION
    _THEME_KEYWORDS = {
        "dark": (
            "dark",
            "noir",
            "cold",
            "grim",
            "黑暗",
            "暗黑",
            "冷峻",
            "冷酷",
            "阴冷",
            "阴郁",
            "宿命",
            "压迫",
            "黑色电影",
        ),
        "serious": ("serious", "solemn", "grave", "严肃", "庄重", "沉重"),
        "mysterious": ("mysterious", "mystery", "suspense", "悬疑", "神秘"),
        "professional": ("professional", "mission", "agent", "任务", "特工"),
        "reliable": ("reliable", "documentary", "纪实", "可信"),
        "action": ("action", "fight", "combat", "chase", "动作", "搏斗", "追逐"),
        "intense": ("intense", "tense", "violent", "紧张", "激烈", "暴力"),
        "epic": ("epic", "heroic", "史诗", "英雄"),
        "passionate": ("passionate", "fiery", "热血", "激情"),
        "warm": ("warm", "tender", "温暖", "温情"),
        "emotional": ("emotional", "sad", "love", "情感", "悲伤", "爱情"),
        "gentle": ("gentle", "quiet", "calm", "温柔", "平静"),
        "youthful": ("youthful", "young", "青春", "少年"),
        "adventure": ("adventure", "journey", "冒险", "旅程"),
        "hopeful": ("hopeful", "hope", "希望", "治愈"),
        "lively": ("lively", "energetic", "活泼", "轻快"),
        "comedy": ("comedy", "funny", "喜剧", "搞笑"),
        "bright": ("bright", "cheerful", "明亮", "欢快"),
        "playful": ("playful", "cute", "俏皮", "可爱"),
    }

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
        segments, _ = await self._synthesize_with_delivery(storyboard, audio_dir)
        return segments

    async def _synthesize_with_delivery(
        self, storyboard: Storyboard, audio_dir: Path
    ) -> tuple[list[NarrationSegment], NarrationDelivery | None]:
        storyboard = Storyboard.model_validate(storyboard)
        if not storyboard.beats:
            raise ValueError("Narration requires at least one storyboard beat")
        beat_ids = [beat.id for beat in storyboard.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Storyboard beat ids must be unique")

        audio_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_parallel)
        delivery = self._select_delivery(storyboard)
        tts = self.tts
        if delivery is not None:
            configure = getattr(self.tts, "configure", None)
            if not callable(configure):
                raise TypeError("TTS exposes voice profiles but cannot configure one")
            tts = configure(delivery)

        voiced_beats = [beat for beat in storyboard.beats if beat.narration.strip()]
        if not voiced_beats:
            raise ValueError("Narration requires at least one spoken storyboard beat")

        async def synthesize(beat) -> NarrationSegment:
            path = audio_dir / self._audio_filename(beat.id)
            async with semaphore:
                duration = await tts.synthesize(beat.narration, path)
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
            *(synthesize(beat) for beat in voiced_beats),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                raise result
        return (
            [result for result in results if isinstance(result, NarrationSegment)],
            delivery,
        )

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
            staging_segments, delivery = await self._synthesize_with_delivery(
                storyboard, staging_dir
            )
            staging_segments = await asyncio.to_thread(
                self._normalize_coverage,
                staging_segments,
                storyboard.target_duration_s,
            )
            staging_segments = await asyncio.to_thread(
                self._fit_segment_windows,
                staging_segments,
            )
            staging_segments = await asyncio.to_thread(
                self._restore_min_coverage,
                staging_segments,
                storyboard.target_duration_s,
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
                delivery=delivery,
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
                f"{manifest.target_duration_s:.1f}s target"
                + (
                    f" using automatically selected {delivery.display_name} "
                    f"({delivery.voice_id})."
                    if delivery is not None
                    else "."
                )
            ),
            output_artifacts=[
                ArtifactRef(name="narration-manifest", path=manifest_path)
            ],
        )
        runtime.submit_result(self.role, result)
        return result

    def _select_delivery(self, storyboard: Storyboard) -> NarrationDelivery | None:
        profiles_method = getattr(self.tts, "voice_profiles", None)
        if not callable(profiles_method):
            return None
        profiles = [VoiceProfile.model_validate(item) for item in profiles_method()]
        if not profiles:
            raise ValueError("TTS returned no narration voice profiles")
        return self._choose_delivery(storyboard, profiles)

    @classmethod
    def _choose_delivery(
        cls, storyboard: Storyboard, profiles: list[VoiceProfile]
    ) -> NarrationDelivery:
        storyboard = Storyboard.model_validate(storyboard)
        if not profiles:
            raise ValueError("Narration voice selection requires voice profiles")
        weighted_fields = [
            (storyboard.title, 2.0),
            (storyboard.logline, 2.0),
            (storyboard.narrative_angle, 3.0),
            *((beat.mood, 2.0) for beat in storyboard.beats),
            *((beat.purpose, 1.0) for beat in storyboard.beats),
            *((beat.story_content, 0.5) for beat in storyboard.beats),
            *((beat.visual_intent, 0.25) for beat in storyboard.beats),
        ]
        theme_scores = {
            theme: sum(
                weight
                for value, weight in weighted_fields
                if any(keyword.casefold() in value.casefold() for keyword in keywords)
            )
            for theme, keywords in cls._THEME_KEYWORDS.items()
        }
        scored = [
            (
                sum(theme_scores.get(trait.casefold(), 0) for trait in profile.traits),
                index,
                profile,
            )
            for index, profile in enumerate(profiles)
        ]
        score, _, selected = max(scored, key=lambda item: (item[0], -item[1]))
        if score == 0:
            selected = next(
                (
                    profile
                    for profile in profiles
                    if "neutral" in {trait.casefold() for trait in profile.traits}
                ),
                profiles[0],
            )
        matched = sorted(
            (
                trait
                for trait in selected.traits
                if theme_scores.get(trait.casefold(), 0) > 0
            ),
            key=lambda trait: (-theme_scores[trait.casefold()], trait),
        )
        matched_set = {trait.casefold() for trait in matched}
        rate = selected.base_rate
        pitch_hz = selected.base_pitch_hz
        if matched_set.intersection({"dark", "serious", "mysterious"}):
            rate -= 5
            pitch_hz -= 2
        elif matched_set.intersection({"action", "intense", "comedy", "lively"}):
            rate += 5
        elif matched_set.intersection({"warm", "emotional", "gentle"}):
            rate -= 3
        rate = max(140, min(205, rate))
        pitch_hz = max(-100, min(100, pitch_hz))
        rationale = (
            f"Selected {selected.display_name} from {len(profiles)} available voices; "
            + (
                "matched storyboard traits: " + ", ".join(matched)
                if matched
                else "used the neutral narrative profile"
            )
        )
        return NarrationDelivery(
            voice_id=selected.voice_id,
            display_name=selected.display_name,
            rate=rate,
            pitch_hz=pitch_hz,
            volume_percent=selected.volume_percent,
            matched_traits=matched,
            rationale=rationale,
        )

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

    def _restore_min_coverage(
        self,
        segments: list[NarrationSegment],
        target_duration_s: float,
    ) -> list[NarrationSegment]:
        """Recover coverage lost while fitting an individual beat window."""
        current_s = sum(segment.duration_s for segment in segments)
        minimum_s = self.min_voice_coverage * target_duration_s
        if current_s >= minimum_s - 1e-6:
            return segments

        maximum_s = self.max_voice_coverage * target_duration_s
        # FFmpeg's tempo filter can trim a few encoder frames, so retain a
        # small audible-safe margin instead of targeting the threshold exactly.
        desired_s = min(minimum_s + 0.5, maximum_s)
        capacities = []
        for segment in segments:
            window_s = max(0.1, segment.target_duration_s - self.breathing_room_s)
            safe_s = segment.duration_s / 0.85
            capacities.append(max(0.0, min(window_s, safe_s) - segment.duration_s))
        available_s = sum(capacities)
        required_s = desired_s - current_s
        if available_s + 1e-6 < minimum_s - current_s:
            raise ValueError(
                "Narration cannot meet minimum coverage within safe beat windows"
            )
        required_s = min(required_s, available_s)

        restored: list[NarrationSegment] = []
        remaining_s = required_s
        remaining_capacity_s = available_s
        for segment, capacity_s in zip(segments, capacities, strict=True):
            growth_s = (
                remaining_s * capacity_s / remaining_capacity_s
                if remaining_capacity_s > 1e-9
                else 0.0
            )
            growth_s = min(growth_s, capacity_s)
            remaining_s -= growth_s
            remaining_capacity_s -= capacity_s
            if growth_s <= 1e-6:
                restored.append(segment)
                continue
            temporary = segment.audio_path.with_suffix(".coverage.wav")
            try:
                tempo = segment.duration_s / (segment.duration_s + growth_s)
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
            restored.append(
                segment.model_copy(
                    update={"duration_s": self._probe_duration(segment.audio_path)}
                )
            )
        if sum(segment.duration_s for segment in restored) < minimum_s - 0.01:
            raise ValueError("Narration remained below minimum coverage after fitting")
        return restored

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
