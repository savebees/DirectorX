from __future__ import annotations

import asyncio
import base64
import os
import re
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from directorx.core.models import (
    GroundingCandidate,
    GroundingDecision,
    GroundingFrame,
    ShotRequest,
    TimeRange,
)
from directorx.services.structured_output import (
    StructuredOutputMode,
    request_structured_output,
)


class FFmpegGroundingFrameExtractor:
    """Extract exact, timestamp-labeled frames from a candidate source window."""

    def __init__(self, max_image_dimension: int = 1024) -> None:
        if max_image_dimension <= 0:
            raise ValueError("max_image_dimension must be positive")
        self.max_image_dimension = max_image_dimension

    async def extract(
        self,
        video_path: Path,
        source_range: TimeRange,
        output_dir: Path,
        *,
        fps: float,
        max_frames: int,
        prefix: str,
    ) -> list[GroundingFrame]:
        if fps <= 0:
            raise ValueError("Grounding frame rate must be positive")
        if max_frames <= 0:
            raise ValueError("Grounding max_frames must be positive")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", prefix) is None:
            raise ValueError(f"Unsafe grounding frame prefix: {prefix}")
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return await asyncio.to_thread(
            self._extract_sync,
            video_path,
            source_range,
            output_dir,
            fps,
            max_frames,
            prefix,
        )

    def _extract_sync(
        self,
        video_path: Path,
        source_range: TimeRange,
        output_dir: Path,
        fps: float,
        max_frames: int,
        prefix: str,
    ) -> list[GroundingFrame]:
        frames: list[GroundingFrame] = []
        for number, timestamp_s in enumerate(
            self._timestamps(source_range, fps, max_frames), start=1
        ):
            frame_id = f"{prefix}-{number:04d}"
            path = output_dir / f"{frame_id}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp_s:.6f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-n",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg did not write grounding frame {frame_id}")
            self._label_frame(path, frame_id, timestamp_s)
            frames.append(
                GroundingFrame(id=frame_id, timestamp_s=timestamp_s, path=path)
            )
        if not frames:
            raise ValueError("Grounding frame extraction produced no frames")
        return frames

    @staticmethod
    def _timestamps(
        source_range: TimeRange, fps: float, max_frames: int
    ) -> list[float]:
        step_s = 1.0 / fps
        timestamps = []
        timestamp_s = source_range.start_s
        while timestamp_s < source_range.end_s:
            timestamps.append(timestamp_s)
            timestamp_s += step_s
        if not timestamps:
            timestamps = [
                source_range.start_s + source_range.duration_s / 2,
            ]
        if len(timestamps) <= max_frames:
            return timestamps
        if max_frames == 1:
            return [timestamps[len(timestamps) // 2]]
        indexes = {
            round(index * (len(timestamps) - 1) / (max_frames - 1))
            for index in range(max_frames)
        }
        return [timestamps[index] for index in sorted(indexes)]

    def _label_frame(self, path: Path, frame_id: str, timestamp_s: float) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("Pillow is required to label grounding frames") from exc

        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(
                (self.max_image_dimension, self.max_image_dimension),
                Image.Resampling.LANCZOS,
            )
            draw = ImageDraw.Draw(image)
            label = f"{frame_id}  {timestamp_s:.3f}s"
            box = draw.textbbox((8, 8), label)
            draw.rectangle(
                (4, 4, box[2] + 12, box[3] + 12),
                fill=(0, 0, 0),
            )
            draw.text((8, 8), label, fill=(255, 255, 255))
            image.save(path, format="JPEG", quality=90, optimize=True)


class OpenAICompatibleGroundingModel:
    """Locate and refine source intervals with timestamped visual evidence."""

    BOUNDARY_ROUNDING_TOLERANCE_S = 0.001

    LOCATE_SYSTEM_PROMPT = (
        "You are a professional footage grounding editor. Your task is to identify "
        "the exact source-video interval that best realizes the requested visual "
        "intent. Use only the supplied timestamped visual evidence and return the "
        "strongest supported typed grounding decision."
    )
    REFINE_SYSTEM_PROMPT = (
        "You are a professional footage grounding editor. Refine the start and end "
        "boundaries of the candidate moment using the supplied dense timestamped "
        "frames. Return the same typed grounding decision."
    )

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        base_url: str = "https://api.siliconflow.cn/v1",
        api_key_env: str = "VLM_API_KEY",
        api_key: str | None = None,
        max_tokens: int = 1200,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        request_interval_s: float = 0.0,
        structured_output_mode: StructuredOutputMode = "json_object",
        client: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if request_interval_s < 0:
            raise ValueError("request_interval_s must be non-negative")
        self.model = model
        self.max_tokens = max_tokens
        self.request_interval_s = request_interval_s
        self.structured_output_mode = structured_output_mode
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0
        if client is not None:
            self._client = client
            return
        secret = api_key or os.environ.get(api_key_env, "")
        if not secret:
            raise RuntimeError(
                f"Missing Grounding VLM API key. Set {api_key_env} in the "
                "process environment."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for Grounding VLM calls"
            ) from exc
        self._client = AsyncOpenAI(
            api_key=secret,
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            max_retries=max_retries,
            default_headers={"User-Agent": "directorx/0.1"},
        )

    async def locate(
        self,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> GroundingDecision:
        return await self._complete(
            self.LOCATE_SYSTEM_PROMPT,
            shot,
            candidate,
            frames,
            "grounding_location",
        )

    async def refine(
        self,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> GroundingDecision:
        return await self._complete(
            self.REFINE_SYSTEM_PROMPT,
            shot,
            candidate,
            frames,
            "grounding_refinement",
        )

    async def _complete(
        self,
        system_prompt: str,
        shot: ShotRequest,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
        schema_name: str,
    ) -> GroundingDecision:
        if not frames:
            raise ValueError("Grounding VLM requires timestamped frames")
        frame_index = ", ".join(
            f"{frame.id}={frame.timestamp_s:.6f}s" for frame in frames
        )
        proposal_context = (
            f"Coarse proposal range: {candidate.proposal_range.start_s:.6f}s to "
            f"{candidate.proposal_range.end_s:.6f}s\n"
            if candidate.proposal_range is not None
            else ""
        )
        request = (
            f"Visual intent: {shot.visual_query}\n"
            f"Story content: {shot.story_content}\n"
            f"Narration: {shot.narration_text}\n"
            f"Mood: {shot.mood}\n"
            f"Candidate anchor scene: {candidate.anchor_scene_id}\n"
            f"Candidate scene IDs: {', '.join(candidate.scene_ids)}\n"
            f"Candidate source range: {candidate.source_range.start_s:.6f}s to "
            f"{candidate.source_range.end_s:.6f}s\n"
            f"{proposal_context}"
            f"Ordered frame index: {frame_index}\n\n"
            "When the visual evidence does not support the intent, set matched to "
            "false, source_range to null, and evidence_frame_ids to an empty list. "
            "When matched, choose source_range.start_s and source_range.end_s within "
            "the candidate range and cite the supporting frame IDs. Populate every "
            "required decision property."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": request}]
        for frame in frames:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"{frame.id} at {frame.timestamp_s:.3f}s",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": self._data_url(frame.path)},
                    },
                ]
            )
        async with self._request_lock:
            now = time.monotonic()
            wait_s = self.request_interval_s - (now - self._last_request_at)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_request_at = time.monotonic()
        decision = await request_structured_output(
            self._client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            schema=GroundingDecision,
            schema_name=schema_name,
            max_tokens=self.max_tokens,
            temperature=0.1,
            mode=self.structured_output_mode,
            validation_retries=1,
            validate=lambda decision: self._validate_decision(
                decision, candidate, frames
            ),
        )
        return self._clamp_boundary_rounding(decision, candidate)

    @staticmethod
    def _validate_decision(
        decision: GroundingDecision,
        candidate: GroundingCandidate,
        frames: list[GroundingFrame],
    ) -> None:
        if not decision.matched:
            return
        source_range = decision.source_range
        if source_range is None:
            raise ValueError("Matched grounding requires a source range")
        tolerance = OpenAICompatibleGroundingModel.BOUNDARY_ROUNDING_TOLERANCE_S
        if (
            source_range.start_s < candidate.source_range.start_s - tolerance
            or source_range.end_s > candidate.source_range.end_s + tolerance
        ):
            raise ValueError(
                "Grounding source range must stay inside the candidate; "
                f"received {source_range.start_s:.6f}..{source_range.end_s:.6f}, "
                f"candidate {candidate.source_range.start_s:.6f}.."
                f"{candidate.source_range.end_s:.6f}"
            )
        frame_ids = {frame.id for frame in frames}
        unknown = [
            frame_id
            for frame_id in decision.evidence_frame_ids
            if frame_id not in frame_ids
        ]
        if unknown:
            raise ValueError(
                "Grounding cited unknown evidence frame IDs: " + ", ".join(unknown)
            )

    @staticmethod
    def _clamp_boundary_rounding(
        decision: GroundingDecision,
        candidate: GroundingCandidate,
    ) -> GroundingDecision:
        """Clamp only sub-millisecond serialization drift to the true candidate."""
        if not decision.matched or decision.source_range is None:
            return decision
        source_range = decision.source_range
        clamped = TimeRange(
            start_s=max(source_range.start_s, candidate.source_range.start_s),
            end_s=min(source_range.end_s, candidate.source_range.end_s),
        )
        return decision.model_copy(update={"source_range": clamped})

    @staticmethod
    def _data_url(path: Path) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for Grounding VLM frames") from exc
        with Image.open(path) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
