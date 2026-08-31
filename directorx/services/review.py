from __future__ import annotations

import asyncio
import base64
import os
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any

from directorx.core.models import GroundingFrame, ReviewReport
from directorx.services.structured_output import (
    StructuredOutputMode,
    request_structured_output,
)


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError("Rendered video duration must be positive")
    return duration


class FFmpegReviewFrameExtractor:
    """Extract a small, evenly spaced visual sample from the rendered video."""

    async def extract(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        max_frames: int,
    ) -> list[GroundingFrame]:
        if max_frames <= 0:
            raise ValueError("Review max_frames must be positive")
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        duration = await asyncio.to_thread(_probe_duration, video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamps = [
            duration / 2 if max_frames == 1 else duration * index / max_frames
            for index in range(max_frames)
        ]
        return await asyncio.to_thread(
            self._extract_sync, video_path, output_dir, timestamps
        )

    @staticmethod
    def _extract_sync(
        video_path: Path, output_dir: Path, timestamps: list[float]
    ) -> list[GroundingFrame]:
        frames: list[GroundingFrame] = []
        for number, timestamp_s in enumerate(timestamps, start=1):
            frame_id = f"review-{number:04d}"
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
                    "-y",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"FFmpeg did not write review frame {frame_id}")
            frames.append(
                GroundingFrame(id=frame_id, timestamp_s=timestamp_s, path=path)
            )
        return frames


class OpenAICompatibleReviewModel:
    """Review a timestamped visual sample with one multimodal request."""

    SYSTEM_PROMPT = (
        "You are a professional video editor reviewing a finished video. "
        "Check whether the video is complete and narratively coherent, and "
        "whether it contains obvious visual errors, broken cuts, black frames, "
        "frozen frames, malformed crops, or corrupted overlays. Do not critique "
        "creative taste. Approve the video when there are no clear defects. Return "
        "one typed review report."
    )

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        base_url: str = "https://api.siliconflow.cn/v1",
        api_key_env: str = "VLM_API_KEY",
        max_tokens: int = 800,
        timeout_s: float = 180.0,
        max_retries: int = 3,
        structured_output_mode: StructuredOutputMode = "json_object",
        client: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.max_tokens = max_tokens
        self.structured_output_mode = structured_output_mode
        if client is not None:
            self._client = client
            return
        secret = os.environ.get(api_key_env, "")
        if not secret:
            raise RuntimeError(f"Missing Review VLM API key. Set {api_key_env}")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for Review VLM calls"
            ) from exc
        self._client = AsyncOpenAI(
            api_key=secret,
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            max_retries=max_retries,
        )

    async def inspect(
        self, duration_s: float, frames: list[GroundingFrame]
    ) -> ReviewReport:
        if not frames:
            raise ValueError("Review requires at least one video frame")
        frame_index = ", ".join(
            f"{frame.id}={frame.timestamp_s:.3f}s" for frame in frames
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Finished video duration: {duration_s:.2f}s\n"
                    f"Frame index: {frame_index}\n\n"
                    "Review the sampled frames as one finished edit. Decide whether "
                    "the narrative appears complete and flows naturally from start "
                    "to finish. Report only clear errors. Return passed=true and an "
                    "empty issues list when the video is acceptable."
                ),
            }
        ]
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
        return await request_structured_output(
            self._client,
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            schema=ReviewReport,
            schema_name="review_report",
            max_tokens=self.max_tokens,
            temperature=0.1,
            mode=self.structured_output_mode,
            validation_retries=1,
        )

    @staticmethod
    def _data_url(path: Path) -> str:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for Review VLM frames") from exc
        with Image.open(path) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
