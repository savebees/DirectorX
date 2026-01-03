from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from directorx.core.models import Scene


SYSTEM_PROMPT = """You describe movie shots for a retrieval system.
Use only visible evidence in the supplied frames.
Write a dense factual caption in the requested language. Include people,
actions, objects, location, visible text, and relationships when they are
observable. Do not infer identities, hidden events, motives, or plot facts.
Return plain text only: no JSON, markdown, headings, or bullet points."""


class OpenAICompatibleDenseCaptioner:
    """Generate plain-text visual descriptions through an OpenAI-compatible VLM."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3.6-35B-A3B",
        base_url: str = "https://api.siliconflow.cn/v1",
        api_key_env: str = "SILICONFLOW_API_KEY",
        api_key: str | None = None,
        output_language: str = "Simplified Chinese",
        max_parallel: int = 2,
        timeout_s: float = 120.0,
        max_retries: int = 0,
        max_frames_per_scene: int = 3,
        max_tokens: int = 1200,
        max_image_dimension: int = 1024,
        jpeg_quality: int = 85,
        request_interval_s: float = 0.0,
        client: Any | None = None,
    ) -> None:
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if max_frames_per_scene <= 0:
            raise ValueError("max_frames_per_scene must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if request_interval_s < 0:
            raise ValueError("request_interval_s must be non-negative")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.output_language = output_language
        self.max_parallel = max_parallel
        self.max_frames_per_scene = max_frames_per_scene
        self.max_tokens = max_tokens
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        self.request_interval_s = request_interval_s
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0
        if client is not None:
            self._client = client
            return
        secret = api_key or os.environ.get(api_key_env, "")
        if not secret:
            raise RuntimeError(
                f"Missing VLM API key. Set {api_key_env} in the process environment."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for the OpenAI-compatible VLM adapter"
            ) from exc
        self._client = AsyncOpenAI(
            api_key=secret,
            base_url=self.base_url,
            timeout=timeout_s,
            max_retries=max_retries,
            default_headers={"User-Agent": "directorx/0.1"},
        )

    async def caption_batch(self, scenes: list[Scene]) -> dict[str, str]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def caption(scene: Scene) -> tuple[str, str]:
            async with semaphore:
                return scene.id, await self._caption_one(scene)

        output: dict[str, str] = {}
        for offset in range(0, len(scenes), 64):
            batch = scenes[offset : offset + 64]
            results = await asyncio.gather(*(caption(scene) for scene in batch))
            for scene_id, dense_caption in results:
                output[scene_id] = dense_caption
        return output

    async def _caption_one(self, scene: Scene) -> str:
        frames = self._select_frames(scene)
        if not frames:
            raise ValueError(f"Scene {scene.id} has no readable keyframes")
        image_parts = [
            {
                "type": "image_url",
                "image_url": {"url": self._data_url(frame)},
            }
            for frame in frames
        ]
        content = [{"type": "text", "text": self._user_prompt(scene)}, *image_parts]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        dense_caption = await self._complete(messages)
        if not dense_caption.strip():
            raise ValueError(f"VLM returned an empty dense caption for {scene.id}")
        return dense_caption.strip()

    async def _complete(self, messages: list[dict[str, Any]]) -> str:
        async with self._request_lock:
            now = time.monotonic()
            wait_s = self.request_interval_s - (now - self._last_request_at)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_request_at = time.monotonic()
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=self.max_tokens,
            stream=False,
            extra_body={"enable_thinking": False},
        )
        return self._message_text(completion)

    @staticmethod
    def _message_text(completion: Any) -> str:
        if not completion.choices:
            raise ValueError("VLM response contained no choices")
        message = completion.choices[0].message
        refusal = message.refusal
        if refusal:
            raise ValueError(f"VLM refused dense caption: {refusal}")
        content = message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("VLM response contained no text")
        return content

    def _select_frames(self, scene: Scene) -> list[Path]:
        frames = [frame.path for frame in scene.keyframes]
        if len(frames) <= self.max_frames_per_scene:
            return frames
        if self.max_frames_per_scene == 1:
            return [frames[len(frames) // 2]]
        indexes = {
            round(index * (len(frames) - 1) / (self.max_frames_per_scene - 1))
            for index in range(self.max_frames_per_scene)
        }
        return [frames[index] for index in sorted(indexes)]

    def _user_prompt(self, scene: Scene) -> str:
        return (
            f"Write in {self.output_language}.\n"
            f"Time range: {scene.source_range.start_s:.3f}s to "
            f"{scene.source_range.end_s:.3f}s.\n"
            "Describe the complete visible content of this shot in dense but "
            "concise prose."
        )

    def _data_url(self, path: Path) -> str:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        if not mime_type.startswith("image/"):
            raise ValueError(f"Keyframe is not an image: {path}")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to normalize VLM keyframes") from exc

        with Image.open(path) as image:
            image.thumbnail(
                (self.max_image_dimension, self.max_image_dimension),
                Image.Resampling.LANCZOS,
            )
            if image.mode != "RGB":
                background = Image.new("RGB", image.size, "white")
                if image.mode in {"RGBA", "LA"}:
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
