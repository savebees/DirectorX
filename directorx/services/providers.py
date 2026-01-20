from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from directorx.core.models import (
    MusicTrack,
    Scene,
    SceneTags,
    Storyboard,
    VideoIndex,
)


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


class OpenAICompatibleScreenwriterModel:
    """Structured text planning against an OpenAI-compatible model endpoint."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "VYCE_API_KEY",
        max_tokens: int = 4000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        secret = os.environ.get(api_key_env, "")
        if not secret:
            raise RuntimeError(
                f"Missing planner API key. Set {api_key_env} in the environment"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for online story planning"
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=secret,
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            max_retries=max_retries,
        )

    async def draft(
        self, prompt: str, video_index: VideoIndex, target_duration_s: float
    ) -> Storyboard:
        selected_scenes = self._select_context_scenes(video_index.scenes, limit=40)
        scene_context = "\n".join(
            f"{scene.id} [{scene.source_range.start_s:.1f}-"
            f"{scene.source_range.end_s:.1f}s] caption={scene.caption!r}; "
            f"tags={scene.tags}; dialogue={scene.transcript[:240]!r}"
            for scene in selected_scenes
        )
        request = (
            "为一部电影剪辑生成中文解说故事板。严格依据场景摘要和对白，"
            "不要编造未提供的情节，避免过度剧透；每个 beat 的 narration "
            "要适合口播，visual_intent 要包含可检索的具体画面。"
            f"目标时长约 {target_duration_s:.1f} 秒，用户要求：{prompt}"
            f"\n\n场景索引：\n{scene_context}"
        )
        messages = [
            {
                "role": "system",
                "content": "只返回符合 schema 的 JSON，不要 markdown，不要解释。",
            },
            {"role": "user", "content": request},
        ]
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "storyboard",
                    "strict": True,
                    "schema": Storyboard.model_json_schema(),
                },
            },
        )
        if not completion.choices:
            raise ValueError("Screenwriter model returned no choices")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Screenwriter model returned no JSON")
        return Storyboard.model_validate_json(content)

    @staticmethod
    def _select_context_scenes(scenes: list[Scene], limit: int) -> list[Scene]:
        """Keep planner context bounded while preserving visual and dialogue.

        The selected scenes remain representative of the whole source video.
        """
        if len(scenes) <= limit:
            return scenes
        # Divide the film into temporal buckets instead of taking a prefix. This
        # keeps a feature-length planner context representative even when every
        # shot has subtitles and therefore qualifies as evidence.
        selected: list[Scene] = []
        scene_count = len(scenes)
        for bucket in range(limit):
            start = bucket * scene_count // limit
            end = max(start + 1, (bucket + 1) * scene_count // limit)
            candidates = scenes[start:end]
            evidence = [scene for scene in candidates if scene.dense_caption]
            if not evidence:
                evidence = [scene for scene in candidates if scene.transcript]
            pool = evidence or candidates
            selected.append(pool[len(pool) // 2])
        return selected


class OpenAICompatibleSceneTagger:
    """Create an information-rich, searchable record for each scene."""

    SYSTEM_PROMPT = """Create searchable metadata for one video scene.
Use the dense visual caption and transcript as your only evidence.

Write CAPTION as a detailed, self-contained description of the scene. Combine
what is visibly shown with the meaning of relevant dialogue, but do not quote
or reproduce the dialogue. Describe people, actions, interactions, setting,
location, objects, visible text, spatial relationships, and any dialogue-based
context that is supported by the transcript. Use as much concrete detail as
the evidence supports.
Do not invent identities, motives, hidden events, or details absent from the
inputs. Then extract concise searchable labels from the same evidence.

Return exactly these six lines, in this order:
CAPTION: <information-rich factual caption>
TAGS: <short keywords or noun phrases separated by semicolons>
CHARACTERS: <observable people or stable generic labels separated by semicolons>
ACTIONS: <visible actions separated by semicolons>
LOCATION: <location, or none>
OBJECTS: <visible searchable objects separated by semicolons>

Use none for an empty field. Avoid duplicates, vague adjectives, and
speculative labels. Return plain text only: no JSON, markdown, bullets,
headings, or extra commentary."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "VYCE_API_KEY",
        max_tokens: int = 1200,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        secret = os.environ.get(api_key_env, "")
        if client is not None:
            self._client = client
        else:
            if not secret:
                raise RuntimeError(
                    f"Missing tagger API key. Set {api_key_env} in the environment"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for scene tagging"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=secret,
                base_url=base_url.rstrip("/"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self.model = model
        self.max_tokens = max_tokens

    async def tag_batch(self, scenes: list[Scene]) -> dict[str, SceneTags]:
        results = await asyncio.gather(*(self._tag_one(scene) for scene in scenes))
        return {scene_id: tags for scene_id, tags in results}

    async def _tag_one(self, scene: Scene) -> tuple[str, SceneTags]:
        request = (
            f"Scene ID: {scene.id}\n"
            f"Transcript: {scene.transcript or '(none)'}\n"
            f"Dense visual caption: {scene.dense_caption or '(none)'}"
        )
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            temperature=0.1,
            max_tokens=self.max_tokens,
        )
        if not completion.choices:
            raise ValueError(f"Scene tagger returned no choices for {scene.id}")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Scene tagger returned no metadata for {scene.id}")
        return scene.id, self._parse_response(content, scene.id)

    @staticmethod
    def _parse_response(content: str, scene_id: str) -> SceneTags:
        fields = ("CAPTION", "TAGS", "CHARACTERS", "ACTIONS", "LOCATION", "OBJECTS")
        values: dict[str, str] = {}
        for line in content.strip().splitlines():
            match = re.fullmatch(
                r"(CAPTION|TAGS|CHARACTERS|ACTIONS|LOCATION|OBJECTS):\s*(.*)",
                line,
            )
            if match is None:
                raise ValueError(
                    f"Scene tagger returned an invalid line for {scene_id}"
                )
            name, value = match.groups()
            if name in values:
                raise ValueError(f"Scene tagger repeated {name} for {scene_id}")
            values[name] = value.strip()

        missing = [name for name in fields if name not in values]
        if missing:
            raise ValueError(
                f"Scene tagger omitted fields for {scene_id}: {', '.join(missing)}"
            )
        caption = values["CAPTION"]
        if not caption or caption.lower() == "none":
            raise ValueError(f"Scene tagger returned an empty caption for {scene_id}")

        def split_items(name: str) -> list[str]:
            value = values[name]
            if value.lower() == "none":
                return []
            return [item.strip() for item in value.split(";") if item.strip()]

        location = values["LOCATION"]
        if location.lower() == "none":
            location = None
        return SceneTags(
            caption=caption,
            tags=split_items("TAGS"),
            characters=split_items("CHARACTERS"),
            actions=split_items("ACTIONS"),
            location=location,
            objects=split_items("OBJECTS"),
        )


class EdgeSpeechTTS:
    """Cross-platform Chinese Edge TTS with fail-fast error handling."""

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: int = 185,
    ) -> None:
        self.voice = voice
        self.rate = rate

    async def synthesize(self, text: str, output_path: Path) -> float:
        return await asyncio.to_thread(self._synthesize_sync, text, output_path)

    def _synthesize_sync(self, text: str, output_path: Path) -> float:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._edge_tts(text, output_path)
        duration = _probe_duration(output_path)
        if duration <= 0:
            raise RuntimeError("Edge TTS produced an empty audio file")
        return duration

    def _edge_tts(self, text: str, output_path: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="video-edge-tts-") as temporary:
            source = Path(temporary) / "speech.mp3"
            import edge_tts

            async def save() -> None:
                communicator = edge_tts.Communicate(
                    text, self.voice, rate=self._edge_rate()
                )
                await communicator.save(str(source))

            asyncio.run(save())
            self._convert_to_wav(source, output_path)

    @staticmethod
    def _convert_to_wav(source: Path, output_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ar",
                "48000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _edge_rate(self) -> str:
        # Edge accepts signed percentage strings, while the CLI exposes a
        # familiar words-per-minute-like integer for the native engines.
        delta = round((self.rate / 185 - 1) * 100)
        return f"{delta:+d}%"


class DirectoryMusicLibrary:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def tracks(self) -> list[MusicTrack]:
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)
        paths = [
            path
            for path in self.directory.iterdir()
            if path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac"}
        ]
        if not paths:
            raise ValueError(f"No supported music files in {self.directory}")
        tracks = []
        for path in paths:
            duration = await asyncio.to_thread(_probe_duration, path)
            tags = [part.lower() for part in re.split(r"[-_\s]+", path.stem) if part]
            tracks.append(
                MusicTrack(path=path, title=path.stem, tags=tags, duration_s=duration)
            )
        return tracks
