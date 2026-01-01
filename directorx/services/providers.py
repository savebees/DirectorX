from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path

from directorx.core.models import (
    CandidateScore,
    MusicTrack,
    Scene,
    ShotRequest,
    Storyboard,
    VideoIndex,
)
from directorx.indexing.vlm import OpenAICompatibleSceneAnnotator


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
        model: str = "gpt-5.6",
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
        """Keep planner context bounded while preserving dialogue and plot evidence."""
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
            evidence = [scene for scene in candidates if scene.plot_event]
            if not evidence:
                evidence = [scene for scene in candidates if scene.transcript]
            pool = evidence or candidates
            selected.append(pool[len(pool) // 2])
        return selected


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


class VLMGroundingModel:
    """Reranks retrieved candidates using visual annotations from the configured VLM."""

    def __init__(self, annotator: OpenAICompatibleSceneAnnotator) -> None:
        self.annotator = annotator

    async def score(
        self, shot: ShotRequest, candidates: list[Scene]
    ) -> list[CandidateScore]:
        annotations = await self.annotator.annotate_batch(candidates)
        query = _content_tokens(f"{shot.visual_query} {shot.narration_text}")
        results: list[CandidateScore] = []
        for rank, scene in enumerate(candidates):
            annotation = annotations.get(scene.id)
            if annotation is None:
                raise ValueError(f"VLM omitted grounding candidate {scene.id}")
            content = _content_tokens(
                f"{annotation.caption} {' '.join(annotation.tags)} "
                f"{' '.join(annotation.characters)} {' '.join(annotation.actions)} "
                f"{annotation.location or ''} {' '.join(annotation.objects)} "
                f"{annotation.plot_event or ''} {scene.transcript}"
            )
            lexical = len(query & content) / max(1, len(query))
            confidence = annotation.confidence
            score = min(1.0, 0.2 + lexical * 0.7 + confidence * 0.1 - rank * 0.005)
            results.append(
                CandidateScore(
                    scene_id=scene.id,
                    score=max(0.0, score),
                    rationale=(
                        "configured VLM grounding with structured scene evidence"
                    ),
                )
            )
        return results


def _content_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    tokens.update(cjk)
    tokens.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    return {token for token in tokens if token}


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
