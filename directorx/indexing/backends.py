from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from directorx.core.models import (
    DialogueLine,
    DialogueWord,
    Keyframe,
    Scene,
    TimeRange,
)


class PySceneDetectDetector:
    """Content-aware shot detection backed by the maintained PySceneDetect API."""

    def __init__(
        self,
        *,
        threshold: float = 27.0,
        min_scene_len_frames: int = 15,
    ) -> None:
        self.threshold = threshold
        self.min_scene_len_frames = min_scene_len_frames

    async def detect(self, video_path: Path, duration_s: float) -> list[TimeRange]:
        def run() -> list[TimeRange]:
            try:
                from scenedetect import AdaptiveDetector, detect
            except ImportError as exc:
                raise RuntimeError(
                    "PySceneDetect is required for hybrid indexing; "
                    "install the scene-index extra"
                ) from exc

            algorithm = AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=self.min_scene_len_frames,
                min_content_val=max(5.0, self.threshold / 2),
            )

            detected = detect(str(video_path), algorithm, start_in_scene=True)
            ranges = [
                TimeRange(start_s=start.get_seconds(), end_s=end.get_seconds())
                for start, end in detected
                if end.get_seconds() > start.get_seconds()
            ]
            if not ranges:
                raise ValueError(f"Scene detector returned no ranges for {video_path}")
            return ranges

        return await asyncio.to_thread(run)


class NoEmbeddedSubtitleError(RuntimeError):
    """The source has no compatible embedded text subtitle track."""


class EmbeddedSubtitleTranscriber:
    """Extracts the first compatible embedded text subtitle track with FFmpeg."""

    TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}

    def __init__(
        self, preferred_languages: tuple[str, ...] = ("chi", "zho", "eng")
    ) -> None:
        self.preferred_languages = preferred_languages

    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        return await asyncio.to_thread(self._transcribe_sync, video_path)

    def _transcribe_sync(self, video_path: Path) -> list[DialogueLine]:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name:stream_tags=language",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        compatible = [
            stream for stream in streams if stream.get("codec_name") in self.TEXT_CODECS
        ]
        if not compatible:
            raise NoEmbeddedSubtitleError(video_path)

        language_rank = {
            language: rank for rank, language in enumerate(self.preferred_languages)
        }
        compatible.sort(
            key=lambda stream: language_rank.get(
                (stream.get("tags") or {}).get("language", ""),
                len(language_rank),
            )
        )
        stream = compatible[0]
        language = (stream.get("tags") or {}).get("language")
        with tempfile.TemporaryDirectory(prefix="video-subtitles-") as temporary:
            subtitle_path = Path(temporary) / "subtitles.srt"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-map",
                    f"0:{stream['index']}",
                    "-c:s",
                    "srt",
                    str(subtitle_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return self._parse_srt(
                subtitle_path.read_text(encoding="utf-8-sig"), language
            )

    @classmethod
    def _parse_srt(cls, text: str, language: str | None) -> list[DialogueLine]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        lines: list[DialogueLine] = []
        for block in re.split(r"\n{2,}", normalized):
            parts = block.splitlines()
            timing_index = next(
                (index for index, value in enumerate(parts) if "-->" in value), None
            )
            if timing_index is None:
                continue
            match = re.match(r"\s*(\S+)\s+-->\s+(\S+)", parts[timing_index])
            if not match:
                continue
            content = " ".join(
                part.strip() for part in parts[timing_index + 1 :] if part.strip()
            )
            content = re.sub(r"<[^>]+>", "", content).strip()
            if not content:
                continue
            start_s = cls._timestamp_seconds(match.group(1))
            end_s = cls._timestamp_seconds(match.group(2))
            if end_s <= start_s:
                continue
            lines.append(
                DialogueLine(
                    text=content,
                    start_s=start_s,
                    end_s=end_s,
                    language=language,
                )
            )
        return lines

    @staticmethod
    def _timestamp_seconds(value: str) -> float:
        cleaned = value.replace(",", ".")
        hours, minutes, seconds = cleaned.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


class SidecarSubtitleTranscriber:
    """Loads an explicit or automatically discovered sidecar SRT file."""

    LANGUAGE_ALIASES = {
        "chi": ("chi", "zho", "zh", "chinese", "中文", "简体", "繁体"),
        "zho": ("chi", "zho", "zh", "chinese", "中文", "简体", "繁体"),
        "eng": ("eng", "en", "english"),
    }

    def __init__(
        self,
        subtitle_path: Path | None = None,
        preferred_languages: tuple[str, ...] = ("chi", "zho", "eng"),
        encoding: str = "utf-8-sig",
    ) -> None:
        self.subtitle_path = subtitle_path
        self.preferred_languages = preferred_languages
        self.encoding = encoding

    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        return await asyncio.to_thread(self._transcribe_sync, video_path)

    def _transcribe_sync(self, video_path: Path) -> list[DialogueLine]:
        subtitle_path = self._resolve_subtitle(video_path)
        return EmbeddedSubtitleTranscriber._parse_srt(
            subtitle_path.read_text(encoding=self.encoding),
            self._language_for_path(subtitle_path),
        )

    def _resolve_subtitle(self, video_path: Path) -> Path:
        if self.subtitle_path is not None:
            path = self.subtitle_path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() != ".srt":
                raise ValueError(f"Only SRT sidecar subtitles are supported: {path}")
            return path

        parent = video_path.resolve().parent
        candidates: set[Path] = set()
        for directory_name in ("", "Subs", "subs", "Subtitles", "subtitles"):
            directory = parent / directory_name if directory_name else parent
            if directory.is_dir():
                candidates.update(directory.glob("*.srt"))
        if not candidates:
            raise FileNotFoundError(f"No sidecar SRT found beside {video_path}")

        source_stem = video_path.stem.casefold()
        language_rank = {
            language: rank for rank, language in enumerate(self.preferred_languages)
        }

        def rank(path: Path) -> tuple[int, int, str]:
            language = self._language_for_path(path)
            preference = language_rank.get(language, len(language_rank))
            exact_stem = 0 if path.stem.casefold() == source_stem else 1
            return preference, exact_stem, str(path).casefold()

        return min(candidates, key=rank)

    @classmethod
    def _language_for_path(cls, path: Path) -> str | None:
        name = path.stem.casefold()
        tokens = set(re.findall(r"[a-z]+|[\u4e00-\u9fff]+", name))
        for canonical, aliases in cls.LANGUAGE_ALIASES.items():
            if any(alias in tokens for alias in aliases):
                return canonical
        return None


class AutoTranscriber:
    """Use sidecar subtitles, embedded subtitles, then Whisper ASR."""

    def __init__(
        self,
        sidecar: SidecarSubtitleTranscriber,
        embedded: EmbeddedSubtitleTranscriber,
        whisper: FasterWhisperTranscriber,
    ) -> None:
        self.sidecar = sidecar
        self.embedded = embedded
        self.whisper = whisper

    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        try:
            return await self.sidecar.transcribe(video_path)
        except FileNotFoundError:
            return await self._transcribe_without_sidecar(video_path)

    async def _transcribe_without_sidecar(
        self, video_path: Path
    ) -> list[DialogueLine]:
        try:
            return await self.embedded.transcribe(video_path)
        except NoEmbeddedSubtitleError:
            return await self.whisper.transcribe(video_path)


class FasterWhisperTranscriber:
    """Batched ASR with VAD and word timestamps, loaded only when selected."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        *,
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        beam_size: int = 5,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model = None

    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        return await asyncio.to_thread(self._transcribe_sync, video_path)

    def _transcribe_sync(self, video_path: Path) -> list[DialogueLine]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is required when no embedded subtitles are available"
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        segments, info = self._model.transcribe(
            str(video_path),
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        output: list[DialogueLine] = []
        for segment in segments:
            text = segment.text.strip()
            if not text or segment.end <= segment.start:
                continue
            words = [
                DialogueWord(
                    text=word.word,
                    start_s=max(segment.start, word.start),
                    end_s=max(word.start + 1e-3, min(segment.end, word.end)),
                    probability=word.probability,
                )
                for word in (segment.words or [])
                if word.start is not None
                and word.end is not None
                and word.end > word.start
            ]
            output.append(
                DialogueLine(
                    text=text,
                    start_s=segment.start,
                    end_s=segment.end,
                    language=info.language,
                    words=words,
                )
            )
        return output


class NullTranscriber:
    async def transcribe(self, video_path: Path) -> list[DialogueLine]:
        return []


class ShotKeyframeSelector:
    """Select duration-aware, sharp representative frames from each shot range."""

    def __init__(
        self,
        *,
        candidate_fps: float = 2.0,
        target_keyframe_interval_s: float = 3.0,
        max_keyframes_per_shot: int = 5,
        max_width: int = 960,
        max_parallel: int = 4,
    ) -> None:
        if candidate_fps <= 0:
            raise ValueError("candidate_fps must be positive")
        if target_keyframe_interval_s <= 0:
            raise ValueError("target_keyframe_interval_s must be positive")
        if max_keyframes_per_shot <= 0:
            raise ValueError("max_keyframes_per_shot must be positive")
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        self.candidate_fps = candidate_fps
        self.target_keyframe_interval_s = target_keyframe_interval_s
        self.max_keyframes_per_shot = max_keyframes_per_shot
        self.max_width = max_width
        self.max_parallel = max_parallel

    async def extract(
        self, video_path: Path, scenes: list[Scene], output_dir: Path
    ) -> dict[str, list[Keyframe]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_parallel)
        output: dict[str, list[Keyframe]] = {scene.id: [] for scene in scenes}

        async def one(scene: Scene) -> None:
            duration = scene.source_range.duration_s
            keyframe_count = self._keyframe_count(duration)
            with tempfile.TemporaryDirectory(
                prefix=f"{scene.id}-candidates-", dir=output_dir
            ) as temporary:
                candidate_pattern = str(Path(temporary) / "candidate-%04d.jpg")
                async with semaphore:
                    process = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{scene.source_range.start_s:.6f}",
                        "-i",
                        str(video_path),
                        "-map",
                        "0:v:0",
                        "-t",
                        f"{duration:.6f}",
                        "-vf",
                        f"fps={self.candidate_fps},scale=min({self.max_width}\\,iw):-2",
                        "-pix_fmt",
                        "yuvj420p",
                        "-q:v",
                        "2",
                        candidate_pattern,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, stderr = await process.communicate()
                if process.returncode != 0:
                    message = stderr.decode("utf-8", errors="replace")[-1200:]
                    raise RuntimeError(
                        f"Candidate frame extraction failed for {scene.id}: {message}"
                    )
                candidates = sorted(Path(temporary).glob("candidate-*.jpg"))
                if not candidates:
                    raise RuntimeError(
                        f"No candidate frames extracted for {scene.id}"
                    )
                selected = await asyncio.to_thread(
                    self._select_candidates,
                    candidates,
                    scene.source_range.start_s,
                    duration,
                    keyframe_count,
                )
                for index, (timestamp, source) in enumerate(selected, start=1):
                    frame_path = output_dir / f"{scene.id}-{index:02d}.jpg"
                    shutil.copyfile(source, frame_path)
                    output[scene.id].append(
                        Keyframe(timestamp_s=timestamp, path=frame_path)
                    )

        await asyncio.gather(
            *(
                one(scene)
                for scene in scenes
            )
        )
        for frames in output.values():
            frames.sort(key=lambda frame: frame.timestamp_s)
        return output

    def _keyframe_count(self, duration_s: float) -> int:
        return min(
            self.max_keyframes_per_shot,
            max(1, math.ceil(duration_s / self.target_keyframe_interval_s)),
        )

    @staticmethod
    def _select_candidates(
        candidates: list[Path],
        start_s: float,
        duration_s: float,
        keyframe_count: int,
    ) -> list[tuple[float, Path]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required for sharpness-aware keyframe selection"
            ) from exc

        scored: list[tuple[float, Path, float]] = []
        for index, path in enumerate(candidates):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            if float(image.mean()) < 2.0 and float(image.std()) < 1.0:
                continue
            sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
            timestamp = min(
                start_s + index / max(1, len(candidates) - 1) * duration_s,
                start_s + duration_s - 1e-3,
            )
            scored.append((timestamp, path, sharpness))
        if not scored:
            raise ValueError("All candidate frames were unusable")

        selected: list[tuple[float, Path]] = []
        for bucket in range(keyframe_count):
            lower = bucket / keyframe_count
            upper = (bucket + 1) / keyframe_count
            bucket_candidates = [
                item
                for item in scored
                if lower <= (item[0] - start_s) / duration_s < upper
            ]
            pool = bucket_candidates or scored
            choice = max(pool, key=lambda item: item[2])
            if choice[1] not in {path for _, path in selected}:
                selected.append((choice[0], choice[1]))
        return sorted(selected, key=lambda item: item[0])


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    cjk = "".join(
        token for token in words if len(token) == 1 and "\u4e00" <= token <= "\u9fff"
    )
    return words + [cjk[index : index + 2] for index in range(max(0, len(cjk) - 1))]


class HashingEmbeddingProvider:
    """Dependency-free deterministic embedding provider for controlled tests."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % self.dimension
                vector[index] += -1.0 if value & 1 else 1.0
            norm = math.sqrt(sum(item * item for item in vector)) or 1.0
            vectors.append([item / norm for item in vector])
        return vectors


class SentenceTransformerEmbeddingProvider:
    """Local semantic embeddings through sentence-transformers (BGE by default)."""

    def __init__(
        self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = 0

    @property
    def dimension(self) -> int:
        if self._model is None:
            self._load()
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic indexing"
            ) from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            self._load()
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]
