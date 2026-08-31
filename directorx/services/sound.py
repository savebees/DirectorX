from __future__ import annotations

import asyncio
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from directorx.core.models import MusicIndex, MusicIndexEntry, TimeRange
from directorx.core.ports import AudioTextEmbeddingProvider, MusicLibrary


class LocalMusicIndexBuilder:
    """Build a persistent CLAP index for a music library."""

    def __init__(
        self,
        music_library: MusicLibrary,
        embedding_provider: AudioTextEmbeddingProvider,
        *,
        model_name: str,
        analysis_window_s: float = 10.0,
        analysis_windows_per_track: int = 3,
    ) -> None:
        if analysis_window_s <= 0:
            raise ValueError("analysis_window_s must be positive")
        if analysis_windows_per_track <= 0:
            raise ValueError("analysis_windows_per_track must be positive")
        self.music_library = music_library
        self.embedding_provider = embedding_provider
        self.model_name = model_name
        self.analysis_window_s = analysis_window_s
        self.analysis_windows_per_track = analysis_windows_per_track

    async def build(self) -> MusicIndex:
        tracks = sorted(
            await self.music_library.tracks(), key=lambda item: str(item.path)
        )
        if not tracks:
            raise ValueError("Music library returned no tracks")
        paths = [track.path for track in tracks]
        if len(paths) != len(set(paths)):
            raise ValueError("Music library returned duplicate track paths")

        entries: list[MusicIndexEntry] = []
        dimension: int | None = None
        for track in tracks:
            windows = self.analysis_windows(track.duration_s)
            embeddings = await self.embedding_provider.embed_audio(track.path, windows)
            embedding = self._mean_embedding(embeddings, f"music track {track.path}")
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError("Music embeddings have inconsistent dimensions")
            entries.append(
                MusicIndexEntry(
                    track=track,
                    embedding=embedding,
                    analysis_windows=windows,
                    model_name=self.model_name,
                )
            )
        assert dimension is not None
        return MusicIndex(
            model_name=self.model_name,
            embedding_dimension=dimension,
            entries=entries,
        )

    def analysis_windows(self, duration_s: float) -> list[TimeRange]:
        if (
            not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or duration_s <= 0
        ):
            raise ValueError("Music track duration must be positive")
        window_s = min(self.analysis_window_s, float(duration_s))
        if duration_s <= window_s or self.analysis_windows_per_track == 1:
            start_s = max(0.0, (duration_s - window_s) / 2)
            return [TimeRange(start_s=start_s, end_s=start_s + window_s)]
        last_start = duration_s - window_s
        starts = [
            last_start * index / (self.analysis_windows_per_track - 1)
            for index in range(self.analysis_windows_per_track)
        ]
        return [TimeRange(start_s=start, end_s=start + window_s) for start in starts]

    @staticmethod
    def _mean_embedding(embeddings: list[list[float]], label: str) -> list[float]:
        if not embeddings:
            raise ValueError(f"{label} returned no embeddings")
        normalized = [
            LocalMusicIndexBuilder._normalize(item, label) for item in embeddings
        ]
        dimension = len(normalized[0])
        if any(len(item) != dimension for item in normalized):
            raise ValueError(f"{label} returned inconsistent embedding dimensions")
        mean = [
            sum(item[index] for item in normalized) / len(normalized)
            for index in range(dimension)
        ]
        return LocalMusicIndexBuilder._normalize(mean, label)

    @staticmethod
    def _normalize(embedding: list[float], label: str) -> list[float]:
        if not embedding or any(not math.isfinite(value) for value in embedding):
            raise ValueError(f"{label} returned an invalid embedding")
        norm = math.sqrt(sum(value * value for value in embedding))
        if norm == 0:
            raise ValueError(f"{label} returned a zero embedding")
        return [value / norm for value in embedding]

    async def build_and_persist(self, destination: Path) -> Path:
        index = await self.build()
        return self.persist(index, destination)

    @staticmethod
    def persist(index: MusicIndex, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "music-index.json"
        temporary: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(
                dir=destination, prefix=".music-index.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary = Path(raw_path)
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(index.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise FileExistsError(path)
            os.replace(temporary, path)
            temporary = None
            return path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class LocalClapAudioTextEmbeddingProvider:
    """Embed text and sampled music windows in the same local CLAP space."""

    sample_rate = 48_000

    def __init__(
        self,
        model_name: str = "laion/larger_clap_music",
        *,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None

    async def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("CLAP text query cannot be empty")
        return await asyncio.to_thread(self._embed_text_sync, text)

    async def embed_audio(
        self, path: Path, windows: list[TimeRange]
    ) -> list[list[float]]:
        if not windows:
            raise ValueError("CLAP audio embedding requires at least one window")
        waveforms = await asyncio.gather(
            *(
                asyncio.to_thread(self._decode_window, path, window)
                for window in windows
            )
        )
        return await asyncio.to_thread(self._embed_audio_sync, waveforms)

    def _embed_text_sync(self, text: str) -> list[float]:
        self._load_backend()
        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        features = self._feature_tensor(features)
        return features[0].detach().cpu().float().tolist()

    def _embed_audio_sync(self, waveforms: list[Any]) -> list[list[float]]:
        self._load_backend()
        # Transformers 5 renamed the CLAP processor keyword from ``audios``
        # to ``audio``. Use the current spelling; older supported releases
        # are handled by the small compatibility fallback below.
        try:
            inputs = self._processor(
                audio=waveforms,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
        except (TypeError, ValueError) as exc:
            if "audios" not in str(exc) and "audio" not in str(exc):
                raise
            inputs = self._processor(
                audios=waveforms,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_audio_features(**inputs)
        features = self._feature_tensor(features)
        return features.detach().cpu().float().tolist()

    @staticmethod
    def _feature_tensor(features: Any) -> Any:
        """Normalize CLAP outputs across Transformers tensor/model-output APIs."""
        pooled = getattr(features, "pooler_output", None)
        return pooled if pooled is not None else features

    def _decode_window(self, path: Path, window: TimeRange):
        try:
            import numpy
        except ImportError as exc:
            raise RuntimeError("numpy is required for CLAP music matching") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{window.start_s:.6f}",
                "-t",
                f"{window.duration_s:.6f}",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-acodec",
                "pcm_f32le",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        waveform = numpy.frombuffer(result.stdout, dtype=numpy.float32)
        if waveform.size == 0:
            raise ValueError(f"FFmpeg decoded no audio from {path}")
        return waveform.copy()

    def _load_backend(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for CLAP music matching"
            ) from exc

        device = self.requested_device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and (
                torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        self._processor = ClapProcessor.from_pretrained(self.model_name)
        self._model = ClapModel.from_pretrained(self.model_name).to(device).eval()
        self._torch = torch
        self._device = torch.device(device)
