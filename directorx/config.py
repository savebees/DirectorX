from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr


class PathsConfig(BaseModel):
    cache_dir: Path
    music_dir: Path
    artifacts_dir: Path


class IndexingConfig(BaseModel):
    detector: Literal["adaptive", "content"]
    keyframe_positions: list[float]
    frame_workers: int = Field(gt=0)
    max_scene_duration_s: float = Field(gt=0)
    annotation_batch_size: int = Field(gt=0)


class TranscriptionConfig(BaseModel):
    provider: Literal["subtitles", "embedded", "whisper", "none"]
    subtitle_path: Path | None
    subtitle_encoding: str
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    language: str | None


class EmbeddingConfig(BaseModel):
    provider: Literal["sentence-transformers", "hashing"]
    model: str
    hashing_dimension: int = Field(gt=0)


class VLMConfig(BaseModel):
    provider: Literal["openai-compatible"]
    model: str
    base_url: str
    api_key_env: str
    workers: int = Field(gt=0)
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    max_frames_per_scene: int = Field(gt=0)
    max_image_dimension: int = Field(gt=0)
    request_interval_s: float = Field(ge=0)


class LLMConfig(BaseModel):
    provider: Literal["openai-compatible"]
    base_url: str
    api_key_env: str
    screenwriter_model: str
    screenwriter_max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)


class TTSConfig(BaseModel):
    provider: Literal["edge"]
    voice: str
    rate: int = Field(gt=0)


class RenderConfig(BaseModel):
    enabled: bool
    aspect: Literal["portrait", "landscape", "square"]
    fps: int = Field(gt=0)
    video_codec: str

    @property
    def dimensions(self) -> tuple[int, int]:
        return {
            "portrait": (1080, 1920),
            "landscape": (1920, 1080),
            "square": (1080, 1080),
        }[self.aspect]


class EditConfig(BaseModel):
    target_duration_s: float = Field(gt=0)


class AppConfig(BaseModel):
    paths: PathsConfig
    indexing: IndexingConfig
    transcription: TranscriptionConfig
    embedding: EmbeddingConfig
    vlm: VLMConfig
    llm: LLMConfig
    tts: TTSConfig
    render: RenderConfig
    edit: EditConfig

    _root: Path = PrivateAttr()

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        source = path.resolve()
        config = cls.model_validate(tomllib.loads(source.read_text(encoding="utf-8")))
        config._root = source.parent
        return config

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self._root / path
