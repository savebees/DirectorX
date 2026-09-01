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
    frame_workers: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    candidate_fps: float = Field(gt=0)
    target_keyframe_interval_s: float = Field(gt=0)
    max_keyframes_per_shot: int = Field(gt=0)


class SceneGroupingConfig(BaseModel):
    model: str
    similarity_threshold: float = Field(ge=0, le=1)
    max_scene_duration_s: float = Field(gt=0)


class TranscriptionConfig(BaseModel):
    provider: Literal["auto", "subtitles", "embedded", "whisper", "none"]
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
    max_vlm_frames_per_scene: int = Field(gt=0)
    max_image_dimension: int = Field(gt=0)
    request_interval_s: float = Field(ge=0)
    structured_output_mode: Literal["json_object", "tool_call", "prompted_json"] = (
        "json_object"
    )


class LLMConfig(BaseModel):
    provider: Literal["openai-compatible"]
    base_url: str
    api_key_env: str
    screenwriter_model: str
    screenwriter_max_tokens: int = Field(gt=0)
    scene_tagger_model: str
    scene_tagger_max_tokens: int = Field(gt=0)
    story_structure_model: str = "gpt-5.6-luna"
    story_structure_max_tokens: int = Field(default=4000, gt=0)
    story_structure_max_scenes_per_chunk: int = Field(default=24, gt=0)
    scene_tagger_max_parallel: int = Field(default=2, gt=0)
    request_interval_s: float = Field(default=1.0, ge=0)
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    screenwriter_fallback_model: str | None = None
    screenwriter_fallback_base_url: str | None = None
    screenwriter_fallback_api_key_env: str | None = None
    structured_output_mode: Literal["json_object", "tool_call", "prompted_json"] = (
        "prompted_json"
    )
    screenwriter_fallback_structured_output_mode: Literal[
        "json_object", "tool_call", "prompted_json"
    ] = "json_object"


class TTSConfig(BaseModel):
    provider: Literal["edge"]
    language: str = "zh-CN"


class GroundingConfig(BaseModel):
    candidate_limit: int = Field(gt=0)
    candidate_padding_s: float = Field(ge=0)
    coarse_fps: float = Field(gt=0)
    refine_fps: float = Field(gt=0)
    refine_margin_s: float = Field(ge=0)
    max_coarse_frames: int = Field(gt=0)
    max_refine_frames: int = Field(gt=0)
    max_parallel: int = Field(gt=0)


class SoundConfig(BaseModel):
    embedding_model: str
    device: str
    analysis_window_s: float = Field(gt=0)
    analysis_windows_per_track: int = Field(gt=0)
    gain_db: float
    duck_under_voice_db: float


class ReviewConfig(BaseModel):
    max_frames: int = Field(gt=0)


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
    min_voice_coverage: float = Field(default=0.65, ge=0, le=1)
    max_voice_coverage: float = Field(default=0.88, ge=0, le=1)
    breathing_room_s: float = Field(default=0.35, ge=0)
    max_freeze_per_clip_s: float = Field(default=0.3, ge=0)
    max_title_duration_s: float = Field(default=8.0, gt=0)


class AppConfig(BaseModel):
    paths: PathsConfig
    indexing: IndexingConfig
    scene_grouping: SceneGroupingConfig
    transcription: TranscriptionConfig
    embedding: EmbeddingConfig
    vlm: VLMConfig
    llm: LLMConfig
    tts: TTSConfig
    grounding: GroundingConfig
    sound: SoundConfig
    review: ReviewConfig
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
