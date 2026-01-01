from __future__ import annotations

from dataclasses import dataclass

from directorx.config import AppConfig
from directorx.core.ports import EmbeddingProvider, Transcriber
from directorx.indexing import (
    EmbeddedSubtitleTranscriber,
    FasterWhisperTranscriber,
    FFmpegKeyframeExtractor,
    HashingEmbeddingProvider,
    HybridVideoIndexer,
    NullTranscriber,
    OpenAICompatibleSceneAnnotator,
    PySceneDetectDetector,
    SentenceTransformerEmbeddingProvider,
    SidecarSubtitleTranscriber,
)


@dataclass(frozen=True)
class IndexingRuntime:
    embedding: EmbeddingProvider
    annotator: OpenAICompatibleSceneAnnotator
    indexer: HybridVideoIndexer


def create_indexing_runtime(config: AppConfig) -> IndexingRuntime:
    embedding = _create_embedding(config)
    annotator = OpenAICompatibleSceneAnnotator(
        model=config.vlm.model,
        base_url=config.vlm.base_url,
        api_key_env=config.vlm.api_key_env,
        max_parallel=config.vlm.workers,
        timeout_s=config.vlm.timeout_s,
        max_retries=config.vlm.retries,
        max_frames_per_scene=config.vlm.max_frames_per_scene,
        max_tokens=config.vlm.max_tokens,
        max_image_dimension=config.vlm.max_image_dimension,
        request_interval_s=config.vlm.request_interval_s,
    )
    indexer = HybridVideoIndexer(
        cache_dir=config.resolve(config.paths.cache_dir),
        scene_detector=PySceneDetectDetector(detector=config.indexing.detector),
        transcriber=_create_transcriber(config),
        keyframe_extractor=FFmpegKeyframeExtractor(
            positions=tuple(config.indexing.keyframe_positions),
            max_parallel=config.indexing.frame_workers,
        ),
        annotator=annotator,
        embedding_provider=embedding,
        max_scene_duration_s=config.indexing.max_scene_duration_s,
        annotation_batch_size=config.indexing.annotation_batch_size,
    )
    return IndexingRuntime(
        embedding=embedding,
        annotator=annotator,
        indexer=indexer,
    )


def _create_transcriber(config: AppConfig) -> Transcriber:
    factories = {
        "subtitles": lambda: SidecarSubtitleTranscriber(
            (
                config.resolve(config.transcription.subtitle_path)
                if config.transcription.subtitle_path is not None
                else None
            ),
            encoding=config.transcription.subtitle_encoding,
        ),
        "embedded": EmbeddedSubtitleTranscriber,
        "whisper": lambda: FasterWhisperTranscriber(
            config.transcription.whisper_model,
            device=config.transcription.whisper_device,
            compute_type=config.transcription.whisper_compute_type,
            language=config.transcription.language,
        ),
        "none": NullTranscriber,
    }
    return factories[config.transcription.provider]()


def _create_embedding(config: AppConfig) -> EmbeddingProvider:
    factories = {
        "sentence-transformers": lambda: SentenceTransformerEmbeddingProvider(
            config.embedding.model
        ),
        "hashing": lambda: HashingEmbeddingProvider(
            dimension=config.embedding.hashing_dimension
        ),
    }
    return factories[config.embedding.provider]()
