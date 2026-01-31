from __future__ import annotations

from dataclasses import dataclass

from directorx.config import AppConfig
from directorx.core.ports import EmbeddingProvider, Transcriber
from directorx.indexing import (
    AutoTranscriber,
    ClipShotVisualEmbeddingProvider,
    EmbeddedSubtitleTranscriber,
    FasterWhisperTranscriber,
    HashingEmbeddingProvider,
    HybridVideoIndexer,
    NullTranscriber,
    OpenAICompatibleDenseCaptioner,
    PySceneDetectDetector,
    SentenceTransformerEmbeddingProvider,
    ShotKeyframeSelector,
    SidecarSubtitleTranscriber,
    VisualSceneGrouper,
)
from directorx.services.providers import (
    OpenAICompatibleSceneTagger,
    OpenAICompatibleStoryStructureModel,
)


@dataclass(frozen=True)
class IndexingRuntime:
    embedding: EmbeddingProvider
    captioner: OpenAICompatibleDenseCaptioner
    tagger: OpenAICompatibleSceneTagger
    story_structure_model: OpenAICompatibleStoryStructureModel
    indexer: HybridVideoIndexer


def create_indexing_runtime(config: AppConfig) -> IndexingRuntime:
    embedding = _create_embedding(config)
    captioner = OpenAICompatibleDenseCaptioner(
        model=config.vlm.model,
        base_url=config.vlm.base_url,
        api_key_env=config.vlm.api_key_env,
        max_parallel=config.vlm.workers,
        timeout_s=config.vlm.timeout_s,
        max_retries=config.vlm.retries,
        max_vlm_frames_per_scene=config.vlm.max_vlm_frames_per_scene,
        max_tokens=config.vlm.max_tokens,
        max_image_dimension=config.vlm.max_image_dimension,
        request_interval_s=config.vlm.request_interval_s,
    )
    tagger = OpenAICompatibleSceneTagger(
        model=config.llm.scene_tagger_model,
        base_url=config.llm.base_url,
        api_key_env=config.llm.api_key_env,
        max_tokens=config.llm.scene_tagger_max_tokens,
        timeout_s=config.llm.timeout_s,
        max_retries=config.llm.retries,
    )
    story_structure_model = OpenAICompatibleStoryStructureModel(
        model=config.llm.story_structure_model,
        base_url=config.llm.base_url,
        api_key_env=config.llm.api_key_env,
        max_tokens=config.llm.story_structure_max_tokens,
        timeout_s=config.llm.timeout_s,
        max_retries=config.llm.retries,
        max_scenes_per_chunk=config.llm.story_structure_max_scenes_per_chunk,
    )
    indexer = HybridVideoIndexer(
        cache_dir=config.resolve(config.paths.cache_dir),
        shot_detector=PySceneDetectDetector(),
        scene_grouper=VisualSceneGrouper(
            ClipShotVisualEmbeddingProvider(config.scene_grouping.model),
            similarity_threshold=config.scene_grouping.similarity_threshold,
            max_scene_duration_s=config.scene_grouping.max_scene_duration_s,
        ),
        transcriber=_create_transcriber(config),
        keyframe_extractor=ShotKeyframeSelector(
            candidate_fps=config.indexing.candidate_fps,
            target_keyframe_interval_s=config.indexing.target_keyframe_interval_s,
            max_keyframes_per_shot=config.indexing.max_keyframes_per_shot,
            max_parallel=config.indexing.frame_workers,
        ),
        captioner=captioner,
        tagger=tagger,
        embedding_provider=embedding,
        batch_size=config.indexing.batch_size,
    )
    return IndexingRuntime(
        embedding=embedding,
        captioner=captioner,
        tagger=tagger,
        story_structure_model=story_structure_model,
        indexer=indexer,
    )


def _create_transcriber(config: AppConfig) -> Transcriber:
    sidecar = SidecarSubtitleTranscriber(
        (
            config.resolve(config.transcription.subtitle_path)
            if config.transcription.subtitle_path is not None
            else None
        ),
        encoding=config.transcription.subtitle_encoding,
    )
    whisper = FasterWhisperTranscriber(
        config.transcription.whisper_model,
        device=config.transcription.whisper_device,
        compute_type=config.transcription.whisper_compute_type,
        language=config.transcription.language,
    )
    factories = {
        "auto": lambda: AutoTranscriber(
            sidecar, EmbeddedSubtitleTranscriber(), whisper
        ),
        "subtitles": lambda: sidecar,
        "embedded": EmbeddedSubtitleTranscriber,
        "whisper": lambda: whisper,
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
