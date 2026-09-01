from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from directorx.agents.director import DirectorAgent
from directorx.agents.editor import EditorAgent
from directorx.agents.footage import FootageAnalystAgent
from directorx.agents.grounding import GroundingAgent, SceneRetriever
from directorx.agents.narration import NarrationAgent
from directorx.agents.render import RenderAgent
from directorx.agents.review import ReviewAgent
from directorx.agents.screenwriter import ScreenwriterAgent
from directorx.agents.sound import SoundAgent
from directorx.config import AppConfig
from directorx.coordination import CoordinationRuntime
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
from directorx.rendering.ffmpeg import FFmpegRenderer
from directorx.services.grounding import (
    FFmpegGroundingFrameExtractor,
    OpenAICompatibleGroundingModel,
)
from directorx.services.providers import (
    DirectoryMusicLibrary,
    EdgeSpeechTTS,
    OpenAICompatibleSceneTagger,
    OpenAICompatibleScreenwriterModel,
    OpenAICompatibleStoryStructureModel,
)
from directorx.services.review import (
    FFmpegReviewFrameExtractor,
    OpenAICompatibleReviewModel,
)
from directorx.services.sound import (
    LocalClapAudioTextEmbeddingProvider,
    LocalMusicIndexBuilder,
)


@dataclass(frozen=True)
class IndexingRuntime:
    embedding: EmbeddingProvider
    captioner: OpenAICompatibleDenseCaptioner
    tagger: OpenAICompatibleSceneTagger
    story_structure_model: OpenAICompatibleStoryStructureModel
    indexer: HybridVideoIndexer


def create_director(
    config: AppConfig,
    *,
    coordination_dir: Path | None = None,
) -> DirectorAgent:
    """Build the complete Director and specialist runtime from config."""
    artifacts_dir = config.resolve(config.paths.artifacts_dir)
    indexing = create_indexing_runtime(config)
    runtime = CoordinationRuntime(
        coordination_dir or artifacts_dir.parent / "coordination"
    )
    llm = config.llm
    screenwriter = ScreenwriterAgent(
        OpenAICompatibleScreenwriterModel(
            model=llm.screenwriter_model,
            base_url=llm.base_url,
            api_key_env=llm.api_key_env,
            max_tokens=llm.screenwriter_max_tokens,
            timeout_s=llm.timeout_s,
            max_retries=llm.retries,
            fallback_model=llm.screenwriter_fallback_model,
            fallback_base_url=llm.screenwriter_fallback_base_url,
            fallback_api_key_env=llm.screenwriter_fallback_api_key_env,
            structured_output_mode=llm.structured_output_mode,
            fallback_structured_output_mode=(
                llm.screenwriter_fallback_structured_output_mode
            ),
            narration_language=config.tts.language,
        ),
        artifacts_dir=artifacts_dir,
    )
    render_width, render_height = config.render.dimensions
    return DirectorAgent(
        runtime,
        FootageAnalystAgent(
            indexing.indexer,
            indexing.story_structure_model,
            artifacts_dir=artifacts_dir,
        ),
        screenwriter_agent=screenwriter,
        narration_agent=create_narration_agent(config),
        grounding_agent=create_grounding_agent(config),
        sound_agent=create_sound_agent(config),
        editor_agent=EditorAgent(
            artifacts_dir=artifacts_dir,
            min_voice_coverage=config.edit.min_voice_coverage,
            max_voice_coverage=config.edit.max_voice_coverage,
            breathing_room_s=config.edit.breathing_room_s,
            max_freeze_per_clip_s=config.edit.max_freeze_per_clip_s,
            max_title_duration_s=config.edit.max_title_duration_s,
        ),
        render_agent=RenderAgent(
            FFmpegRenderer(
                width=render_width,
                height=render_height,
                fps=config.render.fps,
                video_codec=config.render.video_codec,
            ),
            artifacts_dir=artifacts_dir,
            width=render_width,
            height=render_height,
        ),
        review_agent=create_review_agent(config),
        artifacts_dir=artifacts_dir,
    )


def create_narration_agent(config: AppConfig) -> NarrationAgent:
    return NarrationAgent(
        EdgeSpeechTTS(
            language=config.tts.language,
        ),
        min_voice_coverage=config.edit.min_voice_coverage,
        max_voice_coverage=config.edit.max_voice_coverage,
        breathing_room_s=config.edit.breathing_room_s,
    )


def create_grounding_agent(config: AppConfig) -> GroundingAgent:
    embedding = _create_embedding(config)
    grounding = config.grounding
    return GroundingAgent(
        OpenAICompatibleGroundingModel(
            model=config.vlm.model,
            base_url=config.vlm.base_url,
            api_key_env=config.vlm.api_key_env,
            max_tokens=config.vlm.max_tokens,
            timeout_s=config.vlm.timeout_s,
            max_retries=config.vlm.retries,
            request_interval_s=config.vlm.request_interval_s,
            structured_output_mode=config.vlm.structured_output_mode,
        ),
        SceneRetriever(embedding),
        FFmpegGroundingFrameExtractor(config.vlm.max_image_dimension),
        artifacts_dir=config.resolve(config.paths.artifacts_dir),
        candidate_limit=grounding.candidate_limit,
        candidate_padding_s=grounding.candidate_padding_s,
        coarse_fps=grounding.coarse_fps,
        refine_fps=grounding.refine_fps,
        refine_margin_s=grounding.refine_margin_s,
        max_coarse_frames=grounding.max_coarse_frames,
        max_refine_frames=grounding.max_refine_frames,
        max_parallel=grounding.max_parallel,
    )


def create_sound_agent(config: AppConfig) -> SoundAgent:
    sound = config.sound
    provider = LocalClapAudioTextEmbeddingProvider(
        sound.embedding_model,
        device=sound.device,
    )
    return SoundAgent(
        DirectoryMusicLibrary(config.resolve(config.paths.music_dir)),
        provider,
        artifacts_dir=config.resolve(config.paths.artifacts_dir),
        analysis_window_s=sound.analysis_window_s,
        analysis_windows_per_track=sound.analysis_windows_per_track,
        gain_db=sound.gain_db,
        duck_under_voice_db=sound.duck_under_voice_db,
        require_music_index=True,
    )


def create_music_index_builder(config: AppConfig) -> LocalMusicIndexBuilder:
    sound = config.sound
    provider = LocalClapAudioTextEmbeddingProvider(
        sound.embedding_model,
        device=sound.device,
    )
    return LocalMusicIndexBuilder(
        DirectoryMusicLibrary(config.resolve(config.paths.music_dir)),
        provider,
        model_name=sound.embedding_model,
        analysis_window_s=sound.analysis_window_s,
        analysis_windows_per_track=sound.analysis_windows_per_track,
    )


def create_review_agent(config: AppConfig) -> ReviewAgent:
    return ReviewAgent(
        OpenAICompatibleReviewModel(
            model=config.vlm.model,
            base_url=config.vlm.base_url,
            api_key_env=config.vlm.api_key_env,
            max_tokens=config.vlm.max_tokens,
            timeout_s=config.vlm.timeout_s,
            max_retries=config.vlm.retries,
            structured_output_mode=config.vlm.structured_output_mode,
        ),
        FFmpegReviewFrameExtractor(),
        artifacts_dir=config.resolve(config.paths.artifacts_dir),
        max_frames=config.review.max_frames,
        min_voice_coverage=config.edit.min_voice_coverage,
        max_voice_coverage=config.edit.max_voice_coverage,
        max_freeze_per_clip_s=config.edit.max_freeze_per_clip_s,
    )


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
        max_parallel=config.llm.scene_tagger_max_parallel,
        request_interval_s=config.llm.request_interval_s,
        structured_output_mode=config.llm.structured_output_mode,
    )
    story_structure_model = OpenAICompatibleStoryStructureModel(
        model=config.llm.story_structure_model,
        base_url=config.llm.base_url,
        api_key_env=config.llm.api_key_env,
        max_tokens=config.llm.story_structure_max_tokens,
        timeout_s=config.llm.timeout_s,
        max_retries=config.llm.retries,
        max_scenes_per_chunk=config.llm.story_structure_max_scenes_per_chunk,
        structured_output_mode=config.llm.structured_output_mode,
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
