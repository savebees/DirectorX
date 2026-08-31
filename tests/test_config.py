from pathlib import Path

from directorx.bootstrap import create_narration_agent, create_sound_agent
from directorx.config import AppConfig
from directorx.services.providers import DirectoryMusicLibrary, EdgeSpeechTTS
from directorx.services.sound import LocalClapAudioTextEmbeddingProvider


def test_project_config_loads_from_one_entrypoint() -> None:
    config = AppConfig.load(Path("config.toml"))

    assert config.vlm.provider == "openai-compatible"
    assert config.vlm.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.vlm.max_vlm_frames_per_scene == 8
    assert config.scene_grouping.model == "clip-ViT-B-32"
    assert config.scene_grouping.similarity_threshold == 0.8
    assert config.scene_grouping.max_scene_duration_s == 15.0
    assert config.llm.screenwriter_model == "deepseek-ai/DeepSeek-V3.2"
    assert config.llm.api_key_env == "SILICONFLOW_API_KEY"
    assert config.vlm.api_key_env == "SILICONFLOW_API_KEY"
    assert config.transcription.provider == "auto"
    assert config.grounding.candidate_limit == 4
    assert config.grounding.coarse_fps == 1
    assert config.grounding.refine_fps == 6
    assert config.sound.embedding_model == "laion/larger_clap_music"
    assert config.sound.analysis_windows_per_track == 3
    assert config.review.max_frames == 12
    assert config.resolve(config.paths.artifacts_dir).name == "artifacts"
    assert config.render.dimensions == (1920, 1080)

    narration = create_narration_agent(config)
    assert isinstance(narration.tts, EdgeSpeechTTS)
    assert narration.tts.voice == "zh-CN-XiaoxiaoNeural"
    assert narration.tts.rate == 160

    sound = create_sound_agent(config)
    assert isinstance(sound.music_library, DirectoryMusicLibrary)
    assert isinstance(sound.embedding_provider, LocalClapAudioTextEmbeddingProvider)
    assert sound.embedding_provider.model_name == "laion/larger_clap_music"
    assert sound.analysis_window_s == 10
