from pathlib import Path

from directorx.bootstrap import create_narration_agent
from directorx.config import AppConfig
from directorx.services.providers import EdgeSpeechTTS


def test_project_config_loads_from_one_entrypoint() -> None:
    config = AppConfig.load(Path("config.toml"))

    assert config.vlm.provider == "openai-compatible"
    assert config.vlm.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.vlm.max_vlm_frames_per_scene == 8
    assert config.scene_grouping.model == "clip-ViT-B-32"
    assert config.scene_grouping.similarity_threshold == 0.8
    assert config.scene_grouping.max_scene_duration_s == 15.0
    assert config.llm.screenwriter_model == "gpt-5.6-luna"
    assert config.transcription.provider == "auto"
    assert config.resolve(config.paths.artifacts_dir).name == "artifacts"
    assert config.render.dimensions == (1920, 1080)

    narration = create_narration_agent(config)
    assert isinstance(narration.tts, EdgeSpeechTTS)
    assert narration.tts.voice == "zh-CN-XiaoxiaoNeural"
    assert narration.tts.rate == 185
