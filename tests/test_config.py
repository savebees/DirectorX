from pathlib import Path

from directorx.config import AppConfig


def test_project_config_loads_from_one_entrypoint() -> None:
    config = AppConfig.load(Path("config.toml"))

    assert config.vlm.provider == "openai-compatible"
    assert config.vlm.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.vlm.max_vlm_frames_per_scene == 8
    assert config.llm.screenwriter_model == "gpt-5.6-luna"
    assert config.transcription.provider == "auto"
    assert config.resolve(config.paths.artifacts_dir).name == "artifacts"
    assert config.render.dimensions == (1920, 1080)
