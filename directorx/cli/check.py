from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import shutil
from pathlib import Path

from directorx.config import AppConfig

REQUIRED_PACKAGES = {
    "pydantic": "pydantic",
    "edge-tts": "edge_tts",
    "openai": "openai",
    "numpy": "numpy",
    "Pillow": "PIL",
    "scenedetect": "scenedetect",
    "sentence-transformers": "sentence_transformers",
    "torch": "torch",
    "transformers": "transformers",
    "langgraph": "langgraph",
    "langgraph-checkpoint-sqlite": "langgraph.checkpoint.sqlite",
}


def _check(args: argparse.Namespace) -> int:
    config = AppConfig.load(args.config)
    failures: list[str] = []

    for distribution, module in REQUIRED_PACKAGES.items():
        try:
            if importlib.util.find_spec(module) is None:
                raise ModuleNotFoundError
            importlib.import_module(module)
        except Exception:
            failures.append(f"missing Python package: {distribution}")

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            failures.append(f"missing executable on PATH: {executable}")

    video = args.video.resolve()
    if not video.is_file():
        failures.append(f"video does not exist: {video}")

    subtitle = (
        config.resolve(config.transcription.subtitle_path)
        if config.transcription.subtitle_path is not None
        else None
    )
    if subtitle is not None and not subtitle.is_file():
        failures.append(f"configured subtitle file does not exist: {subtitle}")

    music_dir = config.resolve(config.paths.music_dir)
    music_files = (
        [
            path
            for path in music_dir.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        if music_dir.is_dir()
        else []
    )
    if not music_files:
        failures.append(f"music directory has no audio files: {music_dir}")

    for env_name in (config.vlm.api_key_env, config.llm.api_key_env):
        if not os.environ.get(env_name):
            failures.append(f"missing API key environment variable: {env_name}")

    music_index = args.music_index
    if music_index is None:
        music_index = config.resolve(config.paths.artifacts_dir) / "music-index.json"
    if not music_index.is_file():
        failures.append(f"music index does not exist: {music_index.resolve()}")

    print(f"VIDEO={video}")
    print(f"MUSIC_FILES={len(music_files)}")
    print(f"MUSIC_INDEX={music_index.resolve()}")
    if failures:
        print("STATUS=BLOCKED")
        for failure in failures:
            print(f"BLOCKER={failure}")
        return 1
    print("STATUS=READY")
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="python -m directorx.cli.check",
        description="Check prerequisites for a real DirectorX video run",
    )
    value.add_argument("--config", type=Path, default=Path("config.toml"))
    value.add_argument("--video", required=True, type=Path)
    value.add_argument("--music-index", type=Path)
    return value


def main() -> None:
    raise SystemExit(_check(parser().parse_args()))


if __name__ == "__main__":
    main()
