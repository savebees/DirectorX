from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from directorx.bootstrap import create_music_index_builder
from directorx.config import AppConfig


async def _run(args: argparse.Namespace) -> None:
    config = AppConfig.load(args.config)
    builder = create_music_index_builder(config)
    destination = args.output or config.resolve(config.paths.artifacts_dir)
    path = await builder.build_and_persist(destination)
    print(f"MUSIC_INDEX={path}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="python -m directorx.cli.music_index",
        description="Build a persistent CLAP index for the configured music library",
    )
    value.add_argument("--config", type=Path, default=Path("config.toml"))
    value.add_argument("--output", type=Path)
    return value


def main() -> None:
    asyncio.run(_run(parser().parse_args()))


if __name__ == "__main__":
    main()
