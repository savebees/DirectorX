from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from directorx.bootstrap import create_indexing_runtime
from directorx.config import AppConfig


async def _run(args: argparse.Namespace) -> None:
    config = AppConfig.load(args.config)
    index = await create_indexing_runtime(config).indexer.build(args.video)
    print(f"INDEX_FINGERPRINT={index.content_fingerprint}")
    print(f"SCENE_COUNT={len(index.scenes)}")
    print(f"SEARCH_DB={index.search_db_path}")
    print(f"INDEX_JSON={index.search_db_path.parent / 'index.json'}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="python -m directorx.cli.index",
        description="Build a cached scene index for a long-form video",
    )
    value.add_argument("--config", type=Path, default=Path("config.toml"))
    value.add_argument("--video", required=True, type=Path)
    return value


def main() -> None:
    asyncio.run(_run(parser().parse_args()))


if __name__ == "__main__":
    main()
