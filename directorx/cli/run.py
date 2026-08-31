from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from directorx.bootstrap import create_director
from directorx.config import AppConfig
from directorx.workflow import DirectorWorkflow


async def _run(args: argparse.Namespace) -> None:
    config = AppConfig.load(args.config)
    artifacts_dir = config.resolve(config.paths.artifacts_dir)
    project_id = args.project_id or args.video.stem
    project_artifacts_dir = artifacts_dir / project_id
    music_index = args.music_index
    if music_index is None:
        candidate = artifacts_dir / "music-index.json"
        music_index = candidate if candidate.is_file() else None
    director = create_director(
        config,
        coordination_dir=artifacts_dir.parent / "coordination" / project_id,
    )
    workflow = DirectorWorkflow(
        director,
        artifacts_dir=project_artifacts_dir,
        checkpoint_path=args.checkpoint,
        max_revisions=args.max_revisions,
    )
    brief = args.brief
    if args.brief_file is not None:
        brief = args.brief_file.read_text(encoding="utf-8")
    state = await workflow.run(
        project_id=project_id,
        video_path=args.video,
        brief=brief,
        constraints=args.constraint,
        target_duration_s=args.target_duration,
        music_index_path=music_index,
    )
    print(f"PROJECT_ID={state['project_id']}")
    print(f"CURRENT_STAGE={state.get('current_stage', '')}")
    print(f"STATUSES={state.get('statuses', {})}")
    print(f"FINAL_VIDEO={state.get('artifacts', {}).get('rendered-video', '')}")
    if state.get("error"):
        raise SystemExit(state["error"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="python -m directorx.cli.run",
        description="Run the DirectorX LangGraph video editing workflow",
    )
    value.add_argument("--config", type=Path, default=Path("config.toml"))
    value.add_argument("--video", required=True, type=Path)
    brief = value.add_mutually_exclusive_group(required=True)
    brief.add_argument("--brief")
    brief.add_argument("--brief-file", type=Path)
    value.add_argument("--project-id")
    value.add_argument("--constraint", action="append", default=[])
    value.add_argument("--target-duration", type=float, default=60.0)
    value.add_argument("--music-index", type=Path)
    value.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts") / "directorx-checkpoints.sqlite3",
    )
    value.add_argument(
        "--max-revisions",
        type=int,
        default=0,
        help="Reserved for review revision routing; currently only 0 is supported",
    )
    return value


def main() -> None:
    asyncio.run(_run(parser().parse_args()))


if __name__ == "__main__":
    main()
