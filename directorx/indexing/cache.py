from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from directorx.core.models import VideoIndex


@dataclass(frozen=True)
class CacheDecision:
    video_name: str
    index_dir: Path
    cached_index: VideoIndex | None


class VideoIndexCache:
    """Look up indexes by source filename and keep their files together."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.registry_path = self.root / "registry.json"
        self.indexes_dir = self.root / "indexes"

    def resolve(self, video_path: Path) -> CacheDecision:
        source = video_path.resolve()
        video_name = source.name
        registry = self._read_json(self.registry_path) or {}
        index_dir = self.indexes_dir / video_name
        registered_dir = registry.get(video_name, {}).get("index_dir")
        if registered_dir:
            index_dir = self.root / registered_dir
        cached = self._load_index(index_dir, source)
        return CacheDecision(
            video_name=video_name,
            index_dir=index_dir,
            cached_index=cached,
        )

    def commit(self, video_path: Path, index: VideoIndex) -> Path:
        source = video_path.resolve()
        video_name = source.name
        index_dir = self.indexes_dir / video_name
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.json"
        self._write_json_atomic(index_path, index.model_dump(mode="json"))

        registry = self._read_json(self.registry_path) or {}
        registry[video_name] = {
            "index_dir": str(index_dir.relative_to(self.root)),
            "index_version": index.index_version,
        }
        self._write_json_atomic(self.registry_path, registry)
        return index_path

    def _load_index(self, index_dir: Path, source: Path) -> VideoIndex | None:
        index_path = index_dir / "index.json"
        search_path = index_dir / "search.sqlite3"
        payload = self._read_json(index_path)
        if payload is None or not search_path.exists():
            return None
        payload["video_path"] = str(source)
        payload["search_db_path"] = str(search_path)
        return VideoIndex.model_validate(payload)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return value

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
