from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from directorx.core.models import VideoIndex


@dataclass(frozen=True)
class CacheDecision:
    fingerprint: str
    index_dir: Path
    cached_index: VideoIndex | None
    content_was_hashed: bool


class VideoIndexCache:
    """Content-addressed index cache with a cheap path/stat fast path."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.sources_dir = self.root / "sources"
        self.indexes_dir = self.root / "indexes"

    def resolve(self, video_path: Path) -> CacheDecision:
        source = video_path.resolve()
        stat = source.stat()
        registry_path = self._registry_path(source)
        registry = self._read_json(registry_path)

        if registry and self._same_stat(registry, stat):
            fingerprint = str(registry.get("content_fingerprint", ""))
            cached = self._load_index(fingerprint, source)
            if cached is not None:
                return CacheDecision(
                    fingerprint=fingerprint,
                    index_dir=self.indexes_dir / fingerprint,
                    cached_index=cached,
                    content_was_hashed=False,
                )

        fingerprint = self._sha256(source)
        cached = self._load_index(fingerprint, source)
        decision = CacheDecision(
            fingerprint=fingerprint,
            index_dir=self.indexes_dir / fingerprint,
            cached_index=cached,
            content_was_hashed=True,
        )
        if cached is not None:
            self._write_registry(source, stat, fingerprint)
        return decision

    def commit(self, video_path: Path, index: VideoIndex) -> Path:
        if not index.content_fingerprint:
            raise ValueError("Cannot cache an index without a content fingerprint")
        source = video_path.resolve()
        index_dir = self.indexes_dir / index.content_fingerprint
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.json"
        self._write_json_atomic(index_path, index.model_dump(mode="json"))
        self._write_registry(source, source.stat(), index.content_fingerprint)
        return index_path

    def _load_index(self, fingerprint: str, source: Path) -> VideoIndex | None:
        if not fingerprint:
            return None
        index_path = self.indexes_dir / fingerprint / "index.json"
        payload = self._read_json(index_path)
        if payload is None:
            return None
        payload["video_path"] = str(source)
        payload["search_db_path"] = str(index_path.parent / "search.sqlite3")
        return VideoIndex.model_validate(payload)

    def _registry_path(self, source: Path) -> Path:
        path_id = hashlib.sha256(os.fsencode(str(source))).hexdigest()[:24]
        return self.sources_dir / f"{path_id}.json"

    def _write_registry(
        self, source: Path, stat: os.stat_result, fingerprint: str
    ) -> None:
        payload = {
            "source_path": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "content_fingerprint": fingerprint,
        }
        self._write_json_atomic(self._registry_path(source), payload)

    @staticmethod
    def _same_stat(payload: dict[str, Any], stat: os.stat_result) -> bool:
        return (
            payload.get("size") == stat.st_size
            and payload.get("mtime_ns") == stat.st_mtime_ns
        )

    @staticmethod
    def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

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
