from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path

from directorx.core.models import (
    Scene,
    SceneInspection,
    SceneSearchHit,
    TimeRange,
    VideoIndex,
)
from directorx.core.ports import EmbeddingProvider


def scene_document(scene: Scene) -> str:
    values = [
        scene.caption,
        scene.dense_caption,
        scene.transcript,
        " ".join(scene.tags),
        " ".join(scene.characters),
        " ".join(scene.actions),
        scene.location or "",
        " ".join(scene.objects),
    ]
    return "\n".join(value for value in values if value)


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    words.update(cjk[index : index + 2] for index in range(max(0, len(cjk) - 1)))
    words.update(cjk)
    return {word for word in words if word}


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")
    if not left:
        raise ValueError("Embedding vectors cannot be empty")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class SceneSearchStore:
    """SQLite FTS5 plus dense-vector reranking for one source movie."""

    def __init__(self, path: Path, embedding_provider: EmbeddingProvider) -> None:
        self.path = path
        self.embedding_provider = embedding_provider

    def build(self, index: VideoIndex, embeddings: list[list[float]]) -> None:
        if len(index.scenes) != len(embeddings):
            raise ValueError("Every scene must have exactly one embedding")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            file_descriptor, raw_path = tempfile.mkstemp(
                dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
            )
            os.close(file_descriptor)
            temporary = Path(raw_path)
            with sqlite3.connect(temporary) as connection:
                connection.executescript(
                    """
                    PRAGMA journal_mode=OFF;
                    PRAGMA synchronous=FULL;
                    CREATE TABLE scenes (
                        scene_id TEXT PRIMARY KEY,
                        start_s REAL NOT NULL,
                        end_s REAL NOT NULL,
                        caption TEXT NOT NULL,
                        transcript TEXT NOT NULL,
                        document TEXT NOT NULL,
                        scene_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL
                    );
                    CREATE TABLE metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE scenes_fts USING fts5(
                        scene_id UNINDEXED,
                        caption,
                        transcript,
                        document,
                        tokenize='unicode61'
                    );
                    """
                )
                for scene, embedding in zip(index.scenes, embeddings, strict=True):
                    document = scene_document(scene)
                    connection.execute(
                        "INSERT INTO scenes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            scene.id,
                            scene.source_range.start_s,
                            scene.source_range.end_s,
                            scene.caption,
                            scene.transcript,
                            document,
                            scene.model_dump_json(),
                            json.dumps(embedding),
                        ),
                    )
                    connection.execute(
                        "INSERT INTO scenes_fts VALUES (?, ?, ?, ?)",
                        (scene.id, scene.caption, scene.transcript, document),
                    )
                connection.execute(
                    "INSERT INTO metadata VALUES ('embedding_dimension', ?)",
                    (str(self.embedding_provider.dimension),),
                )
                connection.execute(
                    "CREATE INDEX scenes_time_range ON scenes(start_s, end_s)"
                )
                connection.commit()
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        dialogue_only: bool = False,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[SceneSearchHit]:
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        if limit <= 0:
            raise ValueError("Search limit must be positive")
        query_vector = (await self.embedding_provider.embed([query]))[0]
        query_tokens = _tokens(query)

        clauses = []
        parameters: list[object] = []
        if start_s is not None:
            clauses.append("end_s > ?")
            parameters.append(start_s)
        if end_s is not None:
            clauses.append("start_s < ?")
            parameters.append(end_s)
        if dialogue_only:
            clauses.append("transcript != ''")
        sql = "SELECT * FROM scenes"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY start_s"

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            expected_dimension = int(
                connection.execute(
                    "SELECT value FROM metadata WHERE key = 'embedding_dimension'"
                ).fetchone()[0]
            )
            if len(query_vector) != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: index={expected_dimension}, "
                    f"query={len(query_vector)}"
                )
            rows = list(connection.execute(sql, parameters))
            fts_scores = self._fts_scores(
                connection, query, dialogue_only=dialogue_only
            )

        scored: list[SceneSearchHit] = []
        for row in rows:
            searchable = row["transcript"] if dialogue_only else row["document"]
            candidate_tokens = _tokens(searchable)
            matched_terms = sorted(query_tokens & candidate_tokens)
            lexical = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
            if query.lower() in searchable.lower():
                lexical = max(lexical, 1.0)
            semantic = max(
                0.0, _cosine(query_vector, json.loads(row["embedding_json"]))
            )
            fts = fts_scores.get(row["scene_id"], 0.0)
            score = min(1.0, semantic * 0.60 + lexical * 0.25 + fts * 0.15)
            matched_by = []
            if semantic > 0.15:
                matched_by.append("semantic")
            if lexical > 0:
                matched_by.append("dialogue" if dialogue_only else "lexical")
            if fts > 0:
                matched_by.append("fts")
            scored.append(
                SceneSearchHit(
                    scene_id=row["scene_id"],
                    source_range=TimeRange(start_s=row["start_s"], end_s=row["end_s"]),
                    score=score,
                    caption=row["caption"],
                    transcript=row["transcript"],
                    matched_by=matched_by,
                    matched_terms=matched_terms[:24],
                )
            )
        return sorted(scored, key=lambda hit: (-hit.score, hit.source_range.start_s))[
            :limit
        ]

    @staticmethod
    def _fts_scores(
        connection: sqlite3.Connection, query: str, *, dialogue_only: bool
    ) -> dict[str, float]:
        terms = sorted(_tokens(query))
        if not terms:
            return {}
        expression = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        if dialogue_only:
            expression = f"transcript : ({expression})"
        matches = connection.execute(
            "SELECT scene_id, bm25(scenes_fts) AS rank "
            "FROM scenes_fts WHERE scenes_fts MATCH ? ORDER BY rank LIMIT 100",
            (expression,),
        ).fetchall()
        if not matches:
            return {}
        ranks = [float(row[1]) for row in matches]
        best = min(ranks)
        worst = max(ranks)
        if math.isclose(best, worst):
            return {row[0]: 1.0 for row in matches}
        return {
            row[0]: 1.0 - (float(row[1]) - best) / (worst - best) for row in matches
        }

    def inspect(self, scene_id: str) -> SceneInspection:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT scene_json, start_s FROM scenes WHERE scene_id = ?", (scene_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown scene id: {scene_id}")
            previous = connection.execute(
                "SELECT scene_id FROM scenes WHERE start_s < ? "
                "ORDER BY start_s DESC LIMIT 1",
                (row[1],),
            ).fetchone()
            following = connection.execute(
                "SELECT scene_id FROM scenes WHERE start_s > ? "
                "ORDER BY start_s LIMIT 1",
                (row[1],),
            ).fetchone()
        return SceneInspection(
            scene=Scene.model_validate_json(row[0]),
            previous_scene_id=previous[0] if previous else None,
            next_scene_id=following[0] if following else None,
        )
