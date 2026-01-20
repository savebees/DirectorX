from __future__ import annotations

import math

from directorx.core.models import Scene, Shot, TimeRange
from directorx.core.ports import ShotVisualEmbedder


class VisualSceneGrouper:
    """Merge adjacent shots whose visual embeddings remain semantically close."""

    def __init__(
        self,
        visual_embedder: ShotVisualEmbedder,
        *,
        similarity_threshold: float = 0.8,
        max_scene_duration_s: float = 15.0,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [0, 1]")
        if max_scene_duration_s <= 0:
            raise ValueError("max_scene_duration_s must be positive")
        self.visual_embedder = visual_embedder
        self.similarity_threshold = similarity_threshold
        self.max_scene_duration_s = max_scene_duration_s

    async def group(self, shots: list[Shot]) -> list[Scene]:
        if not shots:
            return []
        vectors = await self.visual_embedder.embed(shots)
        if len(vectors) != len(shots):
            raise ValueError("Visual embedder returned the wrong number of vectors")
        if any(not vector for vector in vectors):
            raise ValueError("Visual embedder returned an empty vector")

        groups: list[list[Shot]] = []
        current: list[Shot] = [shots[0]]
        for index in range(1, len(shots)):
            previous = vectors[index - 1]
            candidate = vectors[index]
            same_scene = (
                _cosine(previous, candidate) >= self.similarity_threshold
                and self._duration_with(current, shots[index])
                <= self.max_scene_duration_s
            )
            if same_scene:
                current.append(shots[index])
            else:
                groups.append(current)
                current = [shots[index]]
        groups.append(current)

        return [self._build_scene(index, group) for index, group in enumerate(groups)]

    def _duration_with(self, current: list[Shot], candidate: Shot) -> float:
        return candidate.source_range.end_s - current[0].source_range.start_s

    @staticmethod
    def _build_scene(index: int, shots: list[Shot]) -> Scene:
        start_s = shots[0].source_range.start_s
        end_s = shots[-1].source_range.end_s
        return Scene(
            id=f"scene-{index:05d}",
            source_range=TimeRange(start_s=start_s, end_s=end_s),
            caption="",
            shots=shots,
            dialogue=[line for shot in shots for line in shot.dialogue],
            keyframes=[frame for shot in shots for frame in shot.keyframes],
        )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Visual embedding vectors must have the same dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
