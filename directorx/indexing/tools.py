from __future__ import annotations

from directorx.core.models import SceneInspection, SceneSearchHit

from .store import SceneSearchStore


class SceneSearchTools:
    """Narrow, typed tools exposed to Storyboard and Grounding agents."""

    def __init__(self, store: SceneSearchStore) -> None:
        self.store = store

    async def search_scenes(
        self,
        query: str,
        *,
        limit: int = 8,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[SceneSearchHit]:
        """Find visually or narratively relevant scenes within optional time bounds."""
        return await self.store.search(
            query,
            limit=limit,
            start_s=start_s,
            end_s=end_s,
        )

    async def search_dialogue(
        self,
        query: str,
        *,
        limit: int = 8,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> list[SceneSearchHit]:
        """Search only spoken dialogue/subtitles and return the containing scenes."""
        return await self.store.search(
            query,
            limit=limit,
            dialogue_only=True,
            start_s=start_s,
            end_s=end_s,
        )

    def inspect_scene(self, scene_id: str) -> SceneInspection:
        """Return full scene metadata plus adjacent scene identifiers."""
        return self.store.inspect(scene_id)
