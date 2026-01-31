from __future__ import annotations

from directorx.core.models import StorySummary, TimeRange, VideoIndex


def validate_story_summary(index: VideoIndex, summary: StorySummary) -> StorySummary:
    """Validate LLM hierarchy references and derive trusted time ranges."""
    if not summary.title.strip() or not summary.short_summary.strip():
        raise ValueError("Story summary title and short_summary are required")

    scenes = {scene.id: scene for scene in index.scenes}
    scene_order = {scene.id: position for position, scene in enumerate(index.scenes)}
    if not scenes:
        raise ValueError("Cannot summarize an index with no scenes")

    sequences_by_id = {}
    claimed_scenes: set[str] = set()
    normalized_sequences = []
    for sequence in summary.sequences:
        if sequence.id in sequences_by_id:
            raise ValueError(f"Duplicate sequence id: {sequence.id}")
        if not sequence.title.strip() or not sequence.short_summary.strip():
            raise ValueError(f"Sequence {sequence.id} requires a short_summary")
        if not sequence.scene_ids:
            raise ValueError(f"Sequence {sequence.id} has no source scenes")
        if any(scene_id not in scenes for scene_id in sequence.scene_ids):
            raise ValueError(f"Sequence {sequence.id} references an unknown scene")
        if claimed_scenes.intersection(sequence.scene_ids):
            raise ValueError("A scene is assigned to multiple sequences")
        ordered_ids = sorted(sequence.scene_ids, key=scene_order.__getitem__)
        positions = [scene_order[scene_id] for scene_id in ordered_ids]
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(f"Sequence {sequence.id} must contain contiguous scenes")
        claimed_scenes.update(ordered_ids)
        sequences_by_id[sequence.id] = sequence
        normalized_sequences.append(
            sequence.model_copy(
                update={
                    "scene_ids": ordered_ids,
                    "source_range": _range_for_scenes(ordered_ids, scenes),
                }
            )
        )

    missing_scenes = set(scenes) - claimed_scenes
    if missing_scenes:
        raise ValueError(
            "Story hierarchy omitted scenes: "
            + ", ".join(sorted(missing_scenes, key=scene_order.__getitem__))
        )

    acts_by_id = {}
    claimed_sequences: set[str] = set()
    normalized_acts = []
    normalized_by_id = {item.id: item for item in normalized_sequences}
    for act in summary.acts:
        if act.id in acts_by_id:
            raise ValueError(f"Duplicate act id: {act.id}")
        if not act.title.strip() or not act.short_summary.strip():
            raise ValueError(f"Act {act.id} requires a short_summary")
        if not act.sequence_ids:
            raise ValueError(f"Act {act.id} has no sequences")
        if any(sequence_id not in normalized_by_id for sequence_id in act.sequence_ids):
            raise ValueError(f"Act {act.id} references an unknown sequence")
        if claimed_sequences.intersection(act.sequence_ids):
            raise ValueError("A sequence is assigned to multiple acts")
        ordered_ids = sorted(
            act.sequence_ids,
            key=lambda sequence_id: scene_order[
                normalized_by_id[sequence_id].scene_ids[0]
            ],
        )
        claimed_sequences.update(ordered_ids)
        source_scene_ids = [
            scene_id
            for sequence_id in ordered_ids
            for scene_id in normalized_by_id[sequence_id].scene_ids
        ]
        normalized_acts.append(
            act.model_copy(
                update={
                    "sequence_ids": ordered_ids,
                    "source_scene_ids": source_scene_ids,
                    "source_range": _range_for_scenes(source_scene_ids, scenes),
                }
            )
        )

    if set(normalized_by_id) != claimed_sequences:
        raise ValueError("Story hierarchy omitted sequences from acts")

    valid_scene_ids = set(scenes)
    for arc in summary.character_arcs:
        if not arc.character.strip() or not arc.short_summary.strip():
            raise ValueError("Character arcs require a character and short_summary")
        if any(scene_id not in valid_scene_ids for scene_id in arc.source_scene_ids):
            raise ValueError(
                f"Character arc {arc.character} references an unknown scene"
            )
    for event in summary.major_events:
        if not event.id.strip() or not event.short_summary.strip():
            raise ValueError(f"Event {event.id} requires a short_summary")
        if any(scene_id not in valid_scene_ids for scene_id in event.source_scene_ids):
            raise ValueError(f"Event {event.id} references an unknown scene")

    return summary.model_copy(
        update={"sequences": normalized_sequences, "acts": normalized_acts}
    )


def _range_for_scenes(scene_ids: list[str], scenes: dict) -> TimeRange:
    selected = [scenes[scene_id] for scene_id in scene_ids]
    return TimeRange(
        start_s=min(scene.source_range.start_s for scene in selected),
        end_s=max(scene.source_range.end_s for scene in selected),
    )
