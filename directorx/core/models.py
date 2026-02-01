from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

Seconds = Annotated[float, Field(ge=0)]


class TimeRange(BaseModel):
    start_s: Seconds
    end_s: Seconds

    @model_validator(mode="after")
    def end_follows_start(self) -> TimeRange:
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class Keyframe(BaseModel):
    timestamp_s: Seconds
    path: Path


class DialogueWord(BaseModel):
    text: str
    start_s: Seconds
    end_s: Seconds
    probability: float | None = Field(default=None, ge=0, le=1)


class DialogueLine(BaseModel):
    text: str
    start_s: Seconds
    end_s: Seconds
    speaker: str | None = None
    language: str | None = None
    words: list[DialogueWord] = Field(default_factory=list)


class Shot(BaseModel):
    id: str
    source_range: TimeRange
    dialogue: list[DialogueLine] = Field(default_factory=list)
    keyframes: list[Keyframe] = Field(default_factory=list)


class SceneTags(BaseModel):
    caption: str
    short_summary: str
    tags: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    location: str | None = None
    objects: list[str] = Field(default_factory=list)


class Scene(BaseModel):
    id: str
    source_range: TimeRange
    caption: str
    short_summary: str = ""
    shots: list[Shot] = Field(default_factory=list)
    dense_caption: str = ""
    transcript: str = ""
    dialogue: list[DialogueLine] = Field(default_factory=list)
    keyframes: list[Keyframe] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    location: str | None = None
    objects: list[str] = Field(default_factory=list)


class VideoIndex(BaseModel):
    video_path: Path
    duration_s: Seconds
    scenes: list[Scene]
    index_version: int = 1
    search_db_path: Path | None = None


class StorySequence(BaseModel):
    id: str
    title: str
    short_summary: str
    scene_ids: list[str]
    source_range: TimeRange | None = None


class StoryAct(BaseModel):
    id: str
    title: str
    short_summary: str
    sequence_ids: list[str]
    source_scene_ids: list[str] = Field(default_factory=list)
    source_range: TimeRange | None = None


class CharacterArc(BaseModel):
    character: str
    short_summary: str
    source_scene_ids: list[str] = Field(default_factory=list)


class MajorEvent(BaseModel):
    id: str
    short_summary: str
    source_scene_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class StorySummary(BaseModel):
    title: str
    short_summary: str
    sequences: list[StorySequence]
    acts: list[StoryAct]
    character_arcs: list[CharacterArc] = Field(default_factory=list)
    major_events: list[MajorEvent] = Field(default_factory=list)


class HierarchySearchHit(BaseModel):
    node_id: str
    node_type: str
    parent_id: str | None = None
    source_range: TimeRange
    score: float = Field(ge=0, le=1)
    short_summary: str


class SceneSearchHit(BaseModel):
    scene_id: str
    source_range: TimeRange
    score: float = Field(ge=0, le=1)
    caption: str
    transcript: str = ""
    matched_by: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


class SceneInspection(BaseModel):
    scene: Scene
    previous_scene_id: str | None = None
    next_scene_id: str | None = None


class ScreenplayBeat(BaseModel):
    id: str
    purpose: str
    story_content: str
    visual_intent: str
    mood: str
    target_duration_s: float = Field(gt=0)
    source_sequence_ids: list[str]


class Screenplay(BaseModel):
    title: str
    logline: str
    narrative_angle: str
    beats: list[ScreenplayBeat]
    target_duration_s: float = Field(gt=0)


class BeatNarration(BaseModel):
    beat_id: str
    narration: str
    evidence_scene_ids: list[str]


class NarrationDraft(BaseModel):
    beats: list[BeatNarration]


class ScreenwriterSceneEvidence(BaseModel):
    scene_id: str
    short_summary: str
    caption: str
    tags: list[str] = Field(default_factory=list)


class StoryBeat(BaseModel):
    id: str
    purpose: str
    story_content: str
    narration: str
    visual_intent: str
    mood: str
    target_duration_s: float = Field(gt=0)
    source_sequence_ids: list[str]
    evidence_scene_ids: list[str]


class Storyboard(BaseModel):
    title: str
    logline: str
    narrative_angle: str
    beats: list[StoryBeat]
    target_duration_s: float = Field(gt=0)


class NarrationSegment(BaseModel):
    beat_id: str
    text: str
    audio_path: Path
    duration_s: float = Field(gt=0)


class ShotRequest(BaseModel):
    id: str
    beat_id: str
    narration_text: str
    visual_query: str
    mood: str
    target_duration_s: float = Field(gt=0)


class CandidateScore(BaseModel):
    scene_id: str
    score: float = Field(ge=0, le=1)
    rationale: str


class GroundedClip(BaseModel):
    shot_id: str
    beat_id: str
    source_range: TimeRange
    target_duration_s: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    rationale: str


class MusicTrack(BaseModel):
    path: Path
    title: str
    tags: list[str] = Field(default_factory=list)
    duration_s: float = Field(gt=0)


class SoundPlan(BaseModel):
    track: MusicTrack
    gain_db: float = -23.0
    duck_under_voice_db: float = -9.0


class RenderPlan(BaseModel):
    source_video: Path
    clips: list[GroundedClip]
    narration: list[NarrationSegment]
    sound: SoundPlan
    output_path: Path
    target_duration_s: float = Field(gt=0)
    subtitle_path: Path | None = None

    @property
    def duration_s(self) -> float:
        return sum(clip.target_duration_s for clip in self.clips)
