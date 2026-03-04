from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from directorx.core.models import (
    CharacterArc,
    MajorEvent,
    MusicTrack,
    NarrationDraft,
    Scene,
    SceneTags,
    Screenplay,
    ScreenwriterSceneEvidence,
    StoryAct,
    StorySequence,
    StorySummary,
    VideoIndex,
)


class _SequenceDraft(BaseModel):
    title: str
    short_summary: str
    scene_ids: list[str]


class _SequenceBatch(BaseModel):
    sequences: list[_SequenceDraft]


class _ActDraft(BaseModel):
    title: str
    short_summary: str
    sequence_ids: list[str]


class _CharacterArcDraft(BaseModel):
    character: str
    short_summary: str
    source_scene_ids: list[str]


class _MajorEventDraft(BaseModel):
    id: str
    short_summary: str
    source_scene_ids: list[str]
    confidence: float = Field(ge=0, le=1)


class _StoryStructureDraft(BaseModel):
    title: str
    short_summary: str
    acts: list[_ActDraft]
    character_arcs: list[_CharacterArcDraft]
    major_events: list[_MajorEventDraft]


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


class OpenAICompatibleScreenwriterModel:
    """Structured text planning against an OpenAI-compatible model endpoint."""

    SCREENPLAY_SYSTEM_PROMPT = (
        "You are a professional screenwriter. Your task is to adapt the Director's "
        "creative brief and the source film's story into a coherent screenplay for "
        "a video edit. Choose the narrative angle, organize the story into dramatic "
        "beats, and define what each beat needs to communicate."
    )
    NARRATION_SYSTEM_PROMPT = (
        "You are a professional screenwriter specializing in voice-over scripts. "
        "Your task is to turn the screenplay and its source evidence into polished "
        "narration for each beat. The narration should be natural to speak, "
        "dramatically coherent, and faithful to the source."
    )

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "VYCE_API_KEY",
        max_tokens: int = 4000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if client is None:
            secret = os.environ.get(api_key_env, "")
            if not secret:
                raise RuntimeError(
                    f"Missing planner API key. Set {api_key_env} in the environment"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for online story planning"
                ) from exc
            client = AsyncOpenAI(
                api_key=secret,
                base_url=base_url.rstrip("/"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    async def draft_screenplay(
        self,
        objective: str,
        constraints: list[str],
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> Screenplay:
        request = (
            f"Director's creative brief:\n{objective}\n\n"
            f"Task constraints:\n{self._format_constraints(constraints)}\n\n"
            f"Target duration: {target_duration_s:.1f} seconds.\n\n"
            "Complete source story hierarchy:\n"
            f"{story_summary.model_dump_json(indent=2, exclude_none=True)}"
        )
        return await self._complete(
            self.SCREENPLAY_SYSTEM_PROMPT,
            request,
            Screenplay,
            "screenplay",
        )

    async def draft_narration(
        self,
        objective: str,
        constraints: list[str],
        screenplay: Screenplay,
        evidence_by_beat: dict[str, list[ScreenwriterSceneEvidence]],
    ) -> NarrationDraft:
        evidence = {
            beat_id: [item.model_dump(mode="json") for item in scenes]
            for beat_id, scenes in evidence_by_beat.items()
        }
        request = (
            f"Director's creative brief:\n{objective}\n\n"
            f"Task constraints:\n{self._format_constraints(constraints)}\n\n"
            "Screenplay:\n"
            f"{screenplay.model_dump_json(indent=2)}\n\n"
            "Selected source scene evidence by beat:\n"
            f"{self._json(evidence)}\n\n"
            "Write one narration entry for every screenplay beat. Keep each line "
            "natural for spoken delivery and appropriate to the beat's duration. "
            "Use only the supplied evidence and do not invent facts, motives, or "
            "events. Cite the evidence_scene_ids used for each narration. Every "
            "cited scene must come from that beat's supplied evidence."
        )
        return await self._complete(
            self.NARRATION_SYSTEM_PROMPT,
            request,
            NarrationDraft,
            "narration_draft",
        )

    async def _complete(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        schema_name: str,
    ):
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        )
        if not completion.choices:
            raise ValueError("Screenwriter model returned no choices")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Screenwriter model returned no JSON")
        return schema.model_validate_json(content)

    @staticmethod
    def _format_constraints(constraints: list[str]) -> str:
        return "\n".join(f"- {constraint}" for constraint in constraints) or "- None"

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)


class OpenAICompatibleStoryStructureModel:
    """Build a citation-backed film hierarchy from indexed scene evidence."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "VYCE_API_KEY",
        max_tokens: int = 4000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        max_scenes_per_chunk: int = 24,
        client: Any | None = None,
    ) -> None:
        if max_scenes_per_chunk <= 0:
            raise ValueError("max_scenes_per_chunk must be positive")
        if client is None:
            secret = os.environ.get(api_key_env, "")
            if not secret:
                raise RuntimeError(
                    f"Missing story structure API key. Set {api_key_env} "
                    "in the environment"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for story structure planning"
                ) from exc
            client = AsyncOpenAI(
                api_key=secret,
                base_url=base_url.rstrip("/"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self.model = model
        self.max_tokens = max_tokens
        self.max_scenes_per_chunk = max_scenes_per_chunk
        self._client = client

    async def build(self, video_index: VideoIndex) -> StorySummary:
        if not video_index.scenes:
            raise ValueError("Cannot summarize an index with no scenes")
        sequences: list[StorySequence] = []
        for chunk_number, chunk in enumerate(
            self._chunks(video_index.scenes, self.max_scenes_per_chunk), start=1
        ):
            evidence = self._scene_context(chunk)
            batch = await self._complete(
                "You are a film-footage analyst. Group contiguous indexed scenes "
                "into one or more narrative sequences. Use only the supplied "
                "captions and tags. Return JSON only. "
                "Every input scene must appear exactly once in scene_ids. Do not "
                "invent scene IDs, events, identities, or causes. Keep the source "
                "order and write a concise factual short_summary.",
                f"Scene evidence chunk {chunk_number}:\n{evidence}",
                _SequenceBatch,
            )
            sequences.extend(
                self._normalize_chunk_sequences(batch.sequences, chunk, len(sequences))
            )

        sequence_context = "\n".join(
            f"{sequence.id}: {sequence.short_summary} "
            f"(scenes={','.join(sequence.scene_ids)})"
            for sequence in sequences
        )
        draft = await self._complete(
            "You are a film-footage analyst creating a high-level structure from "
            "already summarized sequences. Group sequences into chronological acts "
            "and write a factual whole-film summary. Return JSON only. Every "
            "sequence must appear exactly once in an act. Character arcs and major "
            "events must cite only the supplied scene IDs. Mark uncertain claims "
            "as uncertain instead of inventing details.",
            f"Sequence summaries:\n{sequence_context}",
            _StoryStructureDraft,
        )
        acts = [
            StoryAct(
                id=f"act-{number:04d}",
                title=act.title,
                short_summary=act.short_summary,
                sequence_ids=act.sequence_ids,
            )
            for number, act in enumerate(draft.acts, start=1)
        ]
        return StorySummary(
            title=draft.title,
            short_summary=draft.short_summary,
            sequences=sequences,
            acts=acts,
            character_arcs=[
                CharacterArc.model_validate(arc.model_dump())
                for arc in draft.character_arcs
            ],
            major_events=[
                MajorEvent.model_validate(event.model_dump())
                for event in draft.major_events
            ],
        )

    async def _complete(self, system: str, user: str, schema: type[BaseModel]):
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=self.max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "story_structure",
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            },
        )
        if not completion.choices:
            raise ValueError("Story structure model returned no choices")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Story structure model returned no JSON")
        return schema.model_validate_json(content)

    @staticmethod
    def _chunks(scenes: list[Scene], size: int) -> list[list[Scene]]:
        return [
            scenes[offset : offset + size] for offset in range(0, len(scenes), size)
        ]

    @staticmethod
    def _scene_context(scenes: list[Scene]) -> str:
        return "\n".join(
            f"{scene.id}: caption={scene.caption!r}; tags={scene.tags}"
            for scene in scenes
        )

    @staticmethod
    def _normalize_chunk_sequences(
        sequences: list[_SequenceDraft],
        chunk: list[Scene],
        sequence_offset: int,
    ) -> list[StorySequence]:
        expected = [scene.id for scene in chunk]
        seen: list[str] = []
        normalized: list[StorySequence] = []
        for sequence in sequences:
            if not sequence.title.strip() or not sequence.short_summary.strip():
                raise ValueError("Sequence title and short_summary are required")
            if not sequence.scene_ids:
                raise ValueError("Sequence must contain source scenes")
            if any(scene_id not in expected for scene_id in sequence.scene_ids):
                raise ValueError("Sequence references a scene outside its chunk")
            seen.extend(sequence.scene_ids)
            normalized.append(
                StorySequence(
                    id=f"sequence-{sequence_offset + len(normalized) + 1:04d}",
                    title=sequence.title,
                    short_summary=sequence.short_summary,
                    scene_ids=sequence.scene_ids,
                )
            )
        if sorted(seen, key=expected.index) != expected or len(seen) != len(set(seen)):
            raise ValueError("Sequence output must cover each chunk scene exactly once")
        return normalized


class OpenAICompatibleSceneTagger:
    """Create an information-rich, searchable record for each scene."""

    SYSTEM_PROMPT = """Create searchable metadata for one video scene.
Use the dense visual caption and transcript as your only evidence.

Write CAPTION as a detailed, self-contained description of the scene. Combine
what is visibly shown with the meaning of relevant dialogue, but do not quote
or reproduce the dialogue. Describe people, actions, interactions, setting,
location, objects, visible text, spatial relationships, and any dialogue-based
context that is supported by the transcript. Use as much concrete detail as
the evidence supports.
Write SHORT_SUMMARY as one concise, self-contained factual sentence capturing
the scene's central action or development. It must remain grounded in the same
evidence and must not merely copy the detailed CAPTION.
Do not invent identities, motives, hidden events, or details absent from the
inputs. Then extract concise searchable labels from the same evidence.

Return exactly these seven lines, in this order:
CAPTION: <information-rich factual caption>
SHORT_SUMMARY: <one concise factual sentence>
TAGS: <short keywords or noun phrases separated by semicolons>
CHARACTERS: <observable people or stable generic labels separated by semicolons>
ACTIONS: <visible actions separated by semicolons>
LOCATION: <location, or none>
OBJECTS: <visible searchable objects separated by semicolons>

Use none for an empty field. Avoid duplicates, vague adjectives, and
speculative labels. Return plain text only: no JSON, markdown, bullets,
headings, or extra commentary."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "VYCE_API_KEY",
        max_tokens: int = 1200,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        secret = os.environ.get(api_key_env, "")
        if client is not None:
            self._client = client
        else:
            if not secret:
                raise RuntimeError(
                    f"Missing tagger API key. Set {api_key_env} in the environment"
                )
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The openai package is required for scene tagging"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=secret,
                base_url=base_url.rstrip("/"),
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self.model = model
        self.max_tokens = max_tokens

    async def tag_batch(self, scenes: list[Scene]) -> dict[str, SceneTags]:
        results = await asyncio.gather(*(self._tag_one(scene) for scene in scenes))
        return {scene_id: tags for scene_id, tags in results}

    async def _tag_one(self, scene: Scene) -> tuple[str, SceneTags]:
        request = (
            f"Scene ID: {scene.id}\n"
            f"Transcript: {scene.transcript or '(none)'}\n"
            f"Dense visual caption: {scene.dense_caption or '(none)'}"
        )
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": request},
            ],
            temperature=0.1,
            max_tokens=self.max_tokens,
        )
        if not completion.choices:
            raise ValueError(f"Scene tagger returned no choices for {scene.id}")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Scene tagger returned no metadata for {scene.id}")
        return scene.id, self._parse_response(content, scene.id)

    @staticmethod
    def _parse_response(content: str, scene_id: str) -> SceneTags:
        fields = (
            "CAPTION",
            "SHORT_SUMMARY",
            "TAGS",
            "CHARACTERS",
            "ACTIONS",
            "LOCATION",
            "OBJECTS",
        )
        values: dict[str, str] = {}
        for line in content.strip().splitlines():
            match = re.fullmatch(
                r"(CAPTION|SHORT_SUMMARY|TAGS|CHARACTERS|ACTIONS|LOCATION|OBJECTS):"
                r"\s*(.*)",
                line,
            )
            if match is None:
                raise ValueError(
                    f"Scene tagger returned an invalid line for {scene_id}"
                )
            name, value = match.groups()
            if name in values:
                raise ValueError(f"Scene tagger repeated {name} for {scene_id}")
            values[name] = value.strip()

        missing = [name for name in fields if name not in values]
        if missing:
            raise ValueError(
                f"Scene tagger omitted fields for {scene_id}: {', '.join(missing)}"
            )
        caption = values["CAPTION"]
        if not caption or caption.lower() == "none":
            raise ValueError(f"Scene tagger returned an empty caption for {scene_id}")
        short_summary = values["SHORT_SUMMARY"]
        if not short_summary or short_summary.lower() == "none":
            raise ValueError(
                f"Scene tagger returned an empty short summary for {scene_id}"
            )

        def split_items(name: str) -> list[str]:
            value = values[name]
            if value.lower() == "none":
                return []
            return [item.strip() for item in value.split(";") if item.strip()]

        location = values["LOCATION"]
        if location.lower() == "none":
            location = None
        return SceneTags(
            caption=caption,
            short_summary=short_summary,
            tags=split_items("TAGS"),
            characters=split_items("CHARACTERS"),
            actions=split_items("ACTIONS"),
            location=location,
            objects=split_items("OBJECTS"),
        )


class EdgeSpeechTTS:
    """Cross-platform Chinese Edge TTS with fail-fast error handling."""

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: int = 185,
    ) -> None:
        self.voice = voice
        self.rate = rate

    async def synthesize(self, text: str, output_path: Path) -> float:
        return await asyncio.to_thread(self._synthesize_sync, text, output_path)

    def _synthesize_sync(self, text: str, output_path: Path) -> float:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._edge_tts(text, output_path)
        duration = _probe_duration(output_path)
        if duration <= 0:
            raise RuntimeError("Edge TTS produced an empty audio file")
        return duration

    def _edge_tts(self, text: str, output_path: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="video-edge-tts-") as temporary:
            source = Path(temporary) / "speech.mp3"
            import edge_tts

            async def save() -> None:
                communicator = edge_tts.Communicate(
                    text, self.voice, rate=self._edge_rate()
                )
                await communicator.save(str(source))

            asyncio.run(save())
            self._convert_to_wav(source, output_path)

    @staticmethod
    def _convert_to_wav(source: Path, output_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ar",
                "48000",
                "-ac",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _edge_rate(self) -> str:
        # Edge accepts signed percentage strings, while the CLI exposes a
        # familiar words-per-minute-like integer for the native engines.
        delta = round((self.rate / 185 - 1) * 100)
        return f"{delta:+d}%"


class DirectoryMusicLibrary:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def tracks(self) -> list[MusicTrack]:
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)
        paths = sorted(
            path
            for path in self.directory.iterdir()
            if path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac"}
        )
        if not paths:
            raise ValueError(f"No supported music files in {self.directory}")
        tracks = []
        for path in paths:
            duration = await asyncio.to_thread(_probe_duration, path)
            tags = [part.lower() for part in re.split(r"[-_\s]+", path.stem) if part]
            tracks.append(
                MusicTrack(path=path, title=path.stem, tags=tags, duration_s=duration)
            )
        return tracks
