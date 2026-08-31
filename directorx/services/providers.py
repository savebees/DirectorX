from __future__ import annotations

import asyncio
import os
import random
import re
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from directorx.core.models import (
    BeatNarration,
    CharacterArc,
    MajorEvent,
    MusicTrack,
    NarrationDraft,
    Scene,
    SceneTags,
    Screenplay,
    ScreenplayBeat,
    ScreenwriterSceneEvidence,
    StoryAct,
    StorySequence,
    StorySummary,
    VideoIndex,
)
from directorx.services.structured_output import (
    StructuredOutputMode,
    request_structured_output,
)


class _SequenceDraft(BaseModel):
    title: str = ""
    short_summary: str
    scene_ids: list[str]


class _SequenceBatch(BaseModel):
    sequences: list[_SequenceDraft]


class _ActDraft(BaseModel):
    title: str = ""
    short_summary: str = ""
    sequence_ids: list[str] = Field(default_factory=list)


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
    title: str = "Untitled"
    short_summary: str = ""
    acts: list[_ActDraft] = Field(default_factory=list)
    character_arcs: list[_CharacterArcDraft] = Field(default_factory=list)
    major_events: list[_MajorEventDraft] = Field(default_factory=list)


class _StructurePartDraft(BaseModel):
    title: str = "Untitled"
    short_summary: str = ""
    acts: list[_ActDraft] = Field(default_factory=list)


class _NarrationRevision(BaseModel):
    beats: list[BeatNarration] = Field(min_length=1)


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
    """Typed screenplay planning against an OpenAI-compatible model endpoint."""

    SCREENPLAY_SYSTEM_PROMPT = (
        "You are a professional screenwriter. Your task is to adapt the Director's "
        "creative brief and the source film's story into a coherent screenplay for "
        "a video edit. Choose the narrative angle, organize the story into dramatic "
        "beats, and define what each beat needs to communicate. Keep every source "
        "sequence ID grounded in the supplied hierarchy. Design 4 to 7 narrative "
        "beats. Each beat may cite 1 to 3 consecutive source_sequence_ids so the "
        "Editor can cut between multiple verified shots. List beats and sequences "
        "in strict source chronology without reusing a sequence, "
        "and never skip an intermediate sequence within a beat (for example, "
        "sequence-0001 plus sequence-0003 must also include sequence-0002). "
        "and make each visual intent describe only "
        "directly visible content from its cited sequence summaries. Do not add shot "
        "types, gestures, objects, text, actions, or transitions that those summaries "
        "do not support. When the creative brief asks for absent footage, adapt to the "
        "closest supported event without claiming that the absent event is visible."
    )
    NARRATION_SYSTEM_PROMPT = (
        "You are a professional screenwriter specializing in voice-over scripts. "
        "Your task is to turn the screenplay and its source evidence into polished "
        "narration for each beat. The narration should be natural to speak, "
        "dramatically coherent, and faithful to the source. Return one narration "
        "record for every screenplay beat and cite only supplied evidence scene IDs."
        " Compose one continuous voice-over with setup, escalation, turn, and payoff, "
        "then return it as ordered beat records without isolated taglines. The "
        "application derives full_narration from those records, so do not duplicate "
        "the same script in full_narration."
    )

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "LLM_API_KEY",
        max_tokens: int = 4000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        fallback_model: str | None = None,
        fallback_base_url: str | None = None,
        fallback_api_key_env: str | None = None,
        structured_output_mode: StructuredOutputMode = "prompted_json",
        fallback_structured_output_mode: StructuredOutputMode = "json_object",
        narration_language: str = "auto",
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
        self._fallback_client = None
        self._fallback_model = fallback_model
        self.structured_output_mode = structured_output_mode
        self.fallback_structured_output_mode = fallback_structured_output_mode
        self.narration_language = narration_language.strip() or "auto"
        if fallback_model and fallback_base_url and fallback_api_key_env:
            secret = os.environ.get(fallback_api_key_env, "")
            if secret:
                try:
                    from openai import AsyncOpenAI
                except ImportError:
                    pass
                else:
                    self._fallback_client = AsyncOpenAI(
                        api_key=secret,
                        base_url=fallback_base_url.rstrip("/"),
                        timeout=timeout_s,
                        max_retries=0,
                    )

    async def draft_screenplay(
        self,
        objective: str,
        constraints: list[str],
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> Screenplay:
        sequence_lines = "\n".join(
            f"{item.id}: {item.short_summary}; scenes={','.join(item.scene_ids)}; "
            f"duration_s={item.source_range.duration_s:.1f}"
            if item.source_range is not None
            else f"{item.id}: {item.short_summary}; scenes={','.join(item.scene_ids)}"
            for item in story_summary.sequences
        )
        act_lines = "\n".join(
            f"{item.id}: {item.title} - {item.short_summary}; "
            f"sequences={','.join(item.sequence_ids)}"
            for item in story_summary.acts
        )
        request = (
            f"Director's creative brief:\n{objective}\n\n"
            f"Task constraints:\n{self._format_constraints(constraints)}\n\n"
            f"Target duration: {target_duration_s:.1f} seconds.\n\n"
            f"Film title: {story_summary.title}\n"
            f"Film summary: {story_summary.short_summary}\n"
            f"Acts:\n{act_lines}\n\n"
            f"Sequences:\n{sequence_lines}"
            "\n\nUse 4 to 7 narrative beats. Prefer 2 or 3 consecutive source "
            "sequences in a beat when they provide useful visual variety. Reserve "
            "only 6 to 8 seconds for the final title reveal. A beat's target duration "
            "must not exceed the summed duration_s of its selected sequences."
        )
        screenplay = await self._complete(
            self.SCREENPLAY_SYSTEM_PROMPT,
            request,
            Screenplay,
            "screenplay",
            validate=lambda screenplay: self._prepare_and_validate_screenplay(
                screenplay, story_summary, target_duration_s
            ),
        )
        screenplay = self._apply_explicit_exclusions(
            screenplay, story_summary, constraints
        )
        return self._normalize_screenplay_chronology(screenplay, story_summary)

    async def draft_narration(
        self,
        objective: str,
        constraints: list[str],
        screenplay: Screenplay,
        evidence_by_beat: dict[str, list[ScreenwriterSceneEvidence]],
    ) -> NarrationDraft:
        screenplay_lines = "\n".join(
            f"{beat.id}: purpose={beat.purpose}; story={beat.story_content}; "
            f"visual={beat.visual_intent}; mood={beat.mood}; "
            f"duration={beat.target_duration_s:.1f}s; "
            f"sequences={','.join(beat.source_sequence_ids)}"
            for beat in screenplay.beats
        )
        evidence_lines = "\n".join(
            f"{beat_id}: "
            + " | ".join(
                f"{item.scene_id}: {item.short_summary}; {item.caption}; "
                f"tags={','.join(item.tags)}"
                for item in scenes
            )
            for beat_id, scenes in evidence_by_beat.items()
        )
        length_instruction = ""
        if self.narration_language.lower() in {"zh", "zh-cn", "zh-hans"}:
            minimum = round(screenplay.target_duration_s * 2.2)
            maximum = round(screenplay.target_duration_s * 3.0)
            preferred = round(screenplay.target_duration_s * 2.5)
            length_instruction = (
                f" Across all beat narration fields, write {minimum}-{maximum} "
                f"Chinese Han characters in total and aim for about {preferred}. "
                "Allocate that character budget in proportion to each beat duration."
            )
            length_instruction += (
                " Keep shorter beats concise. The Editor will deterministically "
                "rebalance beat timing around the finished narration while "
                "preserving the total target duration and source-footage limits."
            )
        request = (
            f"Director's creative brief:\n{objective}\n\n"
            f"Task constraints:\n{self._format_constraints(constraints)}\n\n"
            f"Screenplay beats:\n{screenplay_lines}\n\n"
            f"Selected source scene evidence by beat:\n{evidence_lines}\n\n"
            "Compose one continuous story with explicit sentence-to-sentence "
            "continuity, a setup, escalation, turn, and payoff. Return it in order as "
            "one narration entry for every screenplay beat. Leave full_narration "
            "empty because the application derives it from those entries. Keep each "
            "passage natural for spoken delivery and appropriate to the beat's "
            "duration. Do not write isolated taglines. "
            "Use only the supplied evidence and do not invent facts, motives, or "
            "events. Cite the evidence_scene_ids used for each narration. Every "
            "cited scene must come from that beat's supplied evidence."
            + self._narration_language_instruction()
            + length_instruction
        )
        draft = await self._complete(
            self.NARRATION_SYSTEM_PROMPT,
            request,
            NarrationDraft,
            "narration_draft",
            validate=lambda draft: self._validate_narration(draft, screenplay),
        )
        draft = await self._repair_narration_fit(draft, screenplay)
        self._validate_narration(draft, screenplay)
        return draft

    def _narration_language_instruction(self) -> str:
        if self.narration_language == "auto":
            return ""
        instruction = f" Write every narration line in {self.narration_language}."
        if self.narration_language.lower() in {"zh", "zh-cn", "zh-hans"}:
            instruction += " Use concise, natural Simplified Chinese."
        return instruction

    def _validate_narration_language(self, draft: NarrationDraft) -> None:
        if self.narration_language.lower() not in {"zh", "zh-cn", "zh-hans"}:
            return
        missing = [
            beat.beat_id
            for beat in draft.beats
            if re.search(r"[\u4e00-\u9fff]", beat.narration) is None
        ]
        if missing:
            raise ValueError(
                "Narration must be Simplified Chinese for beats: " + ", ".join(missing)
            )

    def _validate_narration(
        self, draft: NarrationDraft, screenplay: Screenplay
    ) -> None:
        self._validate_narration_language(draft)
        beat_ids = [beat.beat_id for beat in draft.beats]
        expected = [beat.id for beat in screenplay.beats]
        if beat_ids != expected:
            raise ValueError(
                "Narration beats must match screenplay beats in the same order"
            )
        if self.narration_language == "auto":
            return
        joined = "".join(beat.narration for beat in draft.beats)
        draft.full_narration = joined
        if self.narration_language.lower() in {"zh", "zh-cn", "zh-hans"}:
            character_count = len(re.findall(r"[\u4e00-\u9fff]", joined))
            minimum = round(screenplay.target_duration_s * 1.8)
            maximum = round(screenplay.target_duration_s * 4.0)
            if not minimum <= character_count <= maximum:
                raise ValueError(
                    f"Chinese narration allows {minimum}-{maximum} Han characters "
                    f"for {screenplay.target_duration_s:.1f}s; received "
                    f"{character_count}"
                )

    async def _repair_narration_fit(
        self, draft: NarrationDraft, screenplay: Screenplay
    ) -> NarrationDraft:
        if self.narration_language.lower() not in {"zh", "zh-cn", "zh-hans"}:
            return draft
        limits = {
            beat.id: self._beat_han_limit(beat.target_duration_s)
            for beat in screenplay.beats
        }
        overlong = [
            beat
            for beat in draft.beats
            if len(re.findall(r"[\u4e00-\u9fff]", beat.narration))
            > limits[beat.beat_id]
        ]
        if not overlong:
            return draft
        context = "\n".join(f"{beat.beat_id}: {beat.narration}" for beat in draft.beats)
        requirements = "\n".join(
            f"{beat.beat_id}: maximum {limits[beat.beat_id]} Han characters; "
            f"retain evidence_scene_ids={','.join(beat.evidence_scene_ids)}"
            for beat in overlong
        )
        expected_ids = [beat.beat_id for beat in overlong]
        original_evidence = {beat.beat_id: beat.evidence_scene_ids for beat in overlong}
        revision = await self._complete(
            self.NARRATION_SYSTEM_PROMPT,
            "The complete narration is already approved for factual content and "
            "story order. Rewrite only the listed overlong beat passages so each "
            "fits its exact Han-character limit. Preserve the setup-to-payoff "
            "continuity with the surrounding passages, remove repetition first, "
            "and do not add facts. Return exactly the requested beat records in "
            "the listed order.\n\n"
            f"Complete narration context:\n{context}\n\n"
            f"Required revisions:\n{requirements}",
            _NarrationRevision,
            "narration_revision",
            validate=lambda value: self._validate_narration_revision(
                value,
                expected_ids,
                original_evidence,
                limits,
            ),
        )
        replacements = {beat.beat_id: beat for beat in revision.beats}
        fitted = draft.model_copy(
            update={
                "beats": [replacements.get(beat.beat_id, beat) for beat in draft.beats]
            }
        )
        fitted.full_narration = "".join(beat.narration for beat in fitted.beats)
        return fitted

    @staticmethod
    def _validate_narration_revision(
        revision: _NarrationRevision,
        expected_ids: list[str],
        original_evidence: dict[str, list[str]],
        limits: dict[str, int],
    ) -> None:
        actual_ids = [beat.beat_id for beat in revision.beats]
        if actual_ids != expected_ids:
            raise ValueError(
                "Narration revision must contain exactly the requested beats in order"
            )
        for beat in revision.beats:
            if beat.evidence_scene_ids != original_evidence[beat.beat_id]:
                raise ValueError(
                    f"Narration revision changed evidence for {beat.beat_id}"
                )
            count = len(re.findall(r"[\u4e00-\u9fff]", beat.narration))
            if count > limits[beat.beat_id]:
                raise ValueError(
                    f"Narration revision {beat.beat_id} allows at most "
                    f"{limits[beat.beat_id]} Han characters; received {count}"
                )

    @staticmethod
    def _beat_han_limit(target_duration_s: float) -> int:
        return max(4, int(max(0.5, target_duration_s - 0.35) * 3.8))

    async def _complete(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        schema_name: str,
        validate: Callable[[Any], None] | None = None,
    ):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return await request_structured_output(
                    self._client,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    schema=schema,
                    schema_name=schema_name,
                    max_tokens=self.max_tokens,
                    temperature=0.2,
                    mode=self.structured_output_mode,
                    validation_retries=(
                        2 if schema_name.startswith("narration_") else 1
                    ),
                    validate=validate,
                )
            except Exception as error:
                last_error = error
                status = getattr(error, "status_code", None)
                if status not in {429, 500, 502, 503, 504, 524} or attempt == 2:
                    break
                await asyncio.sleep((2**attempt) + random.random())
        if self._fallback_client is not None:
            for attempt in range(2):
                try:
                    return await request_structured_output(
                        self._fallback_client,
                        model=self._fallback_model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        schema=schema,
                        schema_name=schema_name,
                        max_tokens=self.max_tokens,
                        temperature=0.2,
                        mode=self.fallback_structured_output_mode,
                        validation_retries=(
                            2 if schema_name.startswith("narration_") else 1
                        ),
                        validate=validate,
                    )
                except Exception as error:
                    last_error = error
                    if attempt == 1:
                        break
                    await asyncio.sleep(1.0 + random.random())
        raise last_error or ValueError("Screenwriter model request returned no result")

    @staticmethod
    def _prepare_and_validate_screenplay(
        screenplay: Screenplay,
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> None:
        """Canonicalize small gaps and split distant selections into edit beats."""
        sequence_order = {
            sequence.id: position
            for position, sequence in enumerate(story_summary.sequences)
        }
        sequence_ids = [sequence.id for sequence in story_summary.sequences]
        prepared: list[ScreenplayBeat] = []
        split_budget = (
            max(0, 7 - len(screenplay.beats))
            if target_duration_s >= 30
            else len(story_summary.sequences)
        )
        for beat_index, beat in enumerate(screenplay.beats):
            if any(item not in sequence_order for item in beat.source_sequence_ids):
                prepared.append(beat)
                continue
            positions = sorted(
                sequence_order[item] for item in beat.source_sequence_ids
            )
            span = sequence_ids[positions[0] : positions[-1] + 1]
            if len(span) <= 3:
                prepared.append(beat.model_copy(update={"source_sequence_ids": span}))
                continue
            runs: list[list[int]] = [[positions[0]]]
            for position in positions[1:]:
                if position == runs[-1][-1] + 1:
                    runs[-1].append(position)
                else:
                    runs.append([position])
            extra_beats = len(runs) - 1
            if extra_beats > split_budget:
                if beat_index == len(screenplay.beats) - 1:
                    selected_run = runs[-1]
                else:
                    selected_run = max(
                        enumerate(runs),
                        key=lambda item: (len(item[1]), -item[0]),
                    )[1]
                prepared.append(
                    beat.model_copy(
                        update={
                            "source_sequence_ids": [
                                sequence_ids[position] for position in selected_run
                            ]
                        }
                    )
                )
                continue
            split_budget -= extra_beats
            allocated_s = 0.0
            for part, run in enumerate(runs, start=1):
                if part == len(runs):
                    duration_s = beat.target_duration_s - allocated_s
                else:
                    duration_s = beat.target_duration_s * len(run) / len(positions)
                    allocated_s += duration_s
                prepared.append(
                    beat.model_copy(
                        update={
                            "id": f"{beat.id}-part-{part}",
                            "target_duration_s": duration_s,
                            "source_sequence_ids": [
                                sequence_ids[position] for position in run
                            ],
                        }
                    )
                )
        if prepared != screenplay.beats:
            width = max(2, len(str(len(prepared))))
            screenplay.beats = [
                beat.model_copy(update={"id": f"beat-{number:0{width}d}"})
                for number, beat in enumerate(prepared, 1)
            ]
        total_duration_s = sum(beat.target_duration_s for beat in screenplay.beats)
        scale = target_duration_s / total_duration_s
        desired = [beat.target_duration_s * scale for beat in screenplay.beats]
        sequence_durations = {
            sequence.id: sequence.source_range.duration_s
            for sequence in story_summary.sequences
            if sequence.source_range is not None
        }
        capacities = [
            (
                sum(sequence_durations[item] for item in beat.source_sequence_ids)
                if all(item in sequence_durations for item in beat.source_sequence_ids)
                else float("inf")
            )
            for beat in screenplay.beats
        ]
        if capacities and target_duration_s >= 30:
            capacities[-1] = min(capacities[-1], 8.0)
        values = [
            min(duration_s, capacity)
            for duration_s, capacity in zip(desired, capacities, strict=True)
        ]
        remaining_s = target_duration_s - sum(values)
        while remaining_s > 1e-6:
            active = [
                index
                for index, (value, capacity) in enumerate(
                    zip(values, capacities, strict=True)
                )
                if capacity - value > 1e-6
            ]
            if not active:
                raise ValueError(
                    "Selected source sequences cannot fill the target duration"
                )
            weight = sum(desired[index] for index in active)
            applied_s = 0.0
            for index in active:
                share_s = remaining_s * desired[index] / weight
                change_s = min(capacities[index] - values[index], share_s)
                values[index] += change_s
                applied_s += change_s
            if applied_s <= 1e-9:
                raise ValueError("Could not distribute screenplay duration")
            remaining_s -= applied_s
        values[-1] += target_duration_s - sum(values)
        screenplay.beats = [
            beat.model_copy(update={"target_duration_s": duration_s})
            for beat, duration_s in zip(screenplay.beats, values, strict=True)
        ]
        screenplay.target_duration_s = target_duration_s
        OpenAICompatibleScreenwriterModel._validate_screenplay_plan(
            screenplay,
            story_summary,
            target_duration_s,
        )

    @staticmethod
    def _validate_screenplay_plan(
        screenplay: Screenplay,
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> None:
        beat_ids = [beat.id for beat in screenplay.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("Screenplay beat IDs must be unique")
        if target_duration_s >= 30 and not 4 <= len(screenplay.beats) <= 7:
            raise ValueError(
                "Screenplay edits of 30 seconds or longer require 4 to 7 beats"
            )
        if abs(screenplay.target_duration_s - target_duration_s) > 0.1:
            raise ValueError("Screenplay target duration must match the request")
        beat_duration_s = sum(beat.target_duration_s for beat in screenplay.beats)
        if abs(beat_duration_s - screenplay.target_duration_s) > 0.1:
            raise ValueError(
                "Screenplay beat durations must sum to its target duration"
            )
        sequence_order = {
            sequence.id: position
            for position, sequence in enumerate(story_summary.sequences)
        }
        sequence_durations = {
            sequence.id: sequence.source_range.duration_s
            for sequence in story_summary.sequences
            if sequence.source_range is not None
        }
        selected = [
            sequence_id
            for beat in screenplay.beats
            for sequence_id in beat.source_sequence_ids
        ]
        unknown = [
            sequence_id for sequence_id in selected if sequence_id not in sequence_order
        ]
        if unknown:
            raise ValueError(
                "Screenplay cited unknown source sequence IDs: "
                + ", ".join(sorted(set(unknown)))
            )
        if len(selected) != len(set(selected)):
            raise ValueError("Screenplay must not reuse a source sequence across beats")
        for beat in screenplay.beats:
            positions = [sequence_order[item] for item in beat.source_sequence_ids]
            if positions != list(range(min(positions), max(positions) + 1)):
                raise ValueError(
                    f"Screenplay beat {beat.id} must cite consecutive sequences"
                )
            if all(item in sequence_durations for item in beat.source_sequence_ids):
                available_s = sum(
                    sequence_durations[item] for item in beat.source_sequence_ids
                )
                if beat.target_duration_s > available_s + 0.1:
                    raise ValueError(
                        f"Screenplay beat {beat.id} requests "
                        f"{beat.target_duration_s:.1f}s but its selected source "
                        f"sequences provide only {available_s:.1f}s"
                    )

    @staticmethod
    def _apply_explicit_exclusions(
        screenplay: Screenplay,
        story_summary: StorySummary,
        constraints: list[str],
    ) -> Screenplay:
        normalized_constraints = [constraint.casefold() for constraint in constraints]
        exclude_credits = any(
            "exclude" in constraint and "credit" in constraint
            for constraint in normalized_constraints
        )
        if not exclude_credits:
            return screenplay
        summaries = {
            sequence.id: sequence.short_summary.casefold()
            for sequence in story_summary.sequences
        }
        credit_ids = {
            sequence_id
            for sequence_id, summary in summaries.items()
            if "credit" in summary and ("cast" in summary or "crew" in summary)
        }
        kept: list[ScreenplayBeat] = []
        removed_duration_s = 0.0
        for beat in screenplay.beats:
            allowed = [
                sequence_id
                for sequence_id in beat.source_sequence_ids
                if sequence_id not in credit_ids
            ]
            if not allowed:
                removed_duration_s += beat.target_duration_s
            else:
                kept.append(beat.model_copy(update={"source_sequence_ids": allowed}))
        if not removed_duration_s:
            return screenplay
        if not kept:
            raise ValueError("Explicit exclusions removed every screenplay beat")
        kept[-1] = kept[-1].model_copy(
            update={
                "target_duration_s": (kept[-1].target_duration_s + removed_duration_s)
            }
        )
        width = max(2, len(str(len(kept))))
        kept = [
            beat.model_copy(update={"id": f"beat-{number:0{width}d}"})
            for number, beat in enumerate(kept, 1)
        ]
        return screenplay.model_copy(update={"beats": kept})

    @staticmethod
    def _normalize_screenplay_chronology(
        screenplay: Screenplay,
        story_summary: StorySummary,
    ) -> Screenplay:
        sequence_order = {
            sequence.id: position
            for position, sequence in enumerate(story_summary.sequences)
        }
        ordered = sorted(
            [
                beat.model_copy(
                    update={
                        "source_sequence_ids": sorted(
                            beat.source_sequence_ids,
                            key=sequence_order.__getitem__,
                        )
                    }
                )
                for beat in screenplay.beats
            ],
            key=lambda beat: sequence_order[beat.source_sequence_ids[0]],
        )
        if ordered == screenplay.beats:
            return screenplay
        width = max(2, len(str(len(ordered))))
        normalized = [
            beat.model_copy(update={"id": f"beat-{number:0{width}d}"})
            for number, beat in enumerate(ordered, 1)
        ]
        return screenplay.model_copy(update={"beats": normalized})

    @staticmethod
    def _format_constraints(constraints: list[str]) -> str:
        return "\n".join(f"- {constraint}" for constraint in constraints) or "- None"


class OpenAICompatibleStoryStructureModel:
    """Build a citation-backed film hierarchy from indexed scene evidence."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "LLM_API_KEY",
        max_tokens: int = 4000,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        max_scenes_per_chunk: int = 24,
        structured_output_mode: StructuredOutputMode = "prompted_json",
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
        self.structured_output_mode = structured_output_mode
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
                "captions and tags. Return typed sequence records. "
                "Every input scene must appear exactly once in scene_ids. Do not "
                "invent scene IDs, events, identities, or causes. Keep the source "
                "order and write a concise factual short_summary.",
                f"Scene evidence chunk {chunk_number}:\n{evidence}",
                _SequenceBatch,
                validate=lambda result, source=chunk: self._validate_sequence_batch(
                    result, source
                ),
            )
            sequences.extend(
                self._normalize_chunk_sequences(batch.sequences, chunk, len(sequences))
            )

        sequence_context = "\n".join(
            f"{sequence.id}: {sequence.short_summary} "
            f"(scenes={','.join(sequence.scene_ids)})"
            for sequence in sequences
        )
        act_draft = await self._complete(
            "You are a film-footage analyst. Group the supplied chronological "
            "sequences into coherent narrative acts. Use every sequence exactly "
            "once and do not invent events or IDs. Return typed act records.",
            f"Sequence summaries:\n{sequence_context}",
            _StructurePartDraft,
            validate=lambda result: self._validate_act_draft(result, sequences),
        )
        all_acts = act_draft.acts
        acts = [
            StoryAct(
                id=f"act-{number:04d}",
                title=act.title or f"Act {number}",
                short_summary=act.short_summary or "Chronological footage segment.",
                sequence_ids=act.sequence_ids,
            )
            for number, act in enumerate(all_acts, start=1)
        ]
        acts = self._normalize_acts(acts, sequences)
        act_context = "\n".join(
            f"{act.id}: {act.title} - {act.short_summary} "
            f"(sequences={','.join(act.sequence_ids)}; "
            f"scenes={','.join(act.source_scene_ids)})"
            for act in acts
        )
        final_draft = await self._complete(
            "You are a senior film-footage analyst. Create the final film-level "
            "summary from the supplied acts and sequence IDs. Do not invent "
            "facts. Character arcs and major events must cite only supplied "
            "source scene IDs.",
            f"Acts:\n{act_context}",
            _StoryStructureDraft,
            validate=lambda result: self._validate_final_draft(result, video_index),
        )
        character_arcs = [
            CharacterArc(
                character=item.character,
                short_summary=item.short_summary,
                source_scene_ids=item.source_scene_ids,
            )
            for item in final_draft.character_arcs
        ]
        major_events = [
            MajorEvent(
                id=item.id,
                short_summary=item.short_summary,
                source_scene_ids=item.source_scene_ids,
                confidence=item.confidence,
            )
            for item in final_draft.major_events
        ]
        return StorySummary(
            title=(
                final_draft.title.strip() or act_draft.title.strip() or "Untitled Film"
            ),
            short_summary=(
                final_draft.short_summary.strip()
                or act_draft.short_summary.strip()
                or "Chronological summary of the indexed footage."
            ),
            sequences=sequences,
            acts=acts,
            character_arcs=character_arcs,
            major_events=major_events,
        )

    @staticmethod
    def _normalize_acts(
        acts: list[StoryAct], sequences: list[StorySequence]
    ) -> list[StoryAct]:
        expected = [sequence.id for sequence in sequences]
        positions = {sequence_id: index for index, sequence_id in enumerate(expected)}
        sequence_scenes = {sequence.id: sequence.scene_ids for sequence in sequences}
        seen: set[str] = set()
        runs: list[tuple[int, str, str, list[str]]] = []
        for act in acts:
            ids = sorted(
                {
                    item
                    for item in act.sequence_ids
                    if item in positions and item not in seen
                },
                key=positions.__getitem__,
            )
            if not ids:
                continue
            act_runs: list[list[str]] = [[]]
            for sequence_id in ids:
                if (
                    act_runs[-1]
                    and positions[sequence_id] != positions[act_runs[-1][-1]] + 1
                ):
                    act_runs.append([])
                act_runs[-1].append(sequence_id)
            for run in act_runs:
                seen.update(run)
                runs.append(
                    (
                        positions[run[0]],
                        act.title,
                        act.short_summary,
                        run,
                    )
                )
        missing = [item for item in expected if item not in seen]
        for sequence_id in missing:
            runs.append(
                (
                    positions[sequence_id],
                    "Additional footage",
                    "Chronological footage segment.",
                    [sequence_id],
                )
            )
        runs.sort(key=lambda item: item[0])
        return [
            StoryAct(
                id=f"act-{number:04d}",
                title=title,
                short_summary=short_summary,
                sequence_ids=sequence_ids,
                source_scene_ids=[
                    scene_id
                    for sequence_id in sequence_ids
                    for scene_id in sequence_scenes[sequence_id]
                ],
            )
            for number, (_, title, short_summary, sequence_ids) in enumerate(runs, 1)
        ]

    @staticmethod
    def _validate_sequence_batch(batch: _SequenceBatch, scenes: list[Scene]) -> None:
        if any(not item.short_summary.strip() for item in batch.sequences):
            raise ValueError("Every sequence requires a factual short_summary")
        if any(not item.scene_ids for item in batch.sequences):
            raise ValueError("Every sequence requires at least one scene ID")
        expected = [scene.id for scene in scenes]
        actual = [scene_id for item in batch.sequences for scene_id in item.scene_ids]
        if Counter(actual) != Counter(expected):
            raise ValueError(
                "Sequence scene_ids must contain every input scene exactly once; "
                f"expected {expected}, received {actual}"
            )

    @staticmethod
    def _validate_act_draft(
        draft: _StructurePartDraft, sequences: list[StorySequence]
    ) -> None:
        if not draft.acts:
            raise ValueError("Story structure requires at least one act")
        if any(not act.short_summary.strip() for act in draft.acts):
            raise ValueError("Every act requires a factual short_summary")
        expected = [sequence.id for sequence in sequences]
        actual = [sequence_id for act in draft.acts for sequence_id in act.sequence_ids]
        if Counter(actual) != Counter(expected):
            raise ValueError(
                "Act sequence_ids must contain every sequence exactly once; "
                f"expected {expected}, received {actual}"
            )

    @staticmethod
    def _validate_final_draft(
        draft: _StoryStructureDraft, video_index: VideoIndex
    ) -> None:
        if not draft.title.strip() or not draft.short_summary.strip():
            raise ValueError("Final story structure requires a title and summary")
        known_scene_ids = {scene.id for scene in video_index.scenes}
        cited_scene_ids = {
            scene_id
            for item in [*draft.character_arcs, *draft.major_events]
            for scene_id in item.source_scene_ids
        }
        unknown = sorted(cited_scene_ids - known_scene_ids)
        if unknown:
            raise ValueError(
                "Final story structure cited unknown scene IDs: " + ", ".join(unknown)
            )

    async def _complete(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        validate: Callable[[Any], None] | None = None,
    ):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return await self._request_with_backoff(messages, schema, validate)
        except Exception as error:
            raise ValueError(
                "Story structure provider request failed after retries: "
                f"{type(error).__name__}: {error}"
            ) from error

    async def _request_with_backoff(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        validate: Callable[[Any], None] | None,
    ):
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return await request_structured_output(
                    self._client,
                    model=self.model,
                    messages=messages,
                    schema=schema,
                    schema_name=schema.__name__.lstrip("_").lower(),
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    mode=self.structured_output_mode,
                    validation_retries=1,
                    validate=validate,
                )
            except Exception as error:
                last_error = error
                status = getattr(error, "status_code", None)
                if status not in {429, 500, 502, 503, 504, 524} or attempt == 3:
                    raise
                await asyncio.sleep((2**attempt) + random.random())
        raise last_error or RuntimeError("Story structure request failed")

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
        positions = {scene_id: index for index, scene_id in enumerate(expected)}
        seen: set[str] = set()
        runs: list[tuple[int, str, str, list[str]]] = []
        for sequence in sequences:
            if not sequence.short_summary.strip():
                raise ValueError("Sequence short_summary is required")
            scene_ids = sorted(
                {
                    scene_id
                    for scene_id in sequence.scene_ids
                    if scene_id in positions and scene_id not in seen
                },
                key=positions.__getitem__,
            )
            if not scene_ids:
                continue
            sequence_runs: list[list[str]] = [[]]
            for scene_id in scene_ids:
                if (
                    sequence_runs[-1]
                    and positions[scene_id] != positions[sequence_runs[-1][-1]] + 1
                ):
                    sequence_runs.append([])
                sequence_runs[-1].append(scene_id)
            for run in sequence_runs:
                seen.update(run)
                runs.append(
                    (
                        positions[run[0]],
                        sequence.title.strip() or sequence.short_summary[:80],
                        sequence.short_summary,
                        run,
                    )
                )
        for scene_id in expected:
            if scene_id not in seen:
                runs.append(
                    (
                        positions[scene_id],
                        f"Scene {scene_id}",
                        "Unassigned indexed scene.",
                        [scene_id],
                    )
                )
        runs.sort(key=lambda item: item[0])
        return [
            StorySequence(
                id=f"sequence-{sequence_offset + number:04d}",
                title=title,
                short_summary=short_summary,
                scene_ids=scene_ids,
            )
            for number, (_, title, short_summary, scene_ids) in enumerate(runs, 1)
        ]


class OpenAICompatibleSceneTagger:
    """Create an information-rich, searchable record for each scene."""

    SYSTEM_PROMPT = """Create typed searchable metadata for one video scene.
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

Use empty lists or null for unsupported optional fields. Avoid duplicates,
vague adjectives, and speculative labels."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://vyceai.com/v1",
        api_key_env: str = "LLM_API_KEY",
        max_tokens: int = 1200,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        max_parallel: int = 2,
        request_interval_s: float = 1.0,
        structured_output_mode: StructuredOutputMode = "prompted_json",
        client: Any | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if request_interval_s < 0:
            raise ValueError("request_interval_s must be non-negative")
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
        self.max_retries = max_retries
        self.max_parallel = max_parallel
        self.request_interval_s = request_interval_s
        self.structured_output_mode = structured_output_mode
        self._request_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def tag_batch(self, scenes: list[Scene]) -> dict[str, SceneTags]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def tag(scene: Scene) -> tuple[str, SceneTags]:
            async with semaphore:
                return await self._tag_one(scene)

        results = await asyncio.gather(*(tag(scene) for scene in scenes))
        return {scene_id: tags for scene_id, tags in results}

    async def _tag_one(self, scene: Scene) -> tuple[str, SceneTags]:
        request = (
            f"Scene ID: {scene.id}\n"
            f"Transcript: {scene.transcript or '(none)'}\n"
            f"Dense visual caption: {scene.dense_caption or '(none)'}"
        )
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            wait_s = self.request_interval_s - (loop.time() - self._last_request_at)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_request_at = loop.time()
        for attempt in range(self.max_retries + 1):
            try:
                tags = await request_structured_output(
                    self._client,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": request},
                    ],
                    schema=SceneTags,
                    schema_name="scene_tags",
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    mode=self.structured_output_mode,
                    validation_retries=1,
                    validate=self._validate_tags,
                )
                return scene.id, tags
            except Exception as error:
                status = getattr(error, "status_code", None)
                if (
                    status not in {429, 500, 502, 503, 504, 524}
                    or attempt >= self.max_retries
                ):
                    raise
                await asyncio.sleep((2**attempt) + random.random())
        raise ValueError(f"Scene tagger returned no completion for {scene.id}")

    @staticmethod
    def _validate_tags(tags: SceneTags) -> None:
        if not tags.caption.strip() or not tags.short_summary.strip():
            raise ValueError("Scene tags require a caption and short_summary")


class EdgeSpeechTTS:
    """Cross-platform Chinese Edge TTS with fail-fast error handling."""

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: int = 185,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_delay_s < 0:
            raise ValueError("retry_delay_s must be non-negative")
        self.voice = voice
        self.rate = rate
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

    async def synthesize(self, text: str, output_path: Path) -> float:
        return await asyncio.to_thread(self._synthesize_sync, text, output_path)

    def _synthesize_sync(self, text: str, output_path: Path) -> float:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.max_retries + 1):
            try:
                self._edge_tts(text, output_path)
                break
            except Exception:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay_s * (2**attempt))
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
