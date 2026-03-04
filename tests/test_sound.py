import asyncio
import json
from pathlib import Path

import pytest

from directorx.agents import DirectorAgent, FootageAnalystAgent, SoundAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import (
    MusicIndex,
    MusicIndexEntry,
    MusicTrack,
    NarrationManifest,
    NarrationSegment,
    SoundPlan,
    StoryBeat,
    Storyboard,
    TimeRange,
    VideoIndex,
)
from tests.fakes import FixedAudioTextEmbeddingProvider, FixedMusicLibrary


class UnusedIndexer:
    async def build(self, video_path: Path) -> VideoIndex:
        raise AssertionError("The footage agent is not used in this test")


def _storyboard() -> Storyboard:
    return Storyboard(
        title="The Wait",
        logline="A traveler waits through a station blackout.",
        narrative_angle="Build tension around uncertainty.",
        beats=[
            StoryBeat(
                id="beat-1",
                purpose="Establish uncertainty",
                story_content="The traveler arrives at an empty station.",
                narration="The station should have been busy.",
                visual_intent="An empty station platform",
                mood="uneasy",
                target_duration_s=5,
                source_sequence_ids=["sequence-0001"],
                evidence_scene_ids=["scene-0001"],
            ),
            StoryBeat(
                id="beat-2",
                purpose="Escalate tension",
                story_content="The station lights fail.",
                narration="Then every light went out.",
                visual_intent="A station falling into darkness",
                mood="tense",
                target_duration_s=7,
                source_sequence_ids=["sequence-0002"],
                evidence_scene_ids=["scene-0002"],
            ),
        ],
        target_duration_s=12,
    )


def _narration(tmp_path: Path) -> NarrationManifest:
    return NarrationManifest(
        segments=[
            NarrationSegment(
                beat_id="beat-1",
                text="The station should have been busy.",
                audio_path=tmp_path / "beat-1.wav",
                target_duration_s=5,
                duration_s=18,
            ),
            NarrationSegment(
                beat_id="beat-2",
                text="Then every light went out.",
                audio_path=tmp_path / "beat-2.wav",
                target_duration_s=7,
                duration_s=6,
            ),
        ],
        target_duration_s=12,
        duration_s=24,
    )


def _source_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    storyboard_path = tmp_path / "screenwriter" / "storyboard.json"
    narration_path = tmp_path / "narration" / "narration.json"
    storyboard_path.parent.mkdir()
    narration_path.parent.mkdir()
    storyboard_path.write_text(
        _storyboard().model_dump_json(indent=2), encoding="utf-8"
    )
    narration_path.write_text(
        _narration(tmp_path).model_dump_json(indent=2), encoding="utf-8"
    )
    return storyboard_path, narration_path


def _task(paths: tuple[Path, Path], task_id: str = "sound-001") -> TaskContext:
    storyboard_path, narration_path = paths
    return TaskContext(
        task_id=task_id,
        assignee=AgentRole.SOUND,
        objective="Select one background track for the complete edit.",
        input_artifacts=[
            ArtifactRef(name="storyboard", path=storyboard_path),
            ArtifactRef(name="narration-manifest", path=narration_path),
        ],
        expected_output="One persisted sound plan.",
        acceptance_criteria=["Use exactly one background music track"],
    )


def _tracks(tmp_path: Path) -> list[MusicTrack]:
    return [
        MusicTrack(
            path=tmp_path / "music" / "long.mp3",
            title="Long",
            tags=["tense"],
            duration_s=40,
        ),
        MusicTrack(
            path=tmp_path / "music" / "short.mp3",
            title="Short",
            tags=["uneasy"],
            duration_s=4,
        ),
    ]


def _provider() -> FixedAudioTextEmbeddingProvider:
    return FixedAudioTextEmbeddingProvider(
        {
            "long.mp3": [[0.0, 1.0]],
            "short.mp3": [[1.0, 0.0]],
        }
    )


def test_director_delegates_and_sound_selects_one_track_from_artifacts(
    tmp_path: Path,
) -> None:
    paths = _source_artifacts(tmp_path)
    provider = _provider()
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(
        runtime,
        FootageAnalystAgent(UnusedIndexer()),
        artifacts_dir=tmp_path / "artifacts",
        sound_agent=SoundAgent(FixedMusicLibrary(_tracks(tmp_path)), provider),
    )

    result = asyncio.run(director.run_sound_task(_task(paths)))

    plan_path = tmp_path / "artifacts" / "sound-plan.json"
    plan = SoundPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert result.agent == AgentRole.SOUND
    assert result.output_artifacts == [ArtifactRef(name="sound-plan", path=plan_path)]
    assert plan.track.title == "Short"
    assert plan.target_duration_s == 24
    assert plan.track.duration_s == 4
    assert plan.match_score == pytest.approx(1)
    assert "Dominant moods: uneasy, tense" in provider.text_calls[0]
    assert len(provider.audio_calls) == 2
    windows_by_track = {path.name: windows for path, windows in provider.audio_calls}
    assert len(windows_by_track["long.mp3"]) == 3
    assert len(windows_by_track["short.mp3"]) == 1
    assert director.read_result(task_id="sound-001") == result
    result_payload = json.loads(
        (tmp_path / "coordination" / "tasks" / "sound-001.result.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(result_payload) == {
        "task_id",
        "agent",
        "status",
        "summary",
        "output_artifacts",
    }


def test_sound_task_reads_music_index_without_decoding_audio(tmp_path: Path) -> None:
    storyboard_path, narration_path = _source_artifacts(tmp_path)
    index_path = tmp_path / "index" / "music-index.json"
    index_path.parent.mkdir()
    tracks = _tracks(tmp_path)
    index = MusicIndex(
        model_name="fake-clap",
        embedding_dimension=2,
        entries=[
            MusicIndexEntry(
                track=track,
                embedding=([0.0, 1.0] if track.title == "Long" else [1.0, 0.0]),
                analysis_windows=[
                    # The windows are provenance only during Sound execution.
                    TimeRange(start_s=0, end_s=min(track.duration_s, 10))
                ],
                model_name="fake-clap",
            )
            for track in tracks
        ],
    )
    index_path.write_text(index.model_dump_json(), encoding="utf-8")
    task = TaskContext(
        task_id="sound-indexed",
        assignee=AgentRole.SOUND,
        objective="Select one background track.",
        input_artifacts=[
            ArtifactRef(name="storyboard", path=storyboard_path),
            ArtifactRef(name="narration-manifest", path=narration_path),
            ArtifactRef(name="music-index", path=index_path),
        ],
        expected_output="One persisted sound plan.",
        acceptance_criteria=["Use exactly one background music track"],
    )
    provider = FixedAudioTextEmbeddingProvider(
        {}, text_embedding=[1.0, 0.0], model_name="fake-clap"
    )
    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = asyncio.run(
        SoundAgent(provider, require_music_index=True).run_task(
            task, runtime, tmp_path / "artifacts"
        )
    )
    assert result.status == "completed"
    assert provider.audio_calls == []
    assert result.output_artifacts == [
        ArtifactRef(name="sound-plan", path=tmp_path / "artifacts" / "sound-plan.json")
    ]


def test_sound_requires_music_index_for_production_tasks(tmp_path: Path) -> None:
    paths = _source_artifacts(tmp_path)
    task = _task(paths, "sound-missing-index")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = asyncio.run(
        SoundAgent(
            FixedAudioTextEmbeddingProvider({}, model_name="fake-clap"),
            require_music_index=True,
        ).run_task(task, runtime, tmp_path / "artifacts")
    )
    assert result.status == "blocked"
    assert "music-index" in result.summary


def test_sound_uses_tags_then_path_only_to_break_embedding_ties(
    tmp_path: Path,
) -> None:
    tracks = [
        MusicTrack(
            path=tmp_path / "z.mp3",
            title="Z",
            tags=["uneasy"],
            duration_s=20,
        ),
        MusicTrack(
            path=tmp_path / "a.mp3",
            title="A",
            tags=["tense"],
            duration_s=20,
        ),
    ]
    provider = FixedAudioTextEmbeddingProvider(
        {"a.mp3": [[1.0, 0.0]], "z.mp3": [[1.0, 0.0]]}
    )
    plan = asyncio.run(
        SoundAgent(FixedMusicLibrary(tracks), provider).run(
            _storyboard(), _narration(tmp_path)
        )
    )
    assert plan.track.title == "Z"

    tracks = [track.model_copy(update={"tags": []}) for track in tracks]
    plan = asyncio.run(
        SoundAgent(FixedMusicLibrary(tracks), provider).run(
            _storyboard(), _narration(tmp_path)
        )
    )
    assert plan.track.title == "A"


@pytest.mark.parametrize("failure", ["empty-library", "embedding", "artifact"])
def test_sound_persists_blocked_result_on_execution_failure(
    tmp_path: Path, failure: str
) -> None:
    paths = _source_artifacts(tmp_path)
    tracks = _tracks(tmp_path)
    provider = _provider()
    if failure == "empty-library":
        tracks = []
    elif failure == "embedding":
        provider.failing_path = tracks[0].path
    else:
        paths = (tmp_path / "missing" / "storyboard.json", paths[1])
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(paths, f"sound-{failure}")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        SoundAgent(FixedMusicLibrary(tracks), provider).run_task(
            task, runtime, tmp_path / "artifacts"
        )
    )

    assert result.status == "blocked"
    assert result.agent == AgentRole.SOUND
    assert result.output_artifacts == []
    assert "Sound selection blocked:" in result.summary
    assert not (tmp_path / "artifacts" / "sound-plan.json").exists()
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_sound_does_not_overwrite_an_existing_plan(tmp_path: Path) -> None:
    paths = _source_artifacts(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    plan_path = artifacts_dir / "sound-plan.json"
    plan_path.write_text("keep", encoding="utf-8")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(paths, "sound-persistence-failure")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        SoundAgent(FixedMusicLibrary(_tracks(tmp_path)), _provider()).run_task(
            task, runtime, artifacts_dir
        )
    )

    assert result.status == "blocked"
    assert plan_path.read_text(encoding="utf-8") == "keep"
    assert list(artifacts_dir.glob(".sound-plan.*")) == []


def test_sound_rejects_another_agents_task(tmp_path: Path) -> None:
    task = TaskContext(
        task_id="narration-boundary",
        assignee=AgentRole.NARRATION,
        objective="Synthesize narration.",
        expected_output="Narration audio.",
        acceptance_criteria=["Return narration"],
    )

    with pytest.raises(ValueError):
        asyncio.run(
            SoundAgent(FixedMusicLibrary([]), _provider()).run_task(
                task,
                CoordinationRuntime(tmp_path / "coordination"),
                tmp_path / "artifacts",
            )
        )


def test_director_cannot_submit_a_sound_result(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(_source_artifacts(tmp_path), "sound-submit-boundary")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = TaskResult(
        task_id=task.task_id,
        agent=AgentRole.SOUND,
        status="completed",
        summary="Selected one background track.",
    )

    with pytest.raises(PermissionError):
        runtime.submit_result(AgentRole.DIRECTOR, result)
