import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from directorx.agents import DirectorAgent, FootageAnalystAgent, ScreenwriterAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import (
    BeatNarration,
    NarrationDraft,
    Scene,
    Screenplay,
    ScreenplayBeat,
    ScreenwriterSceneEvidence,
    StoryAct,
    Storyboard,
    StorySequence,
    StorySummary,
    TimeRange,
    VideoIndex,
)
from directorx.services.providers import OpenAICompatibleScreenwriterModel


class UnusedIndexer:
    async def build(self, video_path: Path) -> VideoIndex:
        raise AssertionError("The footage agent is not used in this test")


class FixedScreenwriterModel:
    def __init__(
        self,
        screenplay: Screenplay | None = None,
        narration: NarrationDraft | None = None,
    ) -> None:
        self.screenplay = screenplay or _screenplay()
        self.narration = narration or _narration()
        self.calls: list[str] = []
        self.received_summary: StorySummary | None = None
        self.received_evidence = None

    async def draft_screenplay(
        self,
        objective: str,
        constraints: list[str],
        story_summary: StorySummary,
        target_duration_s: float,
    ) -> Screenplay:
        self.calls.append("screenplay")
        self.received_summary = story_summary
        assert objective == "Write a concise suspenseful edit."
        assert constraints == [
            "Avoid spoilers",
            "Ground the story in the indexed scenes",
        ]
        assert target_duration_s == 12
        return self.screenplay

    async def draft_narration(
        self, objective, constraints, screenplay, evidence_by_beat
    ) -> NarrationDraft:
        self.calls.append("narration")
        self.received_evidence = evidence_by_beat
        assert screenplay == self.screenplay
        return self.narration


class FailingScreenwriterModel(FixedScreenwriterModel):
    def __init__(self, stage: str) -> None:
        super().__init__()
        self.stage = stage

    async def draft_screenplay(self, *args, **kwargs) -> Screenplay:
        if self.stage == "screenplay":
            raise RuntimeError("screenplay model unavailable")
        return await super().draft_screenplay(*args, **kwargs)

    async def draft_narration(self, *args, **kwargs) -> NarrationDraft:
        if self.stage == "narration":
            raise RuntimeError("narration model unavailable")
        return await super().draft_narration(*args, **kwargs)


def _source_artifacts(tmp_path: Path) -> tuple[Path, Path, StorySummary]:
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    index_path = footage_dir / "index.json"
    summary_path = footage_dir / "story-summary.json"
    index = VideoIndex(
        video_path=tmp_path / "movie.mp4",
        duration_s=30,
        scenes=[
            Scene(
                id="scene-0001",
                source_range=TimeRange(start_s=0, end_s=10),
                caption="A traveler enters a quiet station.",
                short_summary="A traveler arrives at an empty station.",
                tags=["traveler", "station"],
                transcript="This must never reach the screenwriter model.",
                dense_caption="This must also remain behind the evidence boundary.",
            ),
            Scene(
                id="scene-0002",
                source_range=TimeRange(start_s=10, end_s=20),
                caption="The traveler finds an abandoned suitcase.",
                short_summary="An abandoned suitcase creates suspicion.",
                tags=["suitcase", "suspicion"],
            ),
            Scene(
                id="scene-0003",
                source_range=TimeRange(start_s=20, end_s=30),
                caption="The station lights go dark.",
                short_summary="Darkness leaves the traveler waiting.",
                tags=["darkness", "waiting"],
            ),
        ],
    )
    summary = StorySummary(
        title="The Last Train",
        short_summary="A traveler confronts uncertainty in an empty station.",
        sequences=[
            StorySequence(
                id="sequence-0001",
                title="Arrival",
                short_summary="The traveler arrives and discovers a suitcase.",
                scene_ids=["scene-0001", "scene-0002"],
            ),
            StorySequence(
                id="sequence-0002",
                title="Blackout",
                short_summary="The lights fail while the traveler waits.",
                scene_ids=["scene-0003"],
            ),
        ],
        acts=[
            StoryAct(
                id="act-0001",
                title="Waiting",
                short_summary="Arrival turns into an uneasy wait.",
                sequence_ids=["sequence-0001", "sequence-0002"],
            )
        ],
    )
    index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return index_path, summary_path, summary


def _task(
    index_path: Path,
    summary_path: Path,
    task_id: str = "screenwriter-001",
) -> TaskContext:
    return TaskContext(
        task_id=task_id,
        assignee=AgentRole.SCREENWRITER,
        objective="Write a concise suspenseful edit.",
        constraints=["Avoid spoilers"],
        input_artifacts=[
            ArtifactRef(name="video-index", path=index_path),
            ArtifactRef(name="story-summary", path=summary_path),
        ],
        expected_output="A validated storyboard artifact.",
        acceptance_criteria=["Ground the story in the indexed scenes"],
    )


def _screenplay() -> Screenplay:
    return Screenplay(
        title="The Wait",
        logline="A routine wait becomes a confrontation with darkness.",
        narrative_angle="Build tension around what the traveler cannot see.",
        beats=[
            ScreenplayBeat(
                id="beat-1",
                purpose="Turn waiting into suspense",
                story_content="The station blackout isolates the traveler.",
                visual_intent="A dark station and a solitary waiting figure",
                mood="tense",
                target_duration_s=12,
                source_sequence_ids=["sequence-0002"],
            )
        ],
        target_duration_s=12,
    )


def _narration() -> NarrationDraft:
    return NarrationDraft(
        beats=[
            BeatNarration(
                beat_id="beat-1",
                narration="When the lights disappear, waiting becomes a test.",
                evidence_scene_ids=["scene-0003"],
            )
        ]
    )


def test_director_delegates_and_screenwriter_uses_both_artifacts(
    tmp_path: Path,
) -> None:
    index_path, summary_path, summary = _source_artifacts(tmp_path)
    model = FixedScreenwriterModel()
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(
        runtime,
        FootageAnalystAgent(UnusedIndexer()),
        ScreenwriterAgent(model),
        artifacts_dir=tmp_path / "artifacts",
    )

    result = asyncio.run(
        director.run_screenwriter_task(
            _task(index_path, summary_path), target_duration_s=12
        )
    )

    assert result.status == "completed"
    assert result.agent == AgentRole.SCREENWRITER
    assert model.calls == ["screenplay", "narration"]
    assert model.received_summary.title == summary.title
    evidence = model.received_evidence["beat-1"]
    assert [item.scene_id for item in evidence] == ["scene-0003"]
    assert set(evidence[0].model_dump()) == {
        "scene_id",
        "short_summary",
        "caption",
        "tags",
    }
    assert result.output_artifacts == [
        ArtifactRef(name="storyboard", path=tmp_path / "artifacts" / "storyboard.json")
    ]
    assert director.read_result("screenwriter-001") == result


def test_screenwriter_merges_and_persists_validated_storyboard(
    tmp_path: Path,
) -> None:
    index_path, summary_path, _ = _source_artifacts(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, summary_path, "screenwriter-persist")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FixedScreenwriterModel()).run_task(
            task,
            runtime,
            tmp_path / "configured-artifacts",
            target_duration_s=12,
        )
    )

    path = tmp_path / "configured-artifacts" / "storyboard.json"
    persisted = Storyboard.model_validate_json(path.read_text(encoding="utf-8"))
    assert result.status == "completed"
    assert persisted.narrative_angle == _screenplay().narrative_angle
    assert persisted.beats[0].story_content == _screenplay().beats[0].story_content
    assert persisted.beats[0].narration == _narration().beats[0].narration
    assert persisted.beats[0].source_sequence_ids == ["sequence-0002"]
    assert persisted.beats[0].evidence_scene_ids == ["scene-0003"]


@pytest.mark.parametrize("stage", ["screenplay", "narration"])
def test_screenwriter_persists_blocked_result_when_model_fails(
    tmp_path: Path, stage: str
) -> None:
    index_path, summary_path, _ = _source_artifacts(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, summary_path, f"screenwriter-{stage}-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FailingScreenwriterModel(stage)).run_task(
            task, runtime, tmp_path / "artifacts", target_duration_s=12
        )
    )

    assert result.status == "blocked"
    assert result.output_artifacts == []
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_screenwriter_blocks_invalid_narration_evidence(tmp_path: Path) -> None:
    index_path, summary_path, _ = _source_artifacts(tmp_path)
    invalid = NarrationDraft(
        beats=[
            BeatNarration(
                beat_id="beat-1",
                narration="Unsupported narration.",
                evidence_scene_ids=["scene-0001"],
            )
        ]
    )
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, summary_path, "screenwriter-invalid-evidence")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FixedScreenwriterModel(narration=invalid)).run_task(
            task, runtime, tmp_path / "artifacts", target_duration_s=12
        )
    )

    assert result.status == "blocked"
    assert "outside its source sequences" in result.summary


def test_screenwriter_blocks_when_required_artifact_cannot_load(
    tmp_path: Path,
) -> None:
    index_path, _, _ = _source_artifacts(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(
        index_path,
        tmp_path / "missing" / "story-summary.json",
        "screenwriter-load-fail",
    )
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FixedScreenwriterModel()).run_task(
            task, runtime, tmp_path / "artifacts", target_duration_s=12
        )
    )

    assert result.status == "blocked"
    assert "Screenwriting blocked:" in result.summary
    assert runtime.read_result(AgentRole.DIRECTOR, task.task_id) == result


def test_screenwriter_blocks_when_storyboard_cannot_be_persisted(
    tmp_path: Path,
) -> None:
    index_path, summary_path, _ = _source_artifacts(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    storyboard_path = artifacts / "storyboard.json"
    storyboard_path.write_text("do not overwrite", encoding="utf-8")
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, summary_path, "screenwriter-persistence-fail")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ScreenwriterAgent(FixedScreenwriterModel()).run_task(
            task, runtime, artifacts, target_duration_s=12
        )
    )

    assert result.status == "blocked"
    assert storyboard_path.read_text(encoding="utf-8") == "do not overwrite"


def test_director_cannot_run_or_submit_for_another_role(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(runtime, FootageAnalystAgent(UnusedIndexer()))
    task = TaskContext(
        task_id="footage-boundary",
        assignee=AgentRole.FOOTAGE_ANALYST,
        objective="Index footage.",
        expected_output="An index.",
        acceptance_criteria=["Return an index"],
    )
    with pytest.raises(ValueError):
        asyncio.run(director.run_screenwriter_task(task, target_duration_s=12))


def test_director_cannot_submit_a_screenwriter_result(tmp_path: Path) -> None:
    index_path, summary_path, _ = _source_artifacts(tmp_path)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(index_path, summary_path, "screenwriter-submit-boundary")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = TaskResult(
        task_id=task.task_id,
        agent=AgentRole.SCREENWRITER,
        status="completed",
        summary="Drafted a storyboard.",
    )

    with pytest.raises(PermissionError):
        runtime.submit_result(AgentRole.DIRECTOR, result)


class ScreenwriterCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses = [_screenplay(), _narration()]

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[len(self.calls) - 1]
        message = SimpleNamespace(content=response.model_dump_json())
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_screenwriter_uses_role_prompts_and_minimal_scene_evidence() -> None:
    completions = ScreenwriterCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    model = OpenAICompatibleScreenwriterModel(client=client)
    summary = StorySummary(
        title="The Last Train",
        short_summary="A traveler waits through a station blackout.",
        sequences=[
            StorySequence(
                id="sequence-0002",
                title="Blackout",
                short_summary="The station goes dark.",
                scene_ids=["scene-0003"],
            )
        ],
        acts=[
            StoryAct(
                id="act-0001",
                title="Waiting",
                short_summary="The wait becomes tense.",
                sequence_ids=["sequence-0002"],
            )
        ],
    )
    evidence = ScreenwriterSceneEvidence(
        scene_id="scene-0003",
        short_summary="Darkness leaves the traveler waiting.",
        caption="The station lights go dark.",
        tags=["darkness", "waiting"],
    )

    screenplay = asyncio.run(
        model.draft_screenplay(
            "Write a concise suspenseful edit.", ["Avoid spoilers"], summary, 12
        )
    )
    narration = asyncio.run(
        model.draft_narration(
            "Write a concise suspenseful edit.",
            ["Avoid spoilers"],
            screenplay,
            {"beat-1": [evidence]},
        )
    )

    assert screenplay == _screenplay()
    assert narration == _narration()
    assert len(completions.calls) == 2
    assert (
        completions.calls[0]["messages"][0]["content"]
        == OpenAICompatibleScreenwriterModel.SCREENPLAY_SYSTEM_PROMPT
    )
    planning_request = completions.calls[0]["messages"][1]["content"]
    assert "Complete source story hierarchy" in planning_request
    assert summary.title in planning_request
    assert "sequence-0002" in planning_request
    assert "For every beat" not in planning_request
    assert "Write the editing screenplay" not in planning_request

    assert (
        completions.calls[1]["messages"][0]["content"]
        == OpenAICompatibleScreenwriterModel.NARRATION_SYSTEM_PROMPT
    )
    narration_request = completions.calls[1]["messages"][1]["content"]
    evidence_payload = narration_request.split(
        "Selected source scene evidence by beat:\n", maxsplit=1
    )[1].split("\n\nWrite one narration", maxsplit=1)[0]
    parsed_evidence = json.loads(evidence_payload)
    assert parsed_evidence == {"beat-1": [evidence.model_dump(mode="json")]}
    assert "transcript" not in evidence_payload
    assert "dense_caption" not in evidence_payload
    assert "source_range" not in evidence_payload
