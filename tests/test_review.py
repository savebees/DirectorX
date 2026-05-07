from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from directorx.agents import DirectorAgent, FootageAnalystAgent, ReviewAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    CoordinationRuntime,
    TaskContext,
    TaskResult,
)
from directorx.core.models import GroundingFrame, ReviewIssue, ReviewReport
from tests.fakes import FixedReviewFrameExtractor, FixedReviewModel


class UnusedIndexer:
    async def build(self, video_path: Path):
        raise AssertionError("The footage agent is not used in this test")


def _task(video: Path, task_id: str = "review-001") -> TaskContext:
    return TaskContext(
        task_id=task_id,
        assignee=AgentRole.REVIEW,
        objective="Review the finished video for obvious defects.",
        input_artifacts=[ArtifactRef(name="rendered-video", path=video)],
        expected_output="One persisted review report.",
        acceptance_criteria=["Report whether the finished video is acceptable"],
    )


def _frames(tmp_path: Path) -> list[GroundingFrame]:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    return [GroundingFrame(id="review-0001", timestamp_s=0.5, path=frame)]


def _video(tmp_path: Path) -> Path:
    # The fake probe is patched in tests; this only verifies artifact boundaries.
    video = tmp_path / "final.mp4"
    video.write_bytes(b"video")
    return video


def test_review_agent_persists_successful_report_and_uses_one_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _video(tmp_path)
    monkeypatch.setattr("directorx.agents.review._probe_duration", lambda _: 10.0)
    model = FixedReviewModel()
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(video)
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ReviewAgent(
            model,
            FixedReviewFrameExtractor(_frames(tmp_path)),
        ).run_task(task, runtime, tmp_path / "artifacts")
    )

    report_path = tmp_path / "artifacts" / "review.json"
    assert result.status == "completed"
    assert result.agent == AgentRole.REVIEW
    assert result.output_artifacts == [ArtifactRef(name="review", path=report_path)]
    assert ReviewReport.model_validate_json(report_path.read_text()).passed
    assert len(model.calls) == 1
    payload = json.loads(
        (tmp_path / "coordination" / "tasks" / "review-001.result.json").read_text()
    )
    assert set(payload) == {"task_id", "agent", "status", "summary", "output_artifacts"}


def test_review_issues_return_blocked_but_persist_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("directorx.agents.review._probe_duration", lambda _: 10.0)
    report = ReviewReport(
        passed=False,
        summary="The edit has a visible discontinuity.",
        issues=[ReviewIssue(timestamp_s=4.0, description="Abrupt broken cut")],
    )
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(_video(tmp_path), "review-issues")
    runtime.delegate(AgentRole.DIRECTOR, task)

    result = asyncio.run(
        ReviewAgent(
            FixedReviewModel(report), FixedReviewFrameExtractor(_frames(tmp_path))
        ).run_task(task, runtime, tmp_path / "artifacts")
    )

    assert result.status == "blocked"
    assert ReviewReport.model_validate_json(
        (tmp_path / "artifacts" / "review.json").read_text()
    ).issues


def test_review_model_failure_persists_blocked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("directorx.agents.review._probe_duration", lambda _: 10.0)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(_video(tmp_path), "review-failure")
    runtime.delegate(AgentRole.DIRECTOR, task)
    result = asyncio.run(
        ReviewAgent(
            FixedReviewModel(failure=RuntimeError("review unavailable")),
            FixedReviewFrameExtractor(_frames(tmp_path)),
        ).run_task(task, runtime, tmp_path / "artifacts")
    )
    assert result.status == "blocked"
    assert "review unavailable" in result.summary


def test_review_rejects_non_review_task(tmp_path: Path) -> None:
    task = _task(_video(tmp_path))
    task = task.model_copy(update={"assignee": AgentRole.SOUND})
    with pytest.raises(ValueError):
        asyncio.run(
            ReviewAgent(
                FixedReviewModel(), FixedReviewFrameExtractor(_frames(tmp_path))
            ).run_task(
                task,
                CoordinationRuntime(tmp_path / "coordination"),
                tmp_path / "artifacts",
            )
        )


def test_director_cannot_submit_review_result(tmp_path: Path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    task = _task(_video(tmp_path), "review-boundary")
    runtime.delegate(AgentRole.DIRECTOR, task)
    with pytest.raises(PermissionError):
        runtime.submit_result(
            AgentRole.DIRECTOR,
            TaskResult(
                task_id=task.task_id,
                agent=AgentRole.REVIEW,
                status="completed",
                summary="Approved",
            ),
        )


def test_director_delegates_review_and_reads_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("directorx.agents.review._probe_duration", lambda _: 10.0)
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(
        runtime,
        FootageAnalystAgent(UnusedIndexer()),
        review_agent=ReviewAgent(
            FixedReviewModel(), FixedReviewFrameExtractor(_frames(tmp_path))
        ),
        artifacts_dir=tmp_path / "artifacts",
    )
    result = asyncio.run(director.run_review_task(_task(_video(tmp_path))))
    assert result.agent == AgentRole.REVIEW
