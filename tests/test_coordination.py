import json

import pytest

from directorx.agents import DirectorAgent, FootageAnalystAgent
from directorx.coordination import (
    AgentRole,
    ConsultationRequest,
    ConsultationResponse,
    CoordinationRuntime,
    Decision,
    ProjectMemory,
    TaskContext,
    TaskResult,
)


class UnusedIndexer:
    async def build(self, video_path):
        raise AssertionError("The footage agent is not used in this test")


def test_only_director_controls_project_memory_and_tasks(tmp_path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    director = DirectorAgent(runtime, FootageAnalystAgent(UnusedIndexer()))
    memory = ProjectMemory(
        project_id="demo",
        brief="Create a suspenseful one-minute edit.",
        constraints=["Avoid major spoilers"],
    )
    director.initialize_project(memory)
    with pytest.raises(FileExistsError):
        director.initialize_project(memory)

    assert runtime.read_project_memory(AgentRole.SCREENWRITER) == memory
    with pytest.raises(PermissionError):
        runtime.update_project_memory(AgentRole.SCREENWRITER, memory)

    task = TaskContext(
        task_id="write-001",
        assignee=AgentRole.SCREENWRITER,
        objective="Draft the narration structure.",
        expected_output="A screenplay artifact.",
        acceptance_criteria=["Ground every plot claim in source evidence"],
    )
    director.delegate(task)
    with pytest.raises(FileExistsError):
        director.delegate(task)
    assert runtime.read_task(AgentRole.SCREENWRITER, task.task_id) == task
    with pytest.raises(PermissionError):
        runtime.read_task(AgentRole.SOUND, task.task_id)
    with pytest.raises(PermissionError):
        runtime.delegate(AgentRole.GROUNDING, task)

    result = TaskResult(
        task_id=task.task_id,
        agent=AgentRole.SCREENWRITER,
        status="completed",
        summary="Drafted three narrative beats.",
    )
    runtime.submit_result(AgentRole.SCREENWRITER, result)
    assert director.read_result(task.task_id) == result
    with pytest.raises(FileExistsError):
        runtime.submit_result(AgentRole.SCREENWRITER, result)
    with pytest.raises(PermissionError):
        runtime.submit_result(AgentRole.GROUNDING, result)


def test_consultation_whitelist_and_single_response(tmp_path) -> None:
    runtime = CoordinationRuntime(tmp_path / "coordination")
    request = ConsultationRequest(
        consultation_id="consult-001",
        sender=AgentRole.GROUNDING,
        recipient=AgentRole.FOOTAGE_ANALYST,
        question="Does the source contain the requested explosion shot?",
        reason="The current shot request requires visual evidence.",
        required_answer="Answer yes or no and provide source timestamps.",
    )
    runtime.consult(request)
    response = ConsultationResponse(
        consultation_id=request.consultation_id,
        sender=AgentRole.FOOTAGE_ANALYST,
        recipient=AgentRole.GROUNDING,
        answer="No matching explosion appears in the indexed source.",
        escalate_to_director=True,
    )
    runtime.respond(response)

    with pytest.raises(FileExistsError):
        runtime.respond(response)
    with pytest.raises(PermissionError):
        runtime.consult(
            request.model_copy(
                update={
                    "consultation_id": "consult-002",
                    "sender": AgentRole.SOUND,
                }
            )
        )


def test_review_is_isolated_and_context_files_stay_scoped(tmp_path) -> None:
    root = tmp_path / "coordination"
    runtime = CoordinationRuntime(root)
    review_request = ConsultationRequest(
        consultation_id="review-001",
        sender=AgentRole.REVIEW,
        recipient=AgentRole.DIRECTOR,
        question="Should spoiler severity be evaluated against the approved brief?",
        reason="The review requires the authoritative constraint.",
        required_answer="Confirm the approved spoiler constraint.",
    )
    runtime.consult(review_request)

    with pytest.raises(PermissionError):
        runtime.consult(
            review_request.model_copy(
                update={
                    "consultation_id": "review-002",
                    "recipient": AgentRole.SCREENWRITER,
                }
            )
        )

    request_payload = json.loads(
        (root / "consultations" / "review-001.request.json").read_text()
    )
    assert set(request_payload) == {
        "consultation_id",
        "sender",
        "recipient",
        "question",
        "reason",
        "artifact_refs",
        "required_answer",
    }

    decision = Decision(
        decision_id="decision-001",
        summary="Keep the approved spoiler boundary.",
        rationale="It is part of the user brief.",
    )
    runtime.record_decision(AgentRole.DIRECTOR, decision)
    with pytest.raises(PermissionError):
        runtime.record_decision(AgentRole.REVIEW, decision)
