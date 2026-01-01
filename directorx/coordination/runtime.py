from __future__ import annotations

from pathlib import Path

from .contracts import (
    AgentRole,
    ConsultationRequest,
    ConsultationResponse,
    Decision,
    ProjectMemory,
    TaskContext,
    TaskResult,
)
from .policy import can_consult
from .store import ContextStore


class CoordinationRuntime:
    """Enforce Director authority and context boundaries between agents."""

    def __init__(self, root: Path) -> None:
        self.store = ContextStore(root)

    def initialize_project(self, actor: AgentRole, memory: ProjectMemory) -> Path:
        self._require_director(actor)
        return self.store.create_project_memory(memory)

    def update_project_memory(self, actor: AgentRole, memory: ProjectMemory) -> Path:
        self._require_director(actor)
        return self.store.update_project_memory(memory)

    def read_project_memory(self, actor: AgentRole) -> ProjectMemory:
        return self.store.read_project_memory()

    def delegate(self, actor: AgentRole, task: TaskContext) -> Path:
        self._require_director(actor)
        if task.assignee == AgentRole.DIRECTOR:
            raise ValueError("Director does not delegate specialist tasks to itself")
        return self.store.write_task(task)

    def read_task(self, actor: AgentRole, task_id: str) -> TaskContext:
        task = self.store.read_task(task_id)
        if actor not in {AgentRole.DIRECTOR, task.assignee}:
            raise PermissionError(f"{actor} cannot read task {task_id}")
        return task

    def submit_result(self, actor: AgentRole, result: TaskResult) -> Path:
        task = self.store.read_task(result.task_id)
        if actor != task.assignee or result.agent != task.assignee:
            raise PermissionError(f"{actor} cannot complete task {result.task_id}")
        return self.store.write_task_result(result)

    def read_result(self, actor: AgentRole, task_id: str) -> TaskResult:
        task = self.store.read_task(task_id)
        if actor not in {AgentRole.DIRECTOR, task.assignee}:
            raise PermissionError(f"{actor} cannot read result for task {task_id}")
        return self.store.read_task_result(task_id)

    def consult(self, request: ConsultationRequest) -> Path:
        if not can_consult(request.sender, request.recipient):
            raise PermissionError(
                f"{request.sender} cannot consult {request.recipient}"
            )
        return self.store.write_consultation(request)

    def respond(self, response: ConsultationResponse) -> Path:
        request = self.store.read_consultation(response.consultation_id)
        if response.sender != request.recipient or response.recipient != request.sender:
            raise PermissionError(
                f"Invalid response direction for {response.consultation_id}"
            )
        return self.store.write_consultation_response(response)

    def record_decision(self, actor: AgentRole, decision: Decision) -> Path:
        self._require_director(actor)
        return self.store.write_decision(decision)

    @staticmethod
    def _require_director(actor: AgentRole) -> None:
        if actor != AgentRole.DIRECTOR:
            raise PermissionError(f"{actor} does not have Director authority")
