from __future__ import annotations

from pathlib import Path

from directorx.coordination import (
    AgentRole,
    ConsultationRequest,
    Decision,
    ProjectMemory,
    TaskContext,
    TaskResult,
)
from directorx.coordination.runtime import CoordinationRuntime


class DirectorAgent:
    """Own project authority without embedding specialist implementation details."""

    role = AgentRole.DIRECTOR

    def __init__(self, runtime: CoordinationRuntime) -> None:
        self.runtime = runtime

    def initialize_project(self, memory: ProjectMemory) -> Path:
        return self.runtime.initialize_project(self.role, memory)

    def update_project_memory(self, memory: ProjectMemory) -> Path:
        return self.runtime.update_project_memory(self.role, memory)

    def delegate(self, task: TaskContext) -> Path:
        return self.runtime.delegate(self.role, task)

    def read_result(self, task_id: str) -> TaskResult:
        return self.runtime.read_result(self.role, task_id)

    def consult(self, request: ConsultationRequest) -> Path:
        if request.sender != self.role:
            raise ValueError("Director consultation must identify Director as sender")
        return self.runtime.consult(request)

    def record_decision(self, decision: Decision) -> Path:
        return self.runtime.record_decision(self.role, decision)
