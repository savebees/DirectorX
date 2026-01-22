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

from .footage import FootageAnalystAgent


class DirectorAgent:
    """Own project authority without embedding specialist implementation details."""

    role = AgentRole.DIRECTOR

    def __init__(
        self, runtime: CoordinationRuntime, footage_analyst: FootageAnalystAgent
    ) -> None:
        self.runtime = runtime
        self.footage_analyst = footage_analyst

    def initialize_project(self, memory: ProjectMemory) -> Path:
        return self.runtime.initialize_project(self.role, memory)

    def update_project_memory(self, memory: ProjectMemory) -> Path:
        return self.runtime.update_project_memory(self.role, memory)

    def delegate(self, task: TaskContext) -> Path:
        return self.runtime.delegate(self.role, task)

    async def run_footage_task(self, task: TaskContext, video_path: Path) -> TaskResult:
        if task.assignee != AgentRole.FOOTAGE_ANALYST:
            raise ValueError("Director can only run a Footage Analyst task here")
        self.delegate(task)
        return await self.footage_analyst.run_task(task, video_path, self.runtime)

    def read_result(self, task_id: str) -> TaskResult:
        return self.runtime.read_result(self.role, task_id)

    def consult(self, request: ConsultationRequest) -> Path:
        if request.sender != self.role:
            raise ValueError("Director consultation must identify Director as sender")
        return self.runtime.consult(request)

    def record_decision(self, decision: Decision) -> Path:
        return self.runtime.record_decision(self.role, decision)
