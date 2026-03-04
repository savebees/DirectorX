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
from .grounding import GroundingAgent
from .narration import NarrationAgent
from .screenwriter import ScreenwriterAgent
from .sound import SoundAgent


class DirectorAgent:
    """Own project authority without embedding specialist implementation details."""

    role = AgentRole.DIRECTOR

    def __init__(
        self,
        runtime: CoordinationRuntime,
        footage_analyst: FootageAnalystAgent,
        screenwriter: ScreenwriterAgent | None = None,
        artifacts_dir: Path | None = None,
        *,
        screenwriter_agent: ScreenwriterAgent | None = None,
        narration_agent: NarrationAgent | None = None,
        grounding_agent: GroundingAgent | None = None,
        sound_agent: SoundAgent | None = None,
    ) -> None:
        if screenwriter is not None and screenwriter_agent is not None:
            raise ValueError("Provide only one Screenwriter agent")
        screenwriter = screenwriter_agent or screenwriter
        self.runtime = runtime
        self.footage_analyst = footage_analyst
        self.screenwriter_agent = screenwriter
        self.screenwriter = screenwriter
        self.narration_agent = narration_agent
        self.grounding_agent = grounding_agent
        self.sound_agent = sound_agent
        self.artifacts_dir = artifacts_dir or runtime.store.root / "artifacts"

    def initialize_project(self, memory: ProjectMemory) -> Path:
        return self.runtime.initialize_project(self.role, memory)

    def update_project_memory(self, memory: ProjectMemory) -> Path:
        return self.runtime.update_project_memory(self.role, memory)

    def delegate(self, task: TaskContext) -> Path:
        return self.runtime.delegate(self.role, task)

    async def run_footage_task(
        self,
        task: TaskContext,
        video_path: Path,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != AgentRole.FOOTAGE_ANALYST:
            raise ValueError("Director can only run a Footage Analyst task here")
        self.delegate(task)
        return await self.footage_analyst.run_task(
            task,
            video_path,
            self.runtime,
            artifacts_dir or self.artifacts_dir,
        )

    async def run_screenwriter_task(
        self,
        task: TaskContext,
        prompt: str | None = None,
        target_duration_s: float | None = None,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != AgentRole.SCREENWRITER:
            raise ValueError("Director can only run a Screenwriter task here")
        if self.screenwriter_agent is None:
            raise ValueError("Director has no Screenwriter agent")
        self.delegate(task)
        await self.screenwriter_agent.run_task(
            task,
            self.runtime,
            artifacts_dir or self.artifacts_dir,
            prompt=prompt,
            target_duration_s=target_duration_s,
        )
        return self.read_result(task.task_id)

    async def run_narration_task(
        self,
        task: TaskContext,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != AgentRole.NARRATION:
            raise ValueError("Director can only run a Narration task here")
        if self.narration_agent is None:
            raise ValueError("Director has no Narration agent")
        self.delegate(task)
        await self.narration_agent.run_task(
            task,
            self.runtime,
            artifacts_dir or self.artifacts_dir,
        )
        return self.read_result(task.task_id)

    async def run_grounding_task(
        self,
        task: TaskContext,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != AgentRole.GROUNDING:
            raise ValueError("Director can only run a Grounding task here")
        if self.grounding_agent is None:
            raise ValueError("Director has no Grounding agent")
        self.delegate(task)
        await self.grounding_agent.run_task(
            task,
            self.runtime,
            artifacts_dir or self.artifacts_dir,
        )
        return self.read_result(task.task_id)

    async def run_sound_task(
        self,
        task: TaskContext,
        artifacts_dir: Path | None = None,
    ) -> TaskResult:
        if task.assignee != AgentRole.SOUND:
            raise ValueError("Director can only run a Sound task here")
        if self.sound_agent is None:
            raise ValueError("Director has no Sound agent")
        self.delegate(task)
        await self.sound_agent.run_task(
            task,
            self.runtime,
            artifacts_dir or self.artifacts_dir,
        )
        return self.read_result(task.task_id)

    def read_result(self, task_id: str) -> TaskResult:
        return self.runtime.read_result(self.role, task_id)

    def consult(self, request: ConsultationRequest) -> Path:
        if request.sender != self.role:
            raise ValueError("Director consultation must identify Director as sender")
        return self.runtime.consult(request)

    def record_decision(self, decision: Decision) -> Path:
        return self.runtime.record_decision(self.role, decision)
