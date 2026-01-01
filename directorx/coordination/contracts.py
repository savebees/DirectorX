from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRole(StrEnum):
    DIRECTOR = "director"
    FOOTAGE_ANALYST = "footage_analyst"
    SCREENWRITER = "screenwriter"
    NARRATION = "narration"
    GROUNDING = "grounding"
    SOUND = "sound"
    REVIEW = "review"


class ArtifactRef(StrictModel):
    name: str
    path: Path
    version: int = Field(gt=0)


class ProjectMemory(StrictModel):
    project_id: str
    brief: str
    constraints: list[str] = Field(default_factory=list)
    approved_artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)


class TaskContext(StrictModel):
    task_id: str
    assigned_by: Literal[AgentRole.DIRECTOR] = AgentRole.DIRECTOR
    assignee: AgentRole
    objective: str
    constraints: list[str] = Field(default_factory=list)
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    expected_output: str
    acceptance_criteria: list[str]
    revision_of: str | None = None


class TaskResult(StrictModel):
    task_id: str
    agent: AgentRole
    status: Literal["completed", "blocked"]
    summary: str
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    change_requests: list[str] = Field(default_factory=list)


class ConsultationRequest(StrictModel):
    consultation_id: str
    sender: AgentRole
    recipient: AgentRole
    question: str
    reason: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    required_answer: str


class ConsultationResponse(StrictModel):
    consultation_id: str
    sender: AgentRole
    recipient: AgentRole
    answer: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    escalate_to_director: bool = False


class Decision(StrictModel):
    decision_id: str
    made_by: Literal[AgentRole.DIRECTOR] = AgentRole.DIRECTOR
    summary: str
    rationale: str
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
