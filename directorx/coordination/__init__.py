"""Director-led multi-agent coordination contracts and runtime."""

from .contracts import (
    AgentRole,
    ArtifactRef,
    ConsultationRequest,
    ConsultationResponse,
    Decision,
    ProjectMemory,
    TaskContext,
    TaskResult,
)
from .policy import CONSULTATION_WHITELIST, can_consult
from .roles import ROLE_DEFINITIONS, RoleDefinition
from .runtime import CoordinationRuntime

__all__ = [
    "AgentRole",
    "ArtifactRef",
    "ConsultationRequest",
    "ConsultationResponse",
    "CONSULTATION_WHITELIST",
    "CoordinationRuntime",
    "Decision",
    "ProjectMemory",
    "ROLE_DEFINITIONS",
    "RoleDefinition",
    "TaskContext",
    "TaskResult",
    "can_consult",
]
