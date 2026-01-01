from __future__ import annotations

from .contracts import AgentRole

CONSULTATION_WHITELIST = {
    AgentRole.DIRECTOR: frozenset(
        role for role in AgentRole if role != AgentRole.DIRECTOR
    ),
    AgentRole.FOOTAGE_ANALYST: frozenset({AgentRole.DIRECTOR}),
    AgentRole.SCREENWRITER: frozenset({AgentRole.FOOTAGE_ANALYST, AgentRole.DIRECTOR}),
    AgentRole.NARRATION: frozenset({AgentRole.SCREENWRITER, AgentRole.DIRECTOR}),
    AgentRole.GROUNDING: frozenset(
        {AgentRole.FOOTAGE_ANALYST, AgentRole.SCREENWRITER, AgentRole.DIRECTOR}
    ),
    AgentRole.SOUND: frozenset({AgentRole.NARRATION, AgentRole.DIRECTOR}),
    AgentRole.REVIEW: frozenset({AgentRole.DIRECTOR}),
}


def can_consult(sender: AgentRole, recipient: AgentRole) -> bool:
    return recipient in CONSULTATION_WHITELIST[sender]
