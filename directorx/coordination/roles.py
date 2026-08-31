from __future__ import annotations

from dataclasses import dataclass

from .contracts import AgentRole


@dataclass(frozen=True)
class RoleDefinition:
    title: str
    responsibility: str


ROLE_DEFINITIONS = {
    AgentRole.DIRECTOR: RoleDefinition(
        title="Director Agent",
        responsibility="Own the creative brief, delegation, revisions, and approval.",
    ),
    AgentRole.FOOTAGE_ANALYST: RoleDefinition(
        title="Footage Analyst Agent",
        responsibility="Establish source facts and answer footage evidence questions.",
    ),
    AgentRole.SCREENWRITER: RoleDefinition(
        title="Screenwriter Agent",
        responsibility="Own narrative structure and narration text.",
    ),
    AgentRole.NARRATION: RoleDefinition(
        title="Narration Agent",
        responsibility="Own voice delivery, timing, pronunciation, and subtitles.",
    ),
    AgentRole.GROUNDING: RoleDefinition(
        title="Grounding Agent",
        responsibility="Locate exact source clips that satisfy visual intent.",
    ),
    AgentRole.SOUND: RoleDefinition(
        title="Sound Agent",
        responsibility="Own music selection, sound design, and mix intent.",
    ),
    AgentRole.EDITOR: RoleDefinition(
        title="Editor Agent",
        responsibility=(
            "Reconcile narration timing, grounded footage, pacing, and the final "
            "edit timeline."
        ),
    ),
    AgentRole.RENDER: RoleDefinition(
        title="Render Agent",
        responsibility="Compile grounded clips, narration, and sound into a video.",
    ),
    AgentRole.REVIEW: RoleDefinition(
        title="Review Agent",
        responsibility="Independently review approved artifacts and report issues.",
    ),
}
