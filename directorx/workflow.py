from __future__ import annotations

from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from directorx.agents import DirectorAgent
from directorx.coordination import (
    AgentRole,
    ArtifactRef,
    ProjectMemory,
    TaskContext,
    TaskResult,
)


def _merge_dicts(
    left: dict[str, object] | None, right: dict[str, object] | None
) -> dict[str, object]:
    """Merge independent branch updates without losing sibling results."""
    return {**(left or {}), **(right or {})}


def _last_value(left, right):
    """Allow informational scalar fields to be updated by parallel nodes."""
    return right


def _merge_error(left: str | None, right: str | None) -> str | None:
    """Keep a blocked branch visible when a sibling branch completes."""
    return right if right is not None else left


class WorkflowState(TypedDict, total=False):
    """Small, serializable state shared by the Director graph."""

    project_id: str
    video_path: str
    brief: str
    constraints: list[str]
    target_duration_s: float
    music_index_path: str | None
    artifacts: Annotated[dict[str, str], _merge_dicts]
    statuses: Annotated[dict[str, str], _merge_dicts]
    task_results: Annotated[dict[str, dict], _merge_dicts]
    current_stage: Annotated[str, _last_value]
    error: Annotated[str | None, _merge_error]
    review_passed: bool


class DirectorWorkflow:
    """LangGraph supervisor for the end-to-end DirectorX editing workflow."""

    def __init__(
        self,
        director: DirectorAgent,
        *,
        artifacts_dir: Path,
        checkpoint_path: Path | None = None,
        max_revisions: int = 0,
    ) -> None:
        if max_revisions != 0:
            raise ValueError(
                "Automatic revisions are not implemented; max_revisions must be 0"
            )
        self.director = director
        self.artifacts_dir = artifacts_dir
        self.checkpoint_path = checkpoint_path
        self.max_revisions = max_revisions

    def build_graph(self, checkpointer=None):
        builder = StateGraph(WorkflowState)
        builder.add_node("footage", self._footage)
        builder.add_node("screenwriter", self._screenwriter)
        builder.add_node("narration_gate", self._narration_gate)
        builder.add_node("grounding_gate", self._grounding_gate)
        builder.add_node("narration", self._narration)
        builder.add_node("grounding", self._grounding)
        builder.add_node("sound_gate", self._sound_gate)
        builder.add_node("sound", self._sound)
        builder.add_node("media_join", self._media_join)
        builder.add_node("editor", self._editor)
        builder.add_node("render", self._render)
        builder.add_node("review", self._review)

        builder.add_edge(START, "footage")
        builder.add_conditional_edges(
            "footage",
            lambda state: self._next_stage(state, "footage"),
            {"continue": "screenwriter", "stop": END},
        )
        builder.add_edge("screenwriter", "narration_gate")
        builder.add_edge("screenwriter", "grounding_gate")
        builder.add_conditional_edges(
            "narration_gate",
            lambda state: self._stage_gate_route(state, "screenwriter"),
            {"continue": "narration", "stop": END},
        )
        builder.add_conditional_edges(
            "grounding_gate",
            lambda state: self._stage_gate_route(state, "screenwriter"),
            {"continue": "grounding", "stop": END},
        )
        builder.add_edge("narration", "sound_gate")
        builder.add_conditional_edges(
            "sound_gate",
            lambda state: self._stage_gate_route(state, "narration"),
            {"continue": "sound", "stop": "media_join"},
        )
        builder.add_edge("grounding", "media_join")
        builder.add_edge("sound", "media_join")
        builder.add_conditional_edges(
            "media_join",
            self._media_join_route,
            {"continue": "editor", "stop": END},
        )
        builder.add_conditional_edges(
            "editor",
            lambda state: self._next_stage(state, "editor"),
            {"continue": "render", "stop": END},
        )
        builder.add_conditional_edges(
            "render",
            lambda state: self._next_stage(state, "render"),
            {"continue": "review", "stop": END},
        )
        builder.add_edge("review", END)
        return builder.compile(checkpointer=checkpointer)

    async def run(
        self,
        *,
        project_id: str,
        video_path: Path,
        brief: str,
        constraints: list[str] | None = None,
        target_duration_s: float = 60.0,
        music_index_path: Path | None = None,
    ) -> WorkflowState:
        if (
            not project_id.strip()
            or project_id in {".", ".."}
            or Path(project_id).name != project_id
        ):
            raise ValueError("project_id must be a non-empty directory-safe identifier")
        if not brief.strip():
            raise ValueError("brief cannot be empty")
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if target_duration_s <= 0:
            raise ValueError("target_duration_s must be positive")

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_memory(
            project_id,
            brief,
            constraints or [],
        )
        initial: WorkflowState = {
            "project_id": project_id,
            "video_path": str(video_path.resolve()),
            "brief": brief,
            "constraints": constraints or [],
            "target_duration_s": target_duration_s,
            "music_index_path": (
                str(music_index_path.resolve())
                if music_index_path is not None
                else None
            ),
            "artifacts": {"source-video": str(video_path.resolve())},
            "statuses": {},
            "task_results": {},
            "current_stage": "",
            "error": None,
            "review_passed": False,
        }
        config = {"configurable": {"thread_id": project_id}}
        if self.checkpoint_path is None:
            return await self.build_graph().ainvoke(initial, config=config)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(
            str(self.checkpoint_path)
        ) as saver:
            await saver.setup()
            return await self.build_graph(saver).ainvoke(initial, config=config)

    async def _footage(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "footage",
            AgentRole.FOOTAGE_ANALYST,
            "Index the source video and build a citation-backed story hierarchy.",
            [self._ref(state, "source-video")],
            "A searchable video index and story summary.",
        )
        result = await self.director.run_footage_task(
            task,
            Path(state["video_path"]),
            self.artifacts_dir,
        )
        return self._record(state, "footage", result)

    async def _screenwriter(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "screenwriter",
            AgentRole.SCREENWRITER,
            state["brief"],
            [
                self._ref(state, "video-index"),
                self._ref(state, "story-summary"),
            ],
            "A validated storyboard with beat-level narration.",
        )
        result = await self.director.run_screenwriter_task(
            task,
            target_duration_s=state["target_duration_s"],
            artifacts_dir=self.artifacts_dir,
        )
        return self._record(state, "screenwriter", result)

    async def _narration(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "narration",
            AgentRole.NARRATION,
            "Synthesize one narration audio segment for every storyboard beat.",
            [self._ref(state, "storyboard")],
            "A narration manifest and audio files.",
        )
        result = await self.director.run_narration_task(task, self.artifacts_dir)
        return self._record(state, "narration", result)

    async def _grounding(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "grounding",
            AgentRole.GROUNDING,
            "Ground every storyboard beat to an exact source-video interval.",
            [
                self._ref(state, "video-index"),
                self._ref(state, "scene-search-database"),
                self._ref(state, "story-summary"),
                self._ref(state, "storyboard"),
            ],
            "A grounding manifest with visually verified source clips.",
        )
        result = await self.director.run_grounding_task(task, self.artifacts_dir)
        return self._record(state, "grounding", result)

    async def _sound(self, state: WorkflowState) -> WorkflowState:
        inputs = [
            self._ref(state, "storyboard"),
            self._ref(state, "narration-manifest"),
        ]
        if state.get("music_index_path") is not None:
            inputs.append(
                ArtifactRef(name="music-index", path=Path(state["music_index_path"]))
            )
        task = self._task(
            state,
            "sound",
            AgentRole.SOUND,
            "Select one background music track for the complete edit.",
            inputs,
            "A validated sound plan.",
        )
        result = await self.director.run_sound_task(task, self.artifacts_dir)
        return self._record(state, "sound", result)

    async def _render(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "render",
            AgentRole.RENDER,
            "Render a playable MP4 from grounded clips, narration, and sound.",
            [
                self._ref(state, "edit-timeline"),
                self._ref(state, "narration-manifest"),
                self._ref(state, "sound-plan"),
            ],
            "A playable final.mp4 video.",
        )
        result = await self.director.run_render_task(task, self.artifacts_dir)
        return self._record(state, "render", result)

    async def _editor(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "editor",
            AgentRole.EDITOR,
            (
                "Build the final edit timeline from narrative beats, measured "
                "narration, verified clips, and the selected sound plan."
            ),
            [
                self._ref(state, "storyboard"),
                self._ref(state, "grounding-manifest"),
                self._ref(state, "narration-manifest"),
                self._ref(state, "sound-plan"),
            ],
            "A duration-safe timeline with bounded freeze and voice coverage.",
        )
        result = await self.director.run_editor_task(task, self.artifacts_dir)
        return self._record(state, "editor", result)

    async def _review(self, state: WorkflowState) -> WorkflowState:
        task = self._task(
            state,
            "review",
            AgentRole.REVIEW,
            "Review the rendered video for obvious visual defects.",
            [
                self._ref(state, "rendered-video"),
                self._ref(state, "storyboard"),
                self._ref(state, "narration-manifest"),
                self._ref(state, "edit-timeline"),
            ],
            "A validated visual and narrative review report.",
        )
        result = await self.director.run_review_task(task, self.artifacts_dir)
        update = self._record(state, "review", result)
        update["review_passed"] = result.status == "completed"
        if result.status == "completed":
            memory = self.director.runtime.store.read_project_memory()
            approved = {
                name: ArtifactRef(name=name, path=Path(path))
                for name, path in update["artifacts"].items()
            }
            self.director.update_project_memory(
                ProjectMemory(
                    project_id=memory.project_id,
                    brief=memory.brief,
                    constraints=memory.constraints,
                    approved_artifacts=approved,
                )
            )
        return update

    @staticmethod
    async def _media_join(state: WorkflowState) -> WorkflowState:
        return {"current_stage": "media_join"}

    @staticmethod
    def _media_join_route(state: WorkflowState) -> str:
        return (
            "continue"
            if state.get("statuses", {}).get("grounding") == "completed"
            and state.get("statuses", {}).get("sound") == "completed"
            else "stop"
        )

    def _ensure_project_memory(
        self, project_id: str, brief: str, constraints: list[str]
    ) -> None:
        path = self.director.runtime.store.root / "project-memory.json"
        memory = ProjectMemory(
            project_id=project_id,
            brief=brief,
            constraints=constraints,
        )
        if not path.exists():
            self.director.initialize_project(memory)
            return
        existing = self.director.runtime.store.read_project_memory()
        if existing.project_id != project_id:
            raise ValueError(
                "Coordination directory belongs to project "
                f"{existing.project_id!r}; choose a distinct coordination directory"
            )
        if existing.brief != brief or existing.constraints != constraints:
            raise ValueError(
                "Project memory already exists with a different brief or constraints"
            )

    @staticmethod
    def _next_stage(state: WorkflowState, stage: str) -> str:
        return (
            "continue"
            if state.get("statuses", {}).get(stage) == "completed"
            else "stop"
        )

    @staticmethod
    def _stage_gate_route(state: WorkflowState, stage: str) -> str:
        return (
            "continue"
            if state.get("statuses", {}).get(stage) == "completed"
            else "stop"
        )

    @staticmethod
    async def _narration_gate(state: WorkflowState) -> WorkflowState:
        return {"current_stage": "narration_gate"}

    @staticmethod
    async def _grounding_gate(state: WorkflowState) -> WorkflowState:
        return {"current_stage": "grounding_gate"}

    @staticmethod
    async def _sound_gate(state: WorkflowState) -> WorkflowState:
        return {"current_stage": "sound_gate"}

    def _record(
        self, state: WorkflowState, stage: str, result: TaskResult
    ) -> WorkflowState:
        artifacts = dict(state.get("artifacts", {}))
        for artifact in result.output_artifacts:
            artifacts[artifact.name] = str(artifact.path)
        statuses = dict(state.get("statuses", {}))
        statuses[stage] = result.status
        task_results = dict(state.get("task_results", {}))
        task_results[result.task_id] = result.model_dump(mode="json")
        return {
            "artifacts": artifacts,
            "statuses": statuses,
            "task_results": task_results,
            "current_stage": stage,
            "error": None if result.status == "completed" else result.summary,
        }

    @staticmethod
    def _task(
        state: WorkflowState,
        stage: str,
        assignee: AgentRole,
        objective: str,
        input_artifacts: list[ArtifactRef],
        expected_output: str,
    ) -> TaskContext:
        return TaskContext(
            task_id=f"{state['project_id']}-{stage}",
            assignee=assignee,
            objective=objective,
            constraints=state.get("constraints", []),
            input_artifacts=input_artifacts,
            expected_output=expected_output,
            acceptance_criteria=[
                "Return a validated artifact or a clear blocked result"
            ],
        )

    @staticmethod
    def _ref(state: WorkflowState, name: str) -> ArtifactRef:
        try:
            path = state["artifacts"][name]
        except KeyError as exc:
            raise ValueError(f"Required workflow artifact is missing: {name}") from exc
        return ArtifactRef(name=name, path=Path(path))
