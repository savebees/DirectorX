from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .contracts import (
    ConsultationRequest,
    ConsultationResponse,
    Decision,
    ProjectMemory,
    TaskContext,
    TaskResult,
)


class ContextStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tasks_dir = root / "tasks"
        self.consultations_dir = root / "consultations"
        self.decisions_dir = root / "decisions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(exist_ok=True)
        self.consultations_dir.mkdir(exist_ok=True)
        self.decisions_dir.mkdir(exist_ok=True)

    def create_project_memory(self, memory: ProjectMemory) -> Path:
        return self._create(self.root / "project-memory.json", memory)

    def update_project_memory(self, memory: ProjectMemory) -> Path:
        path = self.root / "project-memory.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return self._write(path, memory)

    def read_project_memory(self) -> ProjectMemory:
        return ProjectMemory.model_validate_json(
            (self.root / "project-memory.json").read_text(encoding="utf-8")
        )

    def write_task(self, task: TaskContext) -> Path:
        return self._create(self.tasks_dir / f"{task.task_id}.json", task)

    def read_task(self, task_id: str) -> TaskContext:
        return TaskContext.model_validate_json(
            (self.tasks_dir / f"{task_id}.json").read_text(encoding="utf-8")
        )

    def write_task_result(self, result: TaskResult) -> Path:
        return self._create(
            self.tasks_dir / f"{result.task_id}.result.json",
            result,
        )

    def read_task_result(self, task_id: str) -> TaskResult:
        return TaskResult.model_validate_json(
            (self.tasks_dir / f"{task_id}.result.json").read_text(encoding="utf-8")
        )

    def write_consultation(self, request: ConsultationRequest) -> Path:
        return self._create(
            self.consultations_dir / f"{request.consultation_id}.request.json",
            request,
        )

    def read_consultation(self, consultation_id: str) -> ConsultationRequest:
        return ConsultationRequest.model_validate_json(
            (self.consultations_dir / f"{consultation_id}.request.json").read_text(
                encoding="utf-8"
            )
        )

    def write_consultation_response(self, response: ConsultationResponse) -> Path:
        path = self.consultations_dir / f"{response.consultation_id}.response.json"
        return self._create(path, response)

    def write_decision(self, decision: Decision) -> Path:
        return self._create(
            self.decisions_dir / f"{decision.decision_id}.json", decision
        )

    @classmethod
    def _create(cls, path: Path, value: BaseModel) -> Path:
        if path.exists():
            raise FileExistsError(path)
        return cls._write(path, value)

    @staticmethod
    def _write(path: Path, value: BaseModel) -> Path:
        path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path
