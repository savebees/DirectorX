from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from pydantic import BaseModel, Field

from directorx.services.structured_output import request_structured_output


class Decision(BaseModel):
    approved: bool
    score: float = Field(ge=0, le=1)


class FakeCompletions:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self.messages = messages
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = self.messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(*messages: SimpleNamespace):
    completions = FakeCompletions(list(messages))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_json_object_mode_validates_and_repairs_once() -> None:
    client, completions = _client(
        SimpleNamespace(content='{"approved":true,"score":0.2}', refusal=None),
        SimpleNamespace(content='{"approved":true,"score":0.9}', refusal=None),
    )

    def require_confident_approval(decision: Decision) -> None:
        if decision.approved and decision.score < 0.8:
            raise ValueError("An approval requires score >= 0.8")

    result = asyncio.run(
        request_structured_output(
            client,
            model="model",
            messages=[{"role": "user", "content": "Decide."}],
            schema=Decision,
            schema_name="decision",
            max_tokens=100,
            temperature=0,
            mode="json_object",
            validation_retries=1,
            validate=require_confident_approval,
        )
    )

    assert result == Decision(approved=True, score=0.9)
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert (
        "failed local schema validation"
        in completions.calls[1]["messages"][-1]["content"]
    )


def test_tool_call_mode_validates_function_arguments() -> None:
    arguments = json.dumps({"approved": False, "score": 0.25})
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="decision", arguments=arguments)
    )
    client, completions = _client(
        SimpleNamespace(content=None, refusal=None, tool_calls=[tool_call])
    )

    result = asyncio.run(
        request_structured_output(
            client,
            model="model",
            messages=[{"role": "user", "content": "Decide."}],
            schema=Decision,
            schema_name="decision",
            max_tokens=100,
            temperature=0,
            mode="tool_call",
        )
    )

    assert result == Decision(approved=False, score=0.25)
    assert completions.calls[0]["tool_choice"]["function"]["name"] == "decision"
    assert completions.calls[0]["tools"][0]["function"]["parameters"]["required"] == [
        "approved",
        "score",
    ]
