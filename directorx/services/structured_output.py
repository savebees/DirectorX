from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

StructuredOutputMode = Literal["json_object", "tool_call", "prompted_json"]
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when a model cannot produce a locally valid typed response."""


async def request_structured_output(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    schema: type[StructuredModel],
    schema_name: str,
    max_tokens: int,
    temperature: float,
    mode: StructuredOutputMode,
    validation_retries: int = 1,
    validate: Callable[[StructuredModel], None] | None = None,
) -> StructuredModel:
    """Request provider-native structured output and validate it locally.

    ``json_object`` uses an OpenAI-compatible provider's JSON mode,
    ``tool_call`` forces a function call whose arguments carry the result, and
    ``prompted_json`` is for compatible gateways that expose neither feature.
    Every mode is validated against the same Pydantic model before returning.
    """
    if validation_retries < 0:
        raise ValueError("validation_retries must be non-negative")
    if mode not in {"json_object", "tool_call", "prompted_json"}:
        raise ValueError(f"Unsupported structured output mode: {mode}")

    schema_payload = schema.model_json_schema()
    schema_instruction = (
        "Return exactly one result matching the following JSON Schema. Do not "
        "add Markdown fences, commentary, or fields outside the schema.\n"
        + json.dumps(schema_payload, ensure_ascii=False, separators=(",", ":"))
    )
    request_messages = _with_schema_instruction(messages, schema_instruction)
    previous_output = ""
    previous_error = ""

    for attempt in range(validation_retries + 1):
        if attempt:
            request_messages = [
                *request_messages,
                {"role": "assistant", "content": previous_output[:8000]},
                {
                    "role": "user",
                    "content": (
                        "The previous result failed local schema validation. "
                        "Correct the result without changing its supported facts. "
                        f"Validation error: {previous_error[:2000]}"
                    ),
                },
            ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        elif mode == "tool_call":
            function = {
                "name": schema_name,
                "description": f"Return the validated {schema_name} result.",
                "parameters": schema_payload,
            }
            kwargs["tools"] = [{"type": "function", "function": function}]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": schema_name},
            }

        completion = await client.chat.completions.create(**kwargs)
        if not completion.choices:
            raise StructuredOutputError(
                f"{schema_name} model response contained no choices"
            )
        message = completion.choices[0].message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise StructuredOutputError(f"{schema_name} model refused: {refusal}")
        try:
            previous_output = _extract_payload(message, mode, schema_name)
            result = schema.model_validate_json(previous_output)
            if validate is not None:
                validate(result)
            return result
        except (StructuredOutputError, ValidationError, ValueError) as error:
            previous_error = str(error)
            if attempt == validation_retries:
                raise StructuredOutputError(
                    f"{schema_name} failed schema validation after "
                    f"{validation_retries + 1} attempt(s): {previous_error}"
                ) from error

    raise AssertionError("structured output retry loop did not return or raise")


def _with_schema_instruction(
    messages: list[dict[str, Any]], instruction: str
) -> list[dict[str, Any]]:
    output = [dict(message) for message in messages]
    if output and output[0].get("role") == "system":
        system_content = output[0].get("content")
        if not isinstance(system_content, str):
            raise ValueError("Structured output system content must be text")
        output[0]["content"] = f"{system_content.rstrip()}\n\n{instruction}"
    else:
        output.insert(0, {"role": "system", "content": instruction})
    return output


def _extract_payload(message: Any, mode: StructuredOutputMode, name: str) -> str:
    if mode == "tool_call":
        tool_calls = getattr(message, "tool_calls", None) or []
        matching = [
            call
            for call in tool_calls
            if getattr(getattr(call, "function", None), "name", None) == name
        ]
        if len(matching) != 1:
            raise StructuredOutputError(
                f"Expected one {name} tool call, received {len(matching)}"
            )
        arguments = getattr(matching[0].function, "arguments", None)
        if not isinstance(arguments, str) or not arguments.strip():
            raise StructuredOutputError(f"{name} tool call contained no arguments")
        return arguments.strip()

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise StructuredOutputError(f"{name} model response contained no JSON")
    return content.strip()
