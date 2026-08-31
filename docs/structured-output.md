# Structured model output

DirectorX treats model responses that control the workflow as typed agent state, not as prose to be interpreted by downstream agents. This follows the common design used by mature open-source agent frameworks:

- LangGraph exposes `with_structured_output` for typed routing and evaluation, while tool calls and graph state remain machine-readable.
- AutoGen's `AssistantAgent` supports a Pydantic `output_content_type` and emits a `StructuredMessage`; its model clients declare whether structured output is supported.
- CrewAI tasks support Pydantic/JSON outputs and guardrails, with conversion or re-request after validation failure.
- MetaGPT separates human-readable `Message.content` from typed Pydantic `instruct_content`.

The practical lesson is capability-driven transport with one local contract. DirectorX therefore supports three transports through `structured_output_mode`:

- `json_object`: use the provider's native JSON mode. This is the default for the SiliconFlow VLM path.
- `tool_call`: force one function call and validate its arguments. Use this only when the selected model reliably supports function calling.
- `prompted_json`: request JSON without provider-specific response-format parameters. This is the default for compatible gateways whose native structured-output capability is unknown.

All three modes serialize the same Pydantic JSON Schema into the request and validate the returned payload locally. JSON syntax, schema constraints, and stage-specific rules are part of one validation loop. An invalid result is sent back once with the validation error so the model can correct the payload without changing supported facts. If the corrected result is still invalid, the owning agent blocks and records the error; it does not silently reinterpret tagged prose or downgrade the edit.

Natural language remains natural language where it belongs. Dense visual captions, screenplay prose, narration, summaries, and rationales are string fields inside typed records. Agent identifiers, source ranges, evidence frame IDs, confidence values, and workflow decisions are structured fields. This boundary keeps creative quality independent from reliable orchestration.

References:

- [LangGraph workflows and agents](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/workflows-agents.mdx)
- [AutoGen AssistantAgent](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py)
- [CrewAI LiteAgent](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/lite_agent.py)
- [MetaGPT Action](https://github.com/FoundationAgents/MetaGPT/blob/main/metagpt/actions/action.py)
