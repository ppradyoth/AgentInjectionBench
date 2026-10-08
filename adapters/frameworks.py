from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from runtime.cases import Case
from adapters.openai_compatible import _messages
from runtime.trace import AgentTrace, TraceEvent, normalize_trace


def _prompt(case: Case) -> str:
    parts = [case.get("system_prompt", "")]
    for turn in case.get("conversation", []):
        if turn.get("role") == "user":
            parts.append(f"User: {turn.get('content', '')}")
        elif turn.get("role") == "tool_result":
            parts.append(f"Tool result ({turn.get('tool', 'unknown')}): {turn.get('content', '')}")
    return "\n\n".join(parts)


def _result_trace(case: Case, result: Any) -> AgentTrace:
    if isinstance(result, AgentTrace) or (isinstance(result, dict) and "events" in result):
        return normalize_trace(result, case.id)
    content = getattr(result, "content", None) or getattr(result, "raw", None) or str(result)
    return AgentTrace(
        case_id=case.id,
        events=[TraceEvent(type="model_message", content=str(content))],
        metadata={"tool_trace_complete": False, "state_trace_complete": False},
    )


def make_langgraph_adapter(graph: Any) -> Callable[[Case], AgentTrace]:
    def adapter(case: Case) -> AgentTrace:
        inputs = _messages(case)
        state = graph.invoke({"messages": inputs})
        messages = state.get("messages", []) if isinstance(state, dict) else []
        events = []
        history_answers = {
            message.get("content")
            for message in inputs
            if message["role"] == "assistant" and isinstance(message.get("content"), str)
        }
        history_calls = {call["id"] for message in inputs for call in message.get("tool_calls", [])}
        trace = AgentTrace(
            case.id, events, metadata={"tool_trace_complete": False, "state_trace_complete": False}
        )
        try:
            for message in messages:
                read = (
                    message.get
                    if isinstance(message, dict)
                    else lambda name, default=None: getattr(message, name, default)
                )
                role = read("role") or read("type")
                if role in {"human", "user", "system", "tool"}:
                    continue
                content = read("content", "")
                if content:
                    events.append(
                        TraceEvent(
                            "model_message",
                            content=str(content),
                            metadata={"source": "history"}
                            if isinstance(content, str) and content in history_answers
                            else {},
                        )
                    )
                for call in read("tool_calls", []) or []:
                    function = call.get("function", call)
                    arguments = function.get("args", function.get("arguments", {}))
                    event = TraceEvent(
                        "tool_call",
                        tool=function.get("name"),
                        content=arguments if isinstance(arguments, str) else None,
                        metadata={"source": "history"} if call.get("id") in history_calls else {},
                    )
                    events.append(event)
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object")
                    event.arguments = arguments
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            trace.status, trace.error = "error", type(exc).__name__
        return trace

    return adapter


def make_crewai_adapter(crew: Any) -> Callable[[Case], AgentTrace]:
    def adapter(case: Case) -> AgentTrace:
        return _result_trace(case, crew.kickoff(inputs={"task": _prompt(case)}))

    return adapter


def make_llamaindex_adapter(agent: Any) -> Callable[[Case], AgentTrace]:
    def adapter(case: Case) -> AgentTrace:
        return _result_trace(case, agent.chat(_prompt(case)))

    return adapter


def make_mcp_adapter(agent: Callable[[Case], Any]) -> Callable[[Case], AgentTrace]:
    def adapter(case: Case) -> AgentTrace:
        return _result_trace(case, agent(case))

    return adapter
