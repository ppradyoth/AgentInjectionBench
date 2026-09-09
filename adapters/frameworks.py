from __future__ import annotations

from collections.abc import Callable
from typing import Any

from runtime.cases import Case
from runtime.trace import AgentTrace, TraceEvent


def _prompt(case: Case) -> str:
    parts = [case.get("system_prompt", "")]
    for turn in case.get("conversation", []):
        if turn.get("role") == "user":
            parts.append(f"User: {turn.get('content', '')}")
        elif turn.get("role") == "tool_result":
            parts.append(f"Tool result ({turn.get('tool', 'unknown')}): {turn.get('content', '')}")
    return "\n\n".join(parts)


def _result_trace(case: Case, result: Any) -> AgentTrace:
    if isinstance(result, AgentTrace):
        return result
    content = getattr(result, "content", None) or getattr(result, "raw", None) or str(result)
    return AgentTrace(case_id=case.id, events=[TraceEvent(type="model_message", content=content)])


def make_langgraph_adapter(graph: Any) -> Callable[[Case], AgentTrace]:
    def adapter(case: Case) -> AgentTrace:
        state = graph.invoke({"messages": [{"role": "user", "content": _prompt(case)}]})
        messages = state.get("messages", []) if isinstance(state, dict) else []
        events = [
            TraceEvent(
                type="tool_call" if getattr(message, "tool_calls", None) else "model_message",
                content=getattr(message, "content", None) or "",
                metadata={"tool_calls": getattr(message, "tool_calls", [])},
            )
            for message in messages
        ]
        return AgentTrace(case_id=case.id, events=events)

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
