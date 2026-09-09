from __future__ import annotations

from runtime.cases import Case
from runtime.trace import AgentTrace, TraceEvent


def adapter(case: Case) -> AgentTrace:
    events = []
    for turn in case.get("conversation", []):
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            event_type = "user_message"
        elif role == "system":
            event_type = "system_message"
        elif role == "tool_result":
            event_type = "tool_result"
        elif role == "assistant":
            event_type = "model_message"
        elif role == "tool_call":
            event_type = "tool_call"
        else:
            continue
        events.append(
            TraceEvent(
                type=event_type,
                content=content,
                tool=turn.get("tool"),
                arguments=turn.get("arguments"),
            )
        )
    return AgentTrace(case_id=case.id, events=events)
