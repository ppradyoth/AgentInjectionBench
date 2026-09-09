from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EVENT_TYPES = {
    "user_message",
    "system_message",
    "tool_call",
    "tool_result",
    "model_message",
    "state_write",
    "error",
}


@dataclass
class TraceEvent:
    type: str
    content: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"Unknown trace event type: {self.type!r}")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("Trace event content must be a string or None")
        if self.arguments is not None and not isinstance(self.arguments, dict):
            raise TypeError("Trace event arguments must be a dictionary or None")


@dataclass
class AgentTrace:
    case_id: str
    events: list[TraceEvent] = field(default_factory=list)
    status: str = "completed"
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("AgentTrace requires a case_id")
        if self.status not in {"completed", "error", "timeout"}:
            raise ValueError(f"Unknown trace status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "events": [asdict(event) for event in self.events],
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], case_id: str | None = None) -> "AgentTrace":
        resolved_case_id = payload.get("case_id", case_id)
        if not resolved_case_id:
            raise ValueError("Trace output requires case_id")
        events = [TraceEvent(**event) for event in payload.get("events", [])]
        return cls(
            case_id=resolved_case_id,
            events=events,
            status=payload.get("status", "completed"),
            error=payload.get("error"),
        )


def normalize_trace(result: AgentTrace | dict[str, Any], case_id: str) -> AgentTrace:
    if isinstance(result, AgentTrace):
        if result.case_id != case_id:
            raise ValueError(f"Adapter returned trace for {result.case_id!r}, expected {case_id!r}")
        return result
    if isinstance(result, dict):
        return AgentTrace.from_dict(result, case_id=case_id)
    raise TypeError("Adapter must return AgentTrace or a trace dictionary")
