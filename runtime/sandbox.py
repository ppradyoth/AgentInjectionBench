from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from runtime.cases import Case


class ToolLimitExceeded(RuntimeError):
    pass


class UnknownTool(RuntimeError):
    pass


@dataclass
class FakeToolSandbox:
    tools: set[str]
    max_calls: int = 20
    calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def for_case(cls, case: Case, max_calls: int = 20) -> "FakeToolSandbox":
        available = {
            tool.get("name")
            for tool in case.get("tools_available", [])
            if isinstance(tool, dict) and tool.get("name")
        }
        return cls(tools=available, max_calls=max_calls)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        if name not in self.tools:
            raise UnknownTool(f"Tool {name!r} is not available in this case")
        if len(self.calls) >= self.max_calls:
            raise ToolLimitExceeded(f"Tool-call limit exceeded ({self.max_calls})")
        normalized_arguments = arguments or {}
        self.calls.append({"tool": name, "arguments": normalized_arguments})
        return json.dumps({"tool": name, "status": "ok", "arguments": normalized_arguments})
