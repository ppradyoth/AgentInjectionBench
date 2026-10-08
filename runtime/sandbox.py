from __future__ import annotations

import json
import re

from jsonschema import Draft202012Validator, ValidationError
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
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def for_case(cls, case: Case, max_calls: int = 20) -> "FakeToolSandbox":
        if max_calls < 1:
            raise ValueError("Tool-call limit must be positive")
        definitions = case.get("tools_available", [])
        if not isinstance(definitions, list):
            raise ValueError("Tools must be a JSON array")
        schemas = {}
        for tool in definitions:
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("name"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool["name"])
            ):
                raise ValueError("Each tool requires a valid name")
            if tool["name"] in schemas:
                raise ValueError("Tool names must be unique")
            schema = tool.get(
                "parameters",
                tool.get(
                    "inputSchema",
                    tool.get(
                        "input_schema",
                        {"type": "object", "properties": {}, "additionalProperties": True},
                    ),
                ),
            )
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("Tool parameters must be an object JSON Schema")
            pending = [schema]
            while pending:
                node = pending.pop()
                if isinstance(node, dict):
                    for key in ("$ref", "$dynamicRef"):
                        if key in node and (
                            not isinstance(node[key], str) or not node[key].startswith("#")
                        ):
                            raise ValueError("Tool schemas may only use local references")
                    pending.extend(node.values())
                elif isinstance(node, list):
                    pending.extend(node)
            Draft202012Validator.check_schema(schema)
            schemas[tool["name"]] = schema
        return cls(tools=set(schemas), max_calls=max_calls, schemas=schemas)

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        if name not in self.tools:
            raise UnknownTool(f"Tool {name!r} is not available in this case")
        if len(self.calls) >= self.max_calls:
            raise ToolLimitExceeded(f"Tool-call limit exceeded ({self.max_calls})")
        normalized_arguments = {} if arguments is None else arguments
        if not isinstance(normalized_arguments, dict):
            raise ValueError("Tool arguments must be an object")
        try:
            Draft202012Validator(self.schemas.get(name, {})).validate(normalized_arguments)
        except ValidationError as exc:
            raise ValueError("Tool arguments do not match the schema") from exc
        self.calls.append({"tool": name, "arguments": normalized_arguments})
        return json.dumps({"tool": name, "status": "simulated", "arguments": normalized_arguments})
