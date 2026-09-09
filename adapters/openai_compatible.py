from __future__ import annotations

import json
import os
import urllib.request

from runtime.cases import Case
from runtime.sandbox import FakeToolSandbox
from runtime.trace import AgentTrace, TraceEvent


def _messages(case: Case) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": case.get("system_prompt", "")}]
    for turn in case.get("conversation", []):
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append({"role": role, "content": content})
        elif role == "tool_result":
            messages.append({
                "role": "user",
                "content": f"[Tool result from {turn.get('tool', 'unknown')}]\n{content}",
            })
    return messages


def _tools(case: Case) -> list[dict]:
    tools = []
    for definition in case.get("tools_available", []):
        if not isinstance(definition, dict) or not definition.get("name"):
            continue
        parameters = definition.get("parameters") or definition.get("inputSchema")
        tools.append({
            "type": "function",
            "function": {
                "name": definition["name"],
                "description": definition.get("description", ""),
                "parameters": parameters or {"type": "object", "properties": {}},
            },
        })
    return tools


def adapter(case: Case) -> AgentTrace:
    base_url = os.environ.get("AIB_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    model = os.environ.get("AIB_MODEL", "qwen2.5:7b")
    api_key = os.environ.get("AIB_API_KEY", "ollama")
    messages = _messages(case)
    tools = _tools(case)
    sandbox = FakeToolSandbox.for_case(case, max_calls=int(os.environ.get("AIB_MAX_TOOL_CALLS", "20")))
    events = [
        TraceEvent(type="system_message", content=case.get("system_prompt", "")),
        *[
            TraceEvent(
                type="user_message" if turn.get("role") == "user" else "tool_result",
                content=turn.get("content", ""),
                tool=turn.get("tool"),
            )
            for turn in case.get("conversation", [])
            if turn.get("role") in {"user", "tool_result"}
        ],
    ]
    for _ in range(sandbox.max_calls + 1):
        request_payload = {"model": model, "messages": messages, "temperature": 0}
        if tools:
            request_payload["tools"] = tools
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(request_payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        events.append(TraceEvent(type="model_message", content=content))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            break
        messages.append(message)
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name", "")
            arguments = json.loads(function.get("arguments") or "{}")
            events.append(TraceEvent(type="tool_call", tool=name, arguments=arguments))
            tool_result = sandbox.call(name, arguments)
            events.append(TraceEvent(type="tool_result", tool=name, content=tool_result))
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id", name),
                "content": tool_result,
            })
    return AgentTrace(case_id=case.id, events=events)
