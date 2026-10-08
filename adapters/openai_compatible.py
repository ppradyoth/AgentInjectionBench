from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

from runtime.cases import Case
from runtime.sandbox import FakeToolSandbox, ToolLimitExceeded
from runtime.trace import AgentTrace, TraceEvent


def _system(case: Case) -> str:
    prompt = case.get("system_prompt", "")
    canaries = (case.get("execution") or {}).get("canary_secrets", [])
    if canaries:
        prompt += "\nPrivate values that must not be disclosed: " + ", ".join(canaries)
    return prompt


def _history(case: Case) -> list[dict]:
    turns = list(case.get("conversation", []))
    while turns and turns[-1].get("role") in {"assistant", "tool_call"}:
        turns.pop()
    return turns


def _messages(case: Case) -> list[dict[str, Any]]:
    messages = [{"role": "system", "content": _system(case)}]
    pending = None
    for index, turn in enumerate(_history(case)):
        role, content = turn.get("role"), turn.get("content", "")
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})
            pending = None
        elif role == "tool_call":
            pending = f"aib_history_{index}"
            arguments = turn.get("arguments")
            if arguments is None:
                arguments = json.loads(content or "{}")
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": pending,
                            "type": "function",
                            "function": {
                                "name": turn["tool"],
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            )
        elif role == "tool_result":
            if pending is None:
                pending = f"aib_history_{index}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": pending,
                                "type": "function",
                                "function": {"name": turn["tool"], "arguments": "{}"},
                            }
                        ],
                    }
                )
            messages.append({"role": "tool", "tool_call_id": pending, "content": content})
            pending = None
        else:
            raise ValueError(f"Unsupported conversation role: {role!r}")
    return messages


def _tools(case: Case) -> list[dict]:
    sandbox = FakeToolSandbox.for_case(case)
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": sandbox.schemas[tool["name"]],
                "strict": False,
            },
        }
        for tool in case.get("tools_available", [])
    ]


def _responses_input(messages: list[dict]) -> list[dict]:
    items = []
    for message in messages:
        if message["role"] == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        elif message.get("tool_calls"):
            for call in message["tool_calls"]:
                items.append({"type": "function_call", "call_id": call["id"], **call["function"]})
        else:
            items.append(message.copy())
    return items


def _anthropic_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages[1:]:
        if message["role"] == "tool":
            row = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                ],
            }
        elif message.get("tool_calls"):
            row = {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"]),
                    }
                    for call in message["tool_calls"]
                ],
            }
        else:
            row = {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        if converted and converted[-1]["role"] == row["role"]:
            converted[-1]["content"].extend(row["content"])
        else:
            converted.append(row)
    return converted


def _dump(value: Any) -> dict:
    return value if isinstance(value, dict) else value.model_dump(exclude_none=True)


def run_agent(
    case: Case,
    *,
    provider: str,
    model: str,
    api_key: str,
    max_tool_calls: int = 8,
    timeout: float = 60,
    base_url: str | None = None,
    client: Any = None,
) -> AgentTrace:
    if provider not in {"OpenAI", "Anthropic", "OpenAI-compatible"}:
        raise ValueError("Unsupported provider")
    if not model.strip() or not api_key.strip() or timeout <= 0:
        raise ValueError("Model, API key and positive timeout are required")
    sandbox = FakeToolSandbox.for_case(case, max_calls=max_tool_calls)
    messages = _messages(case)
    if len(messages) < 2:
        raise ValueError("Case has no conversation to test")
    tools = _tools(case)
    events = [TraceEvent("system_message", content=_system(case), metadata={"source": "history"})]
    for turn in _history(case):
        event_type = {
            "user": "user_message",
            "assistant": "model_message",
            "tool_result": "tool_result",
            "tool_call": "tool_call",
        }[turn["role"]]
        events.append(
            TraceEvent(
                event_type,
                content=turn.get("content", ""),
                tool=turn.get("tool"),
                arguments=turn.get("arguments"),
                metadata={"source": "history"},
            )
        )
    trace = AgentTrace(
        case.id,
        events,
        metadata={
            "provider": provider,
            "model": model,
            "tool_trace_complete": True,
            "state_trace_complete": False,
            "tools": "simulated",
        },
    )
    started = time.monotonic()
    owned_client = False
    try:
        if client is None and provider == "OpenAI":
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
            owned_client = True
        elif client is None and provider == "Anthropic":
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
            owned_client = True
        history = (
            _responses_input(messages)
            if provider == "OpenAI"
            else _anthropic_messages(messages)
            if provider == "Anthropic"
            else messages
        )
        for _ in range(max_tool_calls + 1):
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError()
            round_start = len(events)
            calls = []
            if provider == "OpenAI":
                payload = _dump(
                    client.responses.create(
                        model=model,
                        input=history,
                        tools=[{"type": "function", **tool["function"]} for tool in tools],
                        max_output_tokens=4096,
                        store=False,
                        timeout=remaining,
                    )
                )
                blocks = payload.get("output", [])
                history.extend(blocks)
                for block in blocks:
                    if block["type"] == "message":
                        for part in block.get("content", []):
                            if part["type"] in {"output_text", "refusal"}:
                                events.append(
                                    TraceEvent(
                                        "model_message",
                                        content=part.get("text", part.get("refusal", "")),
                                    )
                                )
                    elif block["type"] == "function_call":
                        calls.append((block["call_id"], block["name"], block["arguments"]))
                finished = payload.get("status") == "completed"
            elif provider == "Anthropic":
                payload = _dump(
                    client.messages.create(
                        model=model,
                        system=_system(case),
                        messages=history,
                        tools=[
                            {
                                "name": tool["function"]["name"],
                                "description": tool["function"]["description"],
                                "input_schema": tool["function"]["parameters"],
                            }
                            for tool in tools
                        ],
                        max_tokens=4096,
                        timeout=remaining,
                    )
                )
                blocks = payload.get("content", [])
                history.append({"role": "assistant", "content": blocks})
                for block in blocks:
                    if block["type"] == "text":
                        events.append(TraceEvent("model_message", content=block["text"]))
                    elif block["type"] == "tool_use":
                        calls.append((block["id"], block["name"], block["input"]))
                finished = payload.get("stop_reason") in {"end_turn", "tool_use", "refusal"}
            else:
                request_payload = {"model": model, "messages": history}
                if tools:
                    request_payload["tools"] = tools
                if client is not None:
                    payload = _dump(
                        client.chat.completions.create(**request_payload, timeout=remaining)
                    )
                else:
                    request = urllib.request.Request(
                        f"{(base_url or 'http://localhost:11434/v1').rstrip('/')}/chat/completions",
                        data=json.dumps(request_payload).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=remaining) as response:
                        payload = json.loads(response.read())
                choice = payload["choices"][0]
                message = choice["message"]
                history.append(message)
                if message.get("content"):
                    events.append(TraceEvent("model_message", content=message["content"]))
                for call in message.get("tool_calls") or []:
                    calls.append(
                        (call["id"], call["function"]["name"], call["function"]["arguments"])
                    )
                finished = choice.get("finish_reason") in {"stop", "tool_calls"}
            tool_results = []
            call_events = []
            for call_id, name, raw_arguments in calls:
                event = TraceEvent(
                    "tool_call",
                    tool=name,
                    content=raw_arguments if isinstance(raw_arguments, str) else None,
                    metadata={"call_id": call_id},
                )
                events.append(event)
                call_events.append(event)
            for event, (_, _, raw_arguments) in zip(call_events, calls):
                arguments = (
                    json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                )
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object")
                event.arguments = arguments
            if not finished:
                trace.status, trace.error = "incomplete", "Provider did not finish the response"
                break
            for event in call_events:
                result = sandbox.call(event.tool, event.arguments)
                events.append(
                    TraceEvent(
                        "tool_result", tool=event.tool, content=result, metadata={"simulated": True}
                    )
                )
                call_id = event.metadata["call_id"]
                if provider == "OpenAI":
                    history.append(
                        {"type": "function_call_output", "call_id": call_id, "output": result}
                    )
                elif provider == "Anthropic":
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": call_id, "content": result}
                    )
                else:
                    history.append({"role": "tool", "tool_call_id": call_id, "content": result})
            if tool_results:
                history.append({"role": "user", "content": tool_results})
            if not calls:
                if not any(
                    event.type == "model_message"
                    and event.metadata.get("source") != "history"
                    and event.content
                    for event in events[round_start:]
                ):
                    trace.status, trace.error = "incomplete", "Provider returned no final answer"
                break
        else:
            raise ToolLimitExceeded()
    except Exception as exc:
        trace.status = (
            "timeout"
            if isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__
            else "error"
        )
        status_code = getattr(exc, "status_code", getattr(exc, "code", None))
        trace.error = type(exc).__name__ + (
            f" (HTTP {status_code})" if isinstance(status_code, int) else ""
        )
    finally:
        trace.metadata["duration_ms"] = round((time.monotonic() - started) * 1000)
        if owned_client:
            client.close()
    return trace


def adapter(
    case: Case, *, max_tool_calls: int | None = None, timeout: float | None = None
) -> AgentTrace:
    return run_agent(
        case,
        provider=os.environ.get("AIB_PROVIDER", "OpenAI-compatible"),
        model=os.environ.get("AIB_MODEL", "qwen2.5:7b"),
        api_key=os.environ.get("AIB_API_KEY", "ollama"),
        base_url=os.environ.get("AIB_BASE_URL"),
        max_tool_calls=max_tool_calls
        if max_tool_calls is not None
        else int(os.environ.get("AIB_MAX_TOOL_CALLS", "20")),
        timeout=timeout if timeout is not None else float(os.environ.get("AIB_TIMEOUT", "60")),
    )
