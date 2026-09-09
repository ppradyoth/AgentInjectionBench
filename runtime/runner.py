from __future__ import annotations

import random
import signal
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from runtime.cases import Case
from runtime.trace import AgentTrace, normalize_trace

Adapter = Callable[[Case], AgentTrace | dict[str, Any]]


@dataclass
class RunResult:
    traces: list[AgentTrace]
    errors: list[dict[str, str]]


class CaseTimeout(TimeoutError):
    pass


def _timeout_handler(_signum, _frame):
    raise CaseTimeout("Case execution timed out")


def _run_adapter(adapter: Adapter, case: Case, timeout: float | None):
    if timeout is None:
        return adapter(case)
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return adapter(case)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_cases(
    cases: Iterable[Case],
    adapter: Adapter,
    *,
    limit: int | None = None,
    seed: int = 42,
    max_tool_calls: int = 20,
    timeout: float | None = None,
) -> RunResult:
    selected = list(cases)
    random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]

    traces: list[AgentTrace] = []
    errors: list[dict[str, str]] = []
    for case in selected:
        try:
            trace = normalize_trace(_run_adapter(adapter, case, timeout), case.id)
            tool_call_count = sum(event.type == "tool_call" for event in trace.events)
            if tool_call_count > max_tool_calls:
                raise ValueError(
                    f"Tool-call limit exceeded ({tool_call_count} > {max_tool_calls})"
                )
            traces.append(trace)
        except CaseTimeout as exc:
            errors.append({"case_id": case.id, "error": str(exc), "status": "timeout"})
        except Exception as exc:
            errors.append({"case_id": case.id, "error": str(exc)})
    return RunResult(traces=traces, errors=errors)
