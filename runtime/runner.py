from __future__ import annotations

import random
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


def run_cases(
    cases: Iterable[Case],
    adapter: Adapter,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> RunResult:
    selected = list(cases)
    random.Random(seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]

    traces: list[AgentTrace] = []
    errors: list[dict[str, str]] = []
    for case in selected:
        try:
            traces.append(normalize_trace(adapter(case), case.id))
        except Exception as exc:
            errors.append({"case_id": case.id, "error": str(exc)})
    return RunResult(traces=traces, errors=errors)
