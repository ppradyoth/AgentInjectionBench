from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    id: str
    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Case":
        if not isinstance(payload.get("id"), str) or not payload["id"]:
            raise ValueError("Every benchmark case needs a non-empty string id")
        return cls(id=payload["id"], payload=payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            case = Case.from_dict(payload)
            if case.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases
