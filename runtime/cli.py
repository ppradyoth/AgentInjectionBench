from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.graders import grade_trace
from evaluation.score import DATA_FILE, dataset_fingerprint
from runtime.cases import load_cases
from runtime.runner import run_cases


def _load_adapter(import_path: str):
    module_name, separator, attribute = import_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Adapter must use the form module:callable")
    module = importlib.import_module(module_name)
    adapter = getattr(module, attribute)
    if not callable(adapter):
        raise TypeError(f"Adapter {import_path!r} is not callable")
    return adapter


def _manifest(
    args: argparse.Namespace,
    data_path: Path,
    dataset_hash: str,
    count: int,
) -> dict[str, Any]:
    return {
        "runner": "agent-injection-bench",
        "runner_version": "0.2.0-dev",
        "adapter": args.adapter,
        "dataset": str(data_path),
        "dataset_sha256": dataset_hash,
        "seed": args.seed,
        "cases_requested": args.limit,
        "cases_run": count,
        "offline": args.offline,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an AgentInjectionBench adapter")
    parser.add_argument("--adapter", required=True, help="Adapter import path: module:callable")
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.data.exists():
        parser.error(f"Dataset not found: {args.data}")

    try:
        adapter = _load_adapter(args.adapter)
        cases = load_cases(args.data)
        result = run_cases(cases, adapter, limit=args.limit, seed=args.seed)
    except (ImportError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    manifest = _manifest(
        args,
        args.data,
        dataset_fingerprint([case.payload for case in cases]),
        len(result.traces),
    )
    cases_by_id = {case.id: case for case in cases}
    outcomes = [grade_trace(cases_by_id[trace.case_id], trace).to_dict() for trace in result.traces]
    payload = {
        "manifest": manifest,
        "traces": [trace.to_dict() for trace in result.traces],
        "outcomes": outcomes,
        "errors": result.errors,
    }

    if args.bundle:
        args.bundle.mkdir(parents=True, exist_ok=True)
        (args.bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.bundle / "traces.jsonl").write_text(
            "".join(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n" for trace in result.traces)
        )
        (args.bundle / "outcomes.jsonl").write_text(
            "".join(json.dumps(outcome, ensure_ascii=False) + "\n" for outcome in outcomes)
        )
        (args.bundle / "errors.json").write_text(json.dumps(result.errors, indent=2) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Cases run: {len(result.traces)}")
        print(f"Adapter errors: {len(result.errors)}")
        print(f"Compromised: {sum(outcome['outcome'] == 'compromised' for outcome in outcomes)}")
        print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
        if args.bundle:
            print(f"Bundle: {args.bundle}")
    return 0 if not result.errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
