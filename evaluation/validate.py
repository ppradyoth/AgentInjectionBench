from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.score import DATA_FILE, load_dataset, load_predictions
from generation.validate_schema import validate_file


def validate_predictions(data_path: Path, predictions_path: Path, allow_missing: bool) -> list[str]:
    errors: list[str] = []
    samples = load_dataset(data_path)
    predictions = load_predictions(predictions_path)
    sample_ids = {sample["id"] for sample in samples}
    unknown = sorted(set(predictions) - sample_ids)
    missing = sorted(sample_ids - set(predictions))
    if unknown:
        errors.append(f"unknown prediction IDs: {', '.join(unknown[:10])}")
    if missing and not allow_missing:
        errors.append(f"missing predictions: {len(missing)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AgentInjectionBench data or submissions")
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--require-execution", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    total, error_count, validation_errors = validate_file(args.data)
    errors = error_count
    if validation_errors:
        print("\n".join(validation_errors[:20]), file=sys.stderr)
    if args.require_execution:
        samples = load_dataset(args.data)
        missing_execution = [sample["id"] for sample in samples if "execution" not in sample]
        if missing_execution:
            errors += len(missing_execution)
            errors_list = [f"missing execution metadata: {sample_id}" for sample_id in missing_execution]
            print("\n".join(errors_list[:20]), file=sys.stderr)
    if args.predictions:
        try:
            prediction_errors = validate_predictions(args.data, args.predictions, args.allow_missing)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            prediction_errors = [str(exc)]
        errors += len(prediction_errors)
        print("\n".join(prediction_errors[:20]), file=sys.stderr)

    print(f"{args.data}: {total} cases, {errors} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
