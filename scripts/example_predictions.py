#!/usr/bin/env python3
"""Create a prediction file from a user-supplied agent adapter.

Replace ``predict`` with a call to the agent or guardrail under test. The output
format stays stable across LangGraph, CrewAI, LlamaIndex, MCP, and custom agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.detectors import DETECTORS
from evaluation.score import DATA_FILE, load_dataset


def predict(sample: dict) -> str:
    """Replace this baseline call with the agent or guardrail under test."""
    return DETECTORS["control_channel_scanner"](sample)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for sample in load_dataset(args.data):
            output.write(json.dumps({"id": sample["id"], "prediction": predict(sample)}) + "\n")


if __name__ == "__main__":
    main()
