from __future__ import annotations

from pathlib import Path

from evaluation.leaderboard import main as leaderboard_main


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    specs = []
    for path in sorted((ROOT / "submissions").glob("*.jsonl")):
        specs.extend(["--predictions", f"{path.stem}={path}"])
    return leaderboard_main(["--baselines", *specs, "-o", str(ROOT / "LEADERBOARD.md")])


if __name__ == "__main__":
    raise SystemExit(main())
