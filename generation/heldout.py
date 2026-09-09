from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _bucket(sample_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def make_heldout(input_path: Path, public_path: Path, heldout_path: Path, ratio: float, seed: int) -> dict:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    heldout = [row for row in rows if _bucket(row["id"], seed) < ratio]
    heldout_ids = {row["id"] for row in heldout}
    public = [row for row in rows if row["id"] not in heldout_ids]
    public_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in ((public_path, public), (heldout_path, heldout)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    manifest = {
        "seed": seed,
        "ratio": ratio,
        "total": len(rows),
        "public": len(public),
        "heldout": len(heldout),
        "heldout_ids": sorted(heldout_ids),
    }
    heldout_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a maintainer-only held-out split")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be between 0 and 1")
    print(json.dumps(make_heldout(args.input, args.public, args.heldout, args.ratio, args.seed), indent=2))


if __name__ == "__main__":
    main()
