#!/usr/bin/env python3
"""Curation pipeline: dedup, quality filter, balance, and split the dataset."""

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def dataset_hash(samples: list[dict]) -> str:
    canonical = "\n".join(
        json.dumps(sample, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for sample in samples
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_split_manifest(splits: dict[str, list[dict]], path: Path, seed: int) -> None:
    manifest = {
        "seed": seed,
        "splits": {
            name: {"count": len(items), "sha256": dataset_hash(items)}
            for name, items in splits.items()
        },
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def content_hash(sample: dict) -> str:
    conv_str = json.dumps(sample.get("conversation", []), sort_keys=True)
    return hashlib.sha256(conv_str.encode()).hexdigest()[:16]


def deduplicate(samples: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for s in samples:
        h = content_hash(s)
        if h not in seen:
            seen.add(h)
            unique.append(s)
    removed = len(samples) - len(unique)
    if removed:
        log.info(f"Dedup: removed {removed} duplicates ({len(unique)} remaining)")
    return unique


def quality_filter(samples: list[dict], taxonomy: dict) -> list[dict]:
    valid_categories = set(taxonomy["attack_categories"].keys())
    valid_intents = set(taxonomy["attacker_intents"].keys())
    valid_surfaces = set(taxonomy["injection_surfaces"].keys())
    valid_complexity = set(taxonomy["complexity_levels"].keys())
    valid_bypasses = set(taxonomy["defense_bypass_techniques"].keys())
    valid_severity = set(taxonomy["severity_levels"].keys())
    valid_tools = set(taxonomy["target_tool_types"])

    passed = []
    rejected = 0

    for s in samples:
        if s.get("attack_category") not in valid_categories:
            rejected += 1
            continue
        if s.get("attacker_intent") not in valid_intents:
            rejected += 1
            continue
        if s.get("injection_surface") not in valid_surfaces:
            rejected += 1
            continue
        if s.get("complexity") not in valid_complexity:
            rejected += 1
            continue
        if s.get("defense_bypass") not in valid_bypasses:
            rejected += 1
            continue
        if s.get("severity") not in valid_severity:
            rejected += 1
            continue

        conv = s.get("conversation", [])
        if not conv or not isinstance(conv, list):
            rejected += 1
            continue

        if not s.get("system_prompt"):
            rejected += 1
            continue

        tools = s.get("target_tools", [])
        if tools and not all(t in valid_tools for t in tools):
            rejected += 1
            continue

        passed.append(s)

    if rejected:
        log.info(f"Quality filter: rejected {rejected} ({len(passed)} remaining)")
    return passed


def balance_dataset(samples: list[dict], max_per_category: int | None = None) -> list[dict]:
    by_category = defaultdict(list)
    for s in samples:
        by_category[s["attack_category"]].append(s)

    log.info("Category distribution:")
    for cat, items in sorted(by_category.items()):
        log.info(f"  {cat}: {len(items)}")

    if max_per_category is None:
        return samples

    balanced = []
    for cat, items in by_category.items():
        if len(items) > max_per_category:
            random.shuffle(items)
            balanced.extend(items[:max_per_category])
            log.info(f"  Capped {cat}: {len(items)} → {max_per_category}")
        else:
            balanced.extend(items)

    return balanced


def reassign_ids(samples: list[dict]) -> list[dict]:
    for i, s in enumerate(samples, 1):
        s["id"] = f"AIB-{i:05d}"
    return samples


def split_dataset(
    samples: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_category = defaultdict(list)
    for s in samples:
        by_category[s["attack_category"]].append(s)

    train, val, test = [], [], []

    for cat, items in by_category.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    log.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def print_stats(samples: list[dict]):
    print(f"\nTotal samples: {len(samples)}")

    print("\nBy category:")
    cat_counts = Counter(s["attack_category"] for s in samples)
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    print("\nBy intent:")
    intent_counts = Counter(s["attacker_intent"] for s in samples)
    for intent, count in intent_counts.most_common():
        print(f"  {intent}: {count}")

    print("\nBy surface:")
    surface_counts = Counter(s["injection_surface"] for s in samples)
    for surface, count in surface_counts.most_common():
        print(f"  {surface}: {count}")

    print("\nBy complexity:")
    comp_counts = Counter(s["complexity"] for s in samples)
    for comp, count in comp_counts.most_common():
        print(f"  {comp}: {count}")

    print("\nBy severity:")
    sev_counts = Counter(s["severity"] for s in samples)
    for sev, count in sev_counts.most_common():
        print(f"  {sev}: {count}")

    print("\nBy defense bypass:")
    bypass_counts = Counter(s["defense_bypass"] for s in samples)
    for bypass, count in bypass_counts.most_common():
        print(f"  {bypass}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Curate AgentInjectionBench dataset")
    parser.add_argument("--input", default=str(DATA_DIR / "agent_injection_bench_raw.jsonl"))
    parser.add_argument("--output", default=str(DATA_DIR / "agent_injection_bench.jsonl"))
    parser.add_argument("--max-per-category", type=int, default=None)
    parser.add_argument("--split", action="store_true", help="Generate train/val/test splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    taxonomy = load_taxonomy()
    samples = load_jsonl(Path(args.input))
    log.info(f"Loaded {len(samples)} raw samples")

    if args.stats_only:
        print_stats(samples)
        return

    samples = deduplicate(samples)
    samples = quality_filter(samples, taxonomy)
    samples = balance_dataset(samples, args.max_per_category)
    samples = reassign_ids(samples)

    save_jsonl(samples, Path(args.output))
    log.info(f"Saved {len(samples)} curated samples → {args.output}")

    if args.split:
        train, val, test = split_dataset(samples)
        splits_dir = DATA_DIR / "splits"
        save_jsonl(train, splits_dir / "train.jsonl")
        save_jsonl(val, splits_dir / "validation.jsonl")
        save_jsonl(test, splits_dir / "test.jsonl")
        save_split_manifest(
            {"train": train, "validation": val, "test": test},
            splits_dir / "manifest.json",
            args.seed,
        )
        log.info(f"Splits saved to {splits_dir}/")

    print_stats(samples)


if __name__ == "__main__":
    main()
