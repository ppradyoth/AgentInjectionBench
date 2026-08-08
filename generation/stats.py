#!/usr/bin/env python3
"""Dataset statistics and distribution analysis for AgentInjectionBench."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def cross_tab(samples: list[dict], key_a: str, key_b: str) -> dict[str, Counter]:
    table = defaultdict(Counter)
    for s in samples:
        a = s.get(key_a, "unknown")
        b = s.get(key_b, "unknown")
        table[a][b] += 1
    return table


def print_cross_tab(table: dict[str, Counter], key_a: str, key_b: str):
    all_b = sorted({b for counts in table.values() for b in counts})
    header = f"{'':30s}" + "".join(f"{b:>15s}" for b in all_b) + f"{'TOTAL':>10s}"
    print(header)
    print("-" * len(header))
    for a in sorted(table):
        row = f"{a:30s}"
        total = 0
        for b in all_b:
            count = table[a][b]
            row += f"{count:>15d}"
            total += count
        row += f"{total:>10d}"
        print(row)


def print_full_stats(samples: list[dict]):
    print(f"{'=' * 60}")
    print("AgentInjectionBench Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(samples)}")
    print()

    fields = [
        ("attack_category", "Attack Category"),
        ("attacker_intent", "Attacker Intent"),
        ("injection_surface", "Injection Surface"),
        ("complexity", "Complexity"),
        ("severity", "Severity"),
        ("defense_bypass", "Defense Bypass"),
        ("ground_truth", "Ground Truth"),
    ]

    for field, label in fields:
        counts = Counter(s.get(field, "unknown") for s in samples)
        print(f"\n{label}:")
        for val, count in counts.most_common():
            pct = 100 * count / len(samples)
            bar = "█" * int(pct / 2)
            print(f"  {val:30s} {count:5d} ({pct:5.1f}%) {bar}")

    all_tools = []
    for s in samples:
        all_tools.extend(s.get("target_tools", []))
    tool_counts = Counter(all_tools)
    print(f"\nTarget Tools (total mentions: {len(all_tools)}):")
    for tool, count in tool_counts.most_common():
        print(f"  {tool:30s} {count:5d}")

    print(f"\n{'=' * 60}")
    print("Cross-tabulation: Category × Intent")
    print(f"{'=' * 60}")
    print_cross_tab(cross_tab(samples, "attack_category", "attacker_intent"), "category", "intent")

    print(f"\n{'=' * 60}")
    print("Cross-tabulation: Category × Severity")
    print(f"{'=' * 60}")
    print_cross_tab(cross_tab(samples, "attack_category", "severity"), "category", "severity")

    print(f"\n{'=' * 60}")
    print("Cross-tabulation: Category × Defense Bypass")
    print(f"{'=' * 60}")
    print_cross_tab(cross_tab(samples, "attack_category", "defense_bypass"), "category", "bypass")


def export_stats_json(samples: list[dict], output_path: Path):
    stats = {
        "total_samples": len(samples),
        "by_category": dict(Counter(s["attack_category"] for s in samples)),
        "by_intent": dict(Counter(s["attacker_intent"] for s in samples)),
        "by_surface": dict(Counter(s["injection_surface"] for s in samples)),
        "by_complexity": dict(Counter(s["complexity"] for s in samples)),
        "by_severity": dict(Counter(s["severity"] for s in samples)),
        "by_bypass": dict(Counter(s["defense_bypass"] for s in samples)),
    }
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AgentInjectionBench dataset statistics")
    parser.add_argument("--input", default=str(DATA_DIR / "agent_injection_bench.jsonl"))
    parser.add_argument("--export-json", default=None, help="Export stats as JSON")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        return

    samples = load_jsonl(path)
    print_full_stats(samples)

    if args.export_json:
        export_stats_json(samples, Path(args.export_json))


if __name__ == "__main__":
    main()
