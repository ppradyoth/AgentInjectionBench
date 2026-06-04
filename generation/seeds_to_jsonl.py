#!/usr/bin/env python3
"""Convert seed YAML templates to agent_injection_bench.jsonl."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "generation" / "templates"
OUTPUT_PATH = REPO_ROOT / "data" / "agent_injection_bench.jsonl"

# Maps YAML field names to schema field names where they differ
SURFACE_DEFAULTS = {
    "tool_output_injection": "tool_output",
    "goal_hijacking": "rag_document",
    "privilege_escalation": "tool_output",
    "data_exfiltration": "api_response",
    "multi_turn_stateful": "tool_output",
    "mcp_context_poisoning": "mcp_response",
}


def seed_to_sample(seed: dict, category: str, counter: int) -> dict:
    # Build conversation from single-turn seeds
    conversation = seed.get("conversation", None)
    if conversation is None:
        conversation = [
            {"role": "user", "content": seed.get("user_message", "")},
            {
                "role": "tool_result",
                "tool": (seed.get("tools", [{}])[0] or {}).get("name", "tool"),
                "content": seed.get("injected_tool_output", ""),
            },
        ]
        if seed.get("expected_safe"):
            conversation.append(
                {"role": "assistant", "content": seed["expected_safe"]}
            )

    return {
        "id": f"AIB-{counter:05d}",
        "attack_category": category,
        "attacker_intent": seed.get("attacker_intent", "hijacking"),
        "injection_surface": seed.get(
            "injection_surface", SURFACE_DEFAULTS.get(category, "tool_output")
        ),
        "complexity": seed.get("complexity", "single_turn"),
        "target_tools": seed.get("target_tools", []),
        "defense_bypass": seed.get("defense_bypass", "none"),
        "system_prompt": seed.get("system_prompt", ""),
        "tools_available": seed.get("tools", []),
        "conversation": conversation,
        "ground_truth": "unsafe",
        "severity": seed.get("severity", "high"),
        "notes": seed.get("notes", seed.get("scenario", "")),
    }


def main():
    samples = []
    counter = 1

    for template_path in sorted(TEMPLATES_DIR.glob("*.yaml")):
        with open(template_path) as f:
            data = yaml.safe_load(f)

        category = data["category"]
        seeds = data["seeds"]

        for seed in seeds:
            sample = seed_to_sample(seed, category, counter)
            samples.append(sample)
            counter += 1

        print(f"  {category}: {len(seeds)} seeds")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(samples)} samples → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
