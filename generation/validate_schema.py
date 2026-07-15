#!/usr/bin/env python3
"""Validate AgentInjectionBench JSONL files against the dataset schema."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.json"

SCHEMA = {
    "required_fields": {
        "id": str,
        "attack_category": str,
        "attacker_intent": str,
        "injection_surface": str,
        "complexity": str,
        "target_tools": list,
        "defense_bypass": str,
        "system_prompt": str,
        "tools_available": list,
        "conversation": list,
        "ground_truth": str,
        "severity": str,
    },
    "optional_fields": {
        "notes": str,
        "metadata": dict,
    },
}

CONVERSATION_ROLES = {
    "user",
    "assistant",
    "tool_call",
    "tool_result",
    "system",
}


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def validate_sample(sample: dict, taxonomy: dict, line_num: int) -> list[str]:
    errors = []
    prefix = f"Line {line_num} (id={sample.get('id', '???')})"

    for field, expected_type in SCHEMA["required_fields"].items():
        if field not in sample:
            errors.append(f"{prefix}: missing required field '{field}'")
        elif not isinstance(sample[field], expected_type):
            errors.append(
                f"{prefix}: field '{field}' expected {expected_type.__name__}, "
                f"got {type(sample[field]).__name__}"
            )

    if sample.get("attack_category") not in taxonomy["attack_categories"]:
        errors.append(f"{prefix}: invalid attack_category '{sample.get('attack_category')}'")

    if sample.get("attacker_intent") not in taxonomy["attacker_intents"]:
        errors.append(f"{prefix}: invalid attacker_intent '{sample.get('attacker_intent')}'")

    if sample.get("injection_surface") not in taxonomy["injection_surfaces"]:
        errors.append(f"{prefix}: invalid injection_surface '{sample.get('injection_surface')}'")

    if sample.get("complexity") not in taxonomy["complexity_levels"]:
        errors.append(f"{prefix}: invalid complexity '{sample.get('complexity')}'")

    if sample.get("defense_bypass") not in taxonomy["defense_bypass_techniques"]:
        errors.append(f"{prefix}: invalid defense_bypass '{sample.get('defense_bypass')}'")

    if sample.get("severity") not in taxonomy["severity_levels"]:
        errors.append(f"{prefix}: invalid severity '{sample.get('severity')}'")

    if sample.get("ground_truth") not in ("safe", "unsafe"):
        errors.append(f"{prefix}: ground_truth must be 'safe' or 'unsafe'")

    # Benign controls (ground_truth=safe) must pin the attack-specific fields to
    # their sentinel values; attacks (ground_truth=unsafe) must not borrow the
    # benign sentinels. This keeps the two classes cleanly separable.
    if sample.get("ground_truth") == "safe":
        for field, sentinel in (
            ("attack_category", "benign"),
            ("attacker_intent", "benign"),
            ("defense_bypass", "none"),
            ("severity", "none"),
        ):
            if sample.get(field) != sentinel:
                errors.append(
                    f"{prefix}: benign sample must have {field}={sentinel!r}, "
                    f"got {sample.get(field)!r}"
                )
    elif sample.get("ground_truth") == "unsafe":
        for field in ("attack_category", "attacker_intent"):
            if sample.get(field) == "benign":
                errors.append(f"{prefix}: unsafe sample must not use benign sentinel for {field}")
        if sample.get("severity") == "none":
            errors.append(f"{prefix}: unsafe sample must not have severity='none'")

    valid_tools = set(taxonomy["target_tool_types"])
    for tool in sample.get("target_tools", []):
        if tool not in valid_tools:
            errors.append(f"{prefix}: invalid target_tool '{tool}'")

    conversation = sample.get("conversation", [])
    if not conversation:
        errors.append(f"{prefix}: conversation must not be empty")

    for i, turn in enumerate(conversation):
        if not isinstance(turn, dict):
            errors.append(f"{prefix}: conversation[{i}] must be a dict")
            continue
        if "role" not in turn:
            errors.append(f"{prefix}: conversation[{i}] missing 'role'")
        elif turn["role"] not in CONVERSATION_ROLES:
            errors.append(f"{prefix}: conversation[{i}] invalid role '{turn['role']}'")

        # Every turn must carry string content. The detectors scan untrusted
        # text only from string ``content`` (``_untrusted_text`` /
        # ``_tool_definition_text``), so a missing or non-string (list/dict)
        # content field would be an *unscanned* surface — a sample no detector
        # can read, silently lowering measured attack difficulty. Enforce the
        # invariant here so such a contribution fails validation loudly instead.
        if "content" not in turn:
            errors.append(f"{prefix}: conversation[{i}] missing 'content'")
        elif not isinstance(turn["content"], str):
            errors.append(
                f"{prefix}: conversation[{i}] 'content' expected str, got "
                f"{type(turn['content']).__name__} — detectors only scan string "
                "content, so structured content would be an unscanned blind spot"
            )

    for tool_def in sample.get("tools_available", []):
        if not isinstance(tool_def, dict):
            errors.append(f"{prefix}: tools_available entry must be a dict")
        elif "name" not in tool_def:
            errors.append(f"{prefix}: tool definition missing 'name'")

    return errors


def validate_file(filepath: Path) -> tuple[int, int, list[str]]:
    taxonomy = load_taxonomy()
    all_errors = []
    seen_ids = set()
    total = 0

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_num}: invalid JSON — {e}")
                continue

            total += 1

            sample_id = sample.get("id")
            if sample_id in seen_ids:
                all_errors.append(f"Line {line_num}: duplicate id '{sample_id}'")
            seen_ids.add(sample_id)

            all_errors.extend(validate_sample(sample, taxonomy, line_num))

    return total, len(all_errors), all_errors


def main():
    if len(sys.argv) < 2:
        data_dir = REPO_ROOT / "data"
        files = list(data_dir.glob("**/*.jsonl"))
        if not files:
            print("No JSONL files found in data/")
            sys.exit(0)
    else:
        files = [Path(p) for p in sys.argv[1:]]

    total_errors = 0
    for filepath in files:
        if not filepath.exists():
            print(f"File not found: {filepath}")
            total_errors += 1
            continue

        count, errors, error_list = validate_file(filepath)
        status = "PASS" if errors == 0 else "FAIL"
        print(f"[{status}] {filepath.name}: {count} samples, {errors} errors")
        for err in error_list[:20]:
            print(f"  {err}")
        if len(error_list) > 20:
            print(f"  ... and {len(error_list) - 20} more errors")
        total_errors += errors

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
