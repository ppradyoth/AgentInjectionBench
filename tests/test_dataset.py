"""Dataset-integrity tests for AgentInjectionBench.

These run the shipped schema validator over the released JSONL and assert a set
of structural invariants that the benchmark relies on: stable IDs, taxonomy
coverage, valid enums, and well-formed conversations. They are the regression
guard behind the `data/**` CI workflow.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

import copy

from generation.validate_schema import (
    CONVERSATION_ROLES,
    SCHEMA,
    load_taxonomy,
    validate_file,
    validate_sample,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "agent_injection_bench.jsonl"
TEMPLATE_DIR = REPO_ROOT / "generation" / "templates"
ID_RE = re.compile(r"^AIB-\d{5}$")


@pytest.fixture(scope="module")
def samples() -> list[dict]:
    with open(DATA_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def taxonomy() -> dict:
    return load_taxonomy()


# ----------------------------- validator -----------------------------

def test_dataset_file_exists():
    assert DATA_FILE.exists(), f"missing dataset file: {DATA_FILE}"


def test_dataset_passes_shipped_validator():
    total, error_count, errors = validate_file(DATA_FILE)
    assert total > 0, "validator found no samples"
    assert error_count == 0, "schema errors:\n" + "\n".join(errors[:20])


def test_dataset_has_a_meaningful_number_of_samples(samples):
    assert len(samples) >= 100


# ----------------------------- ids -----------------------------

def test_ids_follow_the_aib_convention(samples):
    bad = [s["id"] for s in samples if not ID_RE.match(s["id"])]
    assert bad == [], f"ids not matching AIB-NNNNN: {bad[:10]}"


def test_ids_are_unique(samples):
    ids = [s["id"] for s in samples]
    assert len(set(ids)) == len(ids), "duplicate ids present"


# ----------------------------- taxonomy coverage -----------------------------

def test_every_taxonomy_category_is_represented(samples, taxonomy):
    present = {s["attack_category"] for s in samples}
    missing = set(taxonomy["attack_categories"]) - present
    assert missing == set(), f"attack categories with no samples: {missing}"


def test_enum_fields_only_use_taxonomy_values(samples, taxonomy):
    enum_map = {
        "attack_category": "attack_categories",
        "attacker_intent": "attacker_intents",
        "injection_surface": "injection_surfaces",
        "complexity": "complexity_levels",
        "defense_bypass": "defense_bypass_techniques",
        "severity": "severity_levels",
    }
    valid_tools = set(taxonomy["target_tool_types"])
    for s in samples:
        for field, tax_key in enum_map.items():
            assert s[field] in taxonomy[tax_key], (
                f"{s['id']}: {field}={s[field]!r} not in taxonomy[{tax_key}]"
            )
        for tool in s["target_tools"]:
            assert tool in valid_tools, f"{s['id']}: unknown target_tool {tool!r}"


def test_ground_truth_label_space(samples):
    labels = {s["ground_truth"] for s in samples}
    assert labels <= {"safe", "unsafe"}, f"unexpected ground_truth labels: {labels}"
    # The v0.1 release is an all-attack set; the 'unsafe' class must be present.
    assert "unsafe" in labels


def test_every_released_case_has_execution_contract(samples):
    for sample in samples:
        execution = sample.get("execution")
        assert isinstance(execution, dict), f"{sample['id']}: missing execution metadata"
        assert execution["security_property"]
        assert isinstance(execution["forbidden_tools"], list)
        assert isinstance(execution["forbidden_argument_patterns"], list)


# ----------------------------- structure -----------------------------

def test_required_fields_present_with_correct_types(samples):
    for s in samples:
        for field, expected_type in SCHEMA["required_fields"].items():
            assert field in s, f"{s.get('id')}: missing {field}"
            assert isinstance(s[field], expected_type), (
                f"{s['id']}: {field} should be {expected_type.__name__}"
            )


def test_conversations_are_well_formed(samples):
    for s in samples:
        convo = s["conversation"]
        assert convo, f"{s['id']}: empty conversation"
        for i, turn in enumerate(convo):
            assert isinstance(turn, dict), f"{s['id']}: turn {i} not a dict"
            assert turn.get("role") in CONVERSATION_ROLES, (
                f"{s['id']}: turn {i} invalid role {turn.get('role')!r}"
            )
            # Detectors only scan string content, so every turn must carry it.
            assert isinstance(turn.get("content"), str), (
                f"{s['id']}: turn {i} content must be a string (detectors can't "
                f"scan {type(turn.get('content')).__name__})"
            )


def test_validator_rejects_non_string_or_missing_turn_content(samples, taxonomy):
    # Structured (list/dict) or absent content is an unscanned surface — a sample
    # no detector can read. The validator must reject it, not pass it silently.
    base = copy.deepcopy(samples[0])

    structured = copy.deepcopy(base)
    structured["conversation"][0]["content"] = [{"type": "text", "text": "hi"}]
    errs = validate_sample(structured, taxonomy, 1)
    assert any("content" in e and "str" in e for e in errs), errs

    missing = copy.deepcopy(base)
    del missing["conversation"][0]["content"]
    errs = validate_sample(missing, taxonomy, 1)
    assert any("missing 'content'" in e for e in errs), errs

    # The unmodified sample still validates clean (guards against over-tightening).
    assert validate_sample(copy.deepcopy(base), taxonomy, 1) == []


def test_validator_checks_execution_metadata(samples, taxonomy):
    base = copy.deepcopy(samples[0])
    base["execution"] = {
        "canary_secrets": ["AIB_CANARY"],
        "forbidden_tools": ["email_send"],
        "forbidden_argument_patterns": ["system_prompt"],
        "forbid_state_writes": True,
    }
    assert validate_sample(base, taxonomy, 1) == []
    base["execution"]["canary_secrets"] = ["AIB_CANARY", 3]
    assert any("canary_secrets" in error for error in validate_sample(base, taxonomy, 1))


def test_tool_definitions_have_names(samples):
    for s in samples:
        for tool in s["tools_available"]:
            assert isinstance(tool, dict), f"{s['id']}: tool def not a dict"
            assert tool.get("name"), f"{s['id']}: tool def missing name"


def test_validator_rejects_non_string_tool_definition_text(samples, taxonomy):
    # Tool definitions are a detector-scanned surface too (`_tool_definition_text`
    # reads the top-level description and every per-parameter description/title).
    # A non-string description/title is an unscanned blind spot — reject it,
    # mirroring the turn-content invariant.
    base = copy.deepcopy(samples[0])

    bad_desc = copy.deepcopy(base)
    bad_desc["tools_available"][0]["description"] = {"text": "structured"}
    errs = validate_sample(bad_desc, taxonomy, 1)
    assert any("description" in e and "str" in e for e in errs), errs

    bad_param = copy.deepcopy(base)
    bad_param["tools_available"][0]["parameters"] = {
        "properties": {"x": {"description": ["not", "a", "string"]}}
    }
    errs = validate_sample(bad_param, taxonomy, 1)
    assert any("schema" in e and "str" in e for e in errs), errs

    # A clean tool schema with string parameter descriptions still validates.
    ok = copy.deepcopy(base)
    ok["tools_available"][0]["parameters"] = {
        "properties": {"x": {"type": "string", "description": "A normal parameter."}}
    }
    assert validate_sample(ok, taxonomy, 1) == []


# ----------------------------- taxonomy + templates -----------------------------

def test_taxonomy_has_all_referenced_sections(taxonomy):
    for key in (
        "attack_categories",
        "attacker_intents",
        "injection_surfaces",
        "complexity_levels",
        "defense_bypass_techniques",
        "severity_levels",
        "target_tool_types",
    ):
        assert key in taxonomy, f"taxonomy missing section {key}"


def test_seed_templates_are_valid_and_map_to_taxonomy(taxonomy):
    template_files = sorted(TEMPLATE_DIR.glob("*.yaml"))
    assert template_files, "no seed templates found"
    for f in template_files:
        data = yaml.safe_load(f.read_text())
        assert "category" in data, f"{f.name}: missing category"
        assert "seeds" in data, f"{f.name}: missing seeds"
        assert data["seeds"], f"{f.name}: no seeds defined"
        assert data["category"] in taxonomy["attack_categories"], (
            f"{f.name}: category {data['category']!r} not in taxonomy"
        )
