"""Tests for the benign control split.

The benign controls are the precision axis of the benchmark: matched-benign
samples that look attack-adjacent but contain no injection. These tests pin
their integrity and guard against drift between the generator
(``generation.benign_controls``) and the committed dataset.
"""

import json
from pathlib import Path

import pytest

from generation.benign_controls import build_benign_samples
from generation.validate_schema import load_taxonomy, validate_sample

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "agent_injection_bench.jsonl"


@pytest.fixture(scope="module")
def dataset() -> list[dict]:
    with open(DATA_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def benign() -> list[dict]:
    return build_benign_samples()


def test_generator_produces_a_meaningful_number(benign):
    assert len(benign) >= 20


def test_generated_benign_samples_are_schema_valid(benign):
    taxonomy = load_taxonomy()
    for i, s in enumerate(benign):
        errors = validate_sample(s, taxonomy, i)
        assert errors == [], f"{s['id']}: {errors}"


def test_generated_benign_samples_use_sentinels(benign):
    for s in benign:
        assert s["ground_truth"] == "safe"
        assert s["attack_category"] == "benign"
        assert s["attacker_intent"] == "benign"
        assert s["defense_bypass"] == "none"
        assert s["severity"] == "none"
        assert s["metadata"]["control"] is True


def test_matched_category_points_at_a_real_attack_family(benign):
    taxonomy = load_taxonomy()
    attack_cats = set(taxonomy["attack_categories"]) - {"benign"}
    for s in benign:
        assert s["metadata"]["matched_category"] in attack_cats


def test_committed_dataset_contains_the_generated_split(dataset, benign):
    """The dataset's benign rows must be exactly what the generator emits —
    no manual edits, no drift. Regenerate with
    ``python -m generation.benign_controls --append`` if this fails."""
    by_id = {s["id"]: s for s in dataset}
    for gen in benign:
        assert gen["id"] in by_id, f"{gen['id']} missing from dataset"
        assert by_id[gen["id"]] == gen, f"{gen['id']} drifted from generator output"


def test_benign_ids_are_contiguous_after_the_attack_set(benign):
    ids = [int(s["id"].split("-")[1]) for s in benign]
    assert ids == list(range(ids[0], ids[0] + len(ids))), "benign ids must be contiguous"
    assert ids[0] == 121, "benign split should start at AIB-00121"
