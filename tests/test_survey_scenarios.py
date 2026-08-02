"""Tests for the packer-survey scenario generator's data-contract guarantees.

Covers the two guarantees the Jointly.Shop consumer relies on:
  1. Box quantities are always integers >= 1 — a fractional historical split
     (e.g. 0.5 of a bulk bag) must coerce to a whole unit, never truncate to 0.
  2. `validate_scenarios` rejects any out-of-spec scenario before it is written.
"""

import copy

import pytest

from scripts.generate_survey_scenarios import (
    _box_allocations_from_csv,
    validate_scenarios,
)


# ---------------------------------------------------------------------------
# _box_allocations_from_csv — fractional coercion
# ---------------------------------------------------------------------------

def test_fractional_qty_coerced_to_whole_unit():
    """A 0.5 split must become qty 1, not truncate to 0 (the prod-500 cause)."""
    hist = {100: {"box_a": 0.5}}
    assert _box_allocations_from_csv("box_a", hist) == {100: 1}


def test_whole_qty_preserved_as_int():
    hist = {100: {"box_a": 4.0}, 200: {"box_a": 1.0}}
    allocs = _box_allocations_from_csv("box_a", hist)
    assert allocs == {100: 4, 200: 1}
    assert all(isinstance(q, int) for q in allocs.values())


def test_fractional_over_one_rounds():
    hist = {100: {"box_a": 1.5}, 200: {"box_a": 2.4}}
    assert _box_allocations_from_csv("box_a", hist) == {100: 2, 200: 2}


def test_zero_and_absent_excluded():
    hist = {100: {"box_a": 0}, 200: {"box_b": 3.0}}
    assert _box_allocations_from_csv("box_a", hist) == {}


def test_no_output_qty_is_ever_below_one():
    hist = {i: {"box_a": q} for i, q in enumerate([0.1, 0.5, 0.9, 1.0, 2.5])}
    allocs = _box_allocations_from_csv("box_a", hist)
    assert allocs and all(q >= 1 for q in allocs.values())


# ---------------------------------------------------------------------------
# validate_scenarios — contract enforcement
# ---------------------------------------------------------------------------

def _valid_scenario(sid="t1_x_001"):
    return {
        "id": sid,
        "type": "tier1_random",
        "target_dimension": None,
        "source_offer_id": 100,
        "source_box_name": "box_a",
        "box": {
            "tier": "small",
            "target_value_cents": 2000,
            "current_value_cents": 1200,
            "current_value_pct": 60.0,
            "current_items": [
                {"item_id": 1, "name": "Apples", "price_cents": 300, "qty": 2},
                {"item_id": 2, "name": "Pears", "price_cents": 300, "qty": 1},
            ],
        },
        "candidates": [
            {"item_id": 3, "name": "Kiwi", "price_cents": 250},
            {"item_id": 4, "name": "Plum", "price_cents": 200},
        ],
    }


def test_valid_scenario_passes():
    assert validate_scenarios([_valid_scenario()]) == []


def test_box_item_missing_qty_flagged():
    s = _valid_scenario()
    del s["box"]["current_items"][0]["qty"]  # the exact prod breakage
    errs = validate_scenarios([s])
    assert any("bad qty" in e for e in errs)


def test_box_item_qty_zero_flagged():
    s = _valid_scenario()
    s["box"]["current_items"][0]["qty"] = 0
    assert any("bad qty" in e for e in validate_scenarios([s]))


def test_box_item_qty_float_flagged():
    s = _valid_scenario()
    s["box"]["current_items"][0]["qty"] = 1.5
    assert any("bad qty" in e for e in validate_scenarios([s]))


def test_duplicate_candidate_item_id_flagged():
    s = _valid_scenario()
    s["candidates"][1]["item_id"] = s["candidates"][0]["item_id"]
    assert any("duplicate candidate" in e for e in validate_scenarios([s]))


def test_duplicate_scenario_id_flagged():
    a, b = _valid_scenario("dup"), _valid_scenario("dup")
    assert any("duplicate scenario id" in e for e in validate_scenarios([a, b]))


def test_missing_required_key_flagged():
    s = _valid_scenario()
    del s["box"]["current_value_cents"]
    assert any("current_value_cents" in e for e in validate_scenarios([s]))


def test_no_candidates_flagged():
    s = _valid_scenario()
    s["candidates"] = []
    assert any("no candidates" in e for e in validate_scenarios([s]))


@pytest.mark.parametrize("qty", [1, 2, 5])
def test_positive_int_qty_accepted(qty):
    s = _valid_scenario()
    s["box"]["current_items"][0]["qty"] = qty
    assert validate_scenarios([s]) == []


def test_candidate_carrying_qty_flagged():
    """Omitted-qty is the placed-vs-candidate discriminator; a candidate with
    qty would render as a placed item on the consumer."""
    s = _valid_scenario()
    s["candidates"][0]["qty"] = 1
    assert any("must not carry qty" in e for e in validate_scenarios([s]))


def test_empty_current_items_flagged():
    s = _valid_scenario()
    s["box"]["current_items"] = []
    assert any("empty current_items" in e for e in validate_scenarios([s]))
