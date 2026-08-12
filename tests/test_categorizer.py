"""Tests for allocator/categorizer.py — fungible groups, classification, fallbacks."""

import pytest

import allocator.categorizer as categorizer

from allocator.categorizer import (
    assign_classification,
    assign_fungible_group,
    category_name,
)
from allocator.config import (
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
)


_SYNTHETIC_FUNGIBLE_GROUPS = {
    "fixture_aster": (0.61, ["Fixture-Aster :: "], "snack_piece"),
    "fixture_beryl": (0.86, ["Fixture-Beryl :: "], "snack_piece"),
    "fixture_cinder": (0.48, ["Fixture-Cinder :: "], "cooking_piece"),
}
_SYNTHETIC_CLASSIFICATIONS = {
    "fixture_aster": (
        ["Fixture-Aster :: "], "fixture_orbit", "fixture_use_a", "teal", "pentagon"
    ),
    "fixture_beryl": (
        ["Fixture-Beryl :: "], "fixture_wing", "fixture_use_b", "violet", "crescent"
    ),
}
_SYNTHETIC_FALLBACK = {
    CATEGORY_FRUIT: ("fixture_fruit", "fixture_use", "teal", "pentagon"),
    CATEGORY_VEGETABLES: ("fixture_veg", "fixture_use", "indigo", "triangle"),
}


@pytest.fixture(autouse=True)
def synthetic_categorization_rules(monkeypatch):
    monkeypatch.setattr(categorizer, "FUNGIBLE_GROUPS", _SYNTHETIC_FUNGIBLE_GROUPS)
    monkeypatch.setattr(
        categorizer, "ITEM_CLASSIFICATIONS", _SYNTHETIC_CLASSIFICATIONS
    )
    monkeypatch.setattr(categorizer, "CLASSIFICATION_FALLBACK", _SYNTHETIC_FALLBACK)


# ── assign_fungible_group ───────────────────────────────────────────────────


class TestAssignFungibleGroup:
    def test_aster_prefix_match(self):
        group, degree = assign_fungible_group("Fixture-Aster :: Variant")
        assert group == "fixture_aster"
        assert degree == 0.61

    def test_beryl_prefix_match(self):
        group, degree = assign_fungible_group("Fixture-Beryl :: Variant")
        assert group == "fixture_beryl"
        assert degree == 0.86

    def test_cinder_prefix_match(self):
        group, degree = assign_fungible_group("Fixture-Cinder :: Variant")
        assert group == "fixture_cinder"
        assert degree == 0.48

    def test_no_match(self):
        group, degree = assign_fungible_group("Unmatched fixture item")
        assert group is None
        assert degree == 0.0

    def test_case_insensitive_match(self):
        group, degree = assign_fungible_group("fixture-aster :: variant")
        assert group == "fixture_aster"

    def test_first_match_wins_known_limitation(self):
        """First matching group wins — if groups overlap, order matters."""
        group1, _ = assign_fungible_group("Fixture-Aster :: Variant")
        assert group1 == "fixture_aster"

    def test_partial_prefix_no_match(self):
        """A shortened prefix does not match a configured rule."""
        group, degree = assign_fungible_group("Fixture")
        assert group is None


# ── assign_classification ───────────────────────────────────────────────────


class TestAssignClassification:
    def test_aster_classification(self):
        sub, usage, colour, shape = assign_classification(
            "Fixture-Aster :: Variant", CATEGORY_FRUIT
        )
        assert (sub, usage, colour, shape) == (
            "fixture_orbit", "fixture_use_a", "teal", "pentagon"
        )

    def test_beryl_classification(self):
        sub, usage, colour, shape = assign_classification(
            "Fixture-Beryl :: Variant", CATEGORY_FRUIT
        )
        assert (sub, usage, colour, shape) == (
            "fixture_wing", "fixture_use_b", "violet", "crescent"
        )

    def test_fallback_fruit(self):
        """Unknown fruit item falls back to classification_fallback for fruit category."""
        sub, usage, colour, shape = assign_classification(
            "Unknown fixture item", CATEGORY_FRUIT
        )
        expected_sub = _SYNTHETIC_FALLBACK[CATEGORY_FRUIT][0]
        assert sub == expected_sub

    def test_fallback_veg(self):
        """Unknown veg item falls back to classification_fallback for veg category."""
        sub, usage, colour, shape = assign_classification(
            "Unknown fixture item", CATEGORY_VEGETABLES
        )
        expected_sub = _SYNTHETIC_FALLBACK[CATEGORY_VEGETABLES][0]
        assert sub == expected_sub

    def test_fallback_unknown_category(self):
        """Unknown category_id falls back to generic defaults."""
        sub, usage, colour, shape = assign_classification("Mystery Item", 999)
        assert sub == "other"
        assert usage == "cooking"

    def test_case_insensitive_match(self):
        sub, _, _, _ = assign_classification("fixture-aster :: variant", CATEGORY_FRUIT)
        assert sub == "fixture_orbit"


# ── category_name ───────────────────────────────────────────────────────────


class TestCategoryName:
    def test_known_category(self):
        categories = {10: "Fruit", 20: "Vegetables"}
        assert category_name(10, categories) == "fruit"
        assert category_name(20, categories) == "vegetables"

    def test_unknown_category(self):
        categories = {10: "Fruit"}
        assert category_name(99, categories) == "unknown"

    def test_lowercased(self):
        categories = {10: "FRUIT"}
        assert category_name(10, categories) == "fruit"
