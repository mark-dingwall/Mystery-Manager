"""Tests for allocator/config.py — detect_pack_size, config loading."""

import pytest

from allocator.config import (
    BOX_TIERS,
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    DIVERSITY_WEIGHTS,
    FUNGIBLE_GROUPS,
    ITEM_CLASSIFICATIONS,
    VALUE_CEILING_PCT,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
    detect_pack_size,
)


# ── detect_pack_size ────────────────────────────────────────────────────────


class TestDetectPackSize:
    def test_3_pack(self):
        assert detect_pack_size("Avocado - Hass (3 pack)") == 3

    def test_5_pack(self):
        assert detect_pack_size("Lemons - 5 pack") == 5

    def test_value_pack(self):
        assert detect_pack_size("Avocado - Hass (3 value pack)") == 3

    def test_no_pack(self):
        assert detect_pack_size("Broccoli") == 1

    def test_case_insensitive(self):
        assert detect_pack_size("Lemons - 5 Pack") == 5
        assert detect_pack_size("Lemons - 5 PACK") == 5

    def test_no_space_before_pack(self):
        """'5pack' should match (no space before 'pack')."""
        assert detect_pack_size("Item 5pack") == 5

    def test_single_digit(self):
        assert detect_pack_size("Item (2 pack)") == 2

    def test_large_pack(self):
        assert detect_pack_size("Herbs - 10 pack") == 10


# ── Config loading ──────────────────────────────────────────────────────────


class TestConfigLoading:
    def test_box_tiers_loaded(self):
        assert "small" in BOX_TIERS
        assert "medium" in BOX_TIERS
        assert "large" in BOX_TIERS

    def test_box_tiers_have_price_and_target(self):
        for tier in BOX_TIERS.values():
            assert "price" in tier
            assert "target_value" in tier
            assert tier["target_value"] > 0

    def test_target_value_is_percentage_of_price(self):
        for tier in BOX_TIERS.values():
            expected = round(tier["price"] * 1.15)  # BOX_TARGET_PCT=115
            assert tier["target_value"] == expected

    def test_category_ids_are_ints(self):
        assert isinstance(CATEGORY_FRUIT, int)
        assert isinstance(CATEGORY_VEGETABLES, int)

    def test_diversity_weights_positive(self):
        total = sum(DIVERSITY_WEIGHTS.values())
        assert total > 0.0
        for v in DIVERSITY_WEIGHTS.values():
            assert v >= 0.0

    def test_fungible_groups_structure(self):
        for group_name, (degree, prefixes, qty_class) in FUNGIBLE_GROUPS.items():
            assert isinstance(degree, (int, float))
            assert isinstance(prefixes, (list, tuple))
            assert len(prefixes) > 0
            assert isinstance(qty_class, str)

    def test_item_classifications_structure(self):
        for key, (prefixes, sub_cat, usage, colour, shape) in ITEM_CLASSIFICATIONS.items():
            assert isinstance(prefixes, (list, tuple))
            assert isinstance(sub_cat, str)
            assert isinstance(usage, str)
            assert isinstance(colour, str)
            assert isinstance(shape, str)

    def test_scoring_params(self):
        assert VALUE_SWEET_FROM == 114
        assert VALUE_SWEET_TO == 117
        assert VALUE_PENALTY_EXPONENT == 1.25
        assert VALUE_CEILING_PCT == 1.30
