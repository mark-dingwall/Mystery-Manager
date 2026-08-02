"""Tests for allocator/strategies/_helpers.py — constraint checks, mutations, diversity."""

import pytest

from allocator.config import BOX_TIERS, VALUE_CEILING_PCT
from allocator.strategies._helpers import (
    assign_item,
    box_deficit,
    box_fungible_groups,
    can_assign,
    compute_available_tags,
    compute_diversity_score,
    has_hard_fungible_conflict,
    remove_item,
    would_exceed_ceiling,
)


# ── would_exceed_ceiling ────────────────────────────────────────────────────


class TestWouldExceedCeiling:
    def test_under_ceiling(self, make_item, make_box, make_result):
        item = make_item(id=1, price=100)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        assert not would_exceed_ceiling(box, item, 1, result)

    def test_at_ceiling(self, make_item, make_box, make_result):
        """Exactly at ceiling → not exceeded (<=)."""
        ceiling = int(VALUE_CEILING_PCT * BOX_TIERS["small"]["price"])
        item = make_item(id=1, price=ceiling)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        assert not would_exceed_ceiling(box, item, 1, result)

    def test_over_ceiling(self, make_item, make_box, make_result):
        ceiling = int(VALUE_CEILING_PCT * BOX_TIERS["small"]["price"])
        item = make_item(id=1, price=ceiling + 1)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        assert would_exceed_ceiling(box, item, 1, result)

    def test_qty_multiplied(self, make_item, make_box, make_result):
        """Adding qty > 1 counts total additional value."""
        ceiling = int(VALUE_CEILING_PCT * BOX_TIERS["small"]["price"])
        per_unit = ceiling // 2
        item = make_item(id=1, price=per_unit)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        # qty=2 → 2*per_unit = ceiling → not exceeded
        assert not would_exceed_ceiling(box, item, 2, result)
        # qty=3 → 3*per_unit > ceiling → exceeded
        assert would_exceed_ceiling(box, item, 3, result)

    def test_existing_allocations_counted(self, make_item, make_box, make_result):
        ceiling = int(VALUE_CEILING_PCT * BOX_TIERS["small"]["price"])
        item = make_item(id=1, price=ceiling - 10)
        box = make_box(tier="small", allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        # Already have ceiling-10, adding another ceiling-10 → exceeds
        assert would_exceed_ceiling(box, item, 1, result)


# ── has_hard_fungible_conflict ──────────────────────────────────────────────


class TestHasHardFungibleConflict:
    """Tests for per-item and per-group hard ceiling checks."""

    def test_no_fungible_group_under_limit(self, make_item, make_box, make_result):
        """Singleton item with qty=1 on empty box → no conflict."""
        item = make_item(id=1, fungible_group=None)
        box = make_box()
        result = make_result(items=[item], boxes=[box])
        assert not has_hard_fungible_conflict(item, box, result)

    def test_fungible_group_within_limit(self, make_item, make_box, make_result):
        """Adding 1 banana when 1 already in box → within per-item limit."""
        existing = make_item(id=1, fungible_group="banana", fungible_degree=1.0)
        new_item = make_item(id=2, fungible_group="banana", fungible_degree=1.0)
        box = make_box(allocations={1: 1})
        result = make_result(items=[existing, new_item], boxes=[box])
        assert not has_hard_fungible_conflict(new_item, box, result)

    def test_per_item_exceeds_limit(self, make_item, make_box, make_result):
        """Adding would exceed 2x per-item allowance → conflict."""
        # snack_piece small = 2, so 2*2 = 4 is ceiling
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(allocations={1: 4})  # already at ceiling
        result = make_result(items=[item], boxes=[box])
        assert has_hard_fungible_conflict(item, box, result)

    def test_different_groups_no_conflict(self, make_item, make_box, make_result):
        existing = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        new_item = make_item(id=2, fungible_group="banana", fungible_degree=1.0)
        box = make_box(allocations={1: 1})
        result = make_result(items=[existing, new_item], boxes=[box])
        assert not has_hard_fungible_conflict(new_item, box, result)

    def test_empty_box_no_conflict(self, make_item, make_box, make_result):
        item = make_item(id=1, fungible_group="banana", fungible_degree=1.0)
        box = make_box()
        result = make_result(items=[item], boxes=[box])
        assert not has_hard_fungible_conflict(item, box, result)

    def test_large_tier_higher_limit(self, make_item, make_box, make_result):
        """Large tier has higher per-item allowance."""
        # snack_piece large = 6, so 2*6 = 12 is ceiling
        existing = make_item(id=1, fungible_group="banana", fungible_degree=1.0)
        new_item = make_item(id=2, fungible_group="banana", fungible_degree=1.0)
        box = make_box(tier="large", allocations={1: 7})
        result = make_result(items=[existing, new_item], boxes=[box])
        assert not has_hard_fungible_conflict(new_item, box, result)


# ── box_fungible_groups ─────────────────────────────────────────────────────


class TestBoxFungibleGroups:
    def test_empty_box(self, make_box, make_result, make_item):
        box = make_box()
        result = make_result(items=[make_item()], boxes=[box])
        assert box_fungible_groups(box, result) == set()

    def test_mixed_groups(self, make_item, make_box, make_result):
        items = [
            make_item(id=1, fungible_group="apple"),
            make_item(id=2, fungible_group="banana"),
            make_item(id=3, fungible_group=None),
        ]
        box = make_box(allocations={1: 1, 2: 1, 3: 1})
        result = make_result(items=items, boxes=[box])
        assert box_fungible_groups(box, result) == {"apple", "banana"}


# ── can_assign ──────────────────────────────────────────────────────────────


class TestCanAssign:
    def test_basic_assignment_ok(self, make_item, make_box, make_result):
        item = make_item(id=1, price=100, overage=5)
        box = make_box()
        result = make_result(items=[item], boxes=[box])
        assert can_assign(item, 1, box, result)

    def test_excluded_item(self, make_item, make_box, make_result):
        from allocator.models import ExclusionRule
        item = make_item(id=1, name="Bananas")
        rule = ExclusionRule(pattern="banana", source="note")
        box = make_box(exclusions=[rule])
        result = make_result(items=[item], boxes=[box])
        assert not can_assign(item, 1, box, result)

    def test_insufficient_overage(self, make_item, make_box, make_result):
        item = make_item(id=1, overage=2)
        box = make_box()
        result = make_result(items=[item], boxes=[box])
        assert not can_assign(item, 3, box, result)

    def test_would_exceed_ceiling(self, make_item, make_box, make_result):
        ceiling = int(VALUE_CEILING_PCT * BOX_TIERS["small"]["price"])
        item = make_item(id=1, price=ceiling + 1, overage=5)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        assert not can_assign(item, 1, box, result)

    def test_hard_fungible_conflict(self, make_item, make_box, make_result):
        """Adding would exceed 2x per-item allowance → can't assign."""
        # snack_piece small = 2, ceiling = 4
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7, overage=10)
        box = make_box(allocations={1: 4})
        result = make_result(items=[item], boxes=[box])
        assert not can_assign(item, 1, box, result)

    def test_all_constraints_pass(self, make_item, make_box, make_result):
        item = make_item(id=1, price=100, overage=10, fungible_group=None)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        assert can_assign(item, 1, box, result)


# ── assign_item / remove_item ──────────────────────────────────────────────


class TestAssignRemoveItem:
    def test_assign_to_empty_box(self, make_box):
        box = make_box()
        assign_item(1, 2, box)
        assert box.allocations == {1: 2}

    def test_assign_increments(self, make_box):
        box = make_box(allocations={1: 3})
        assign_item(1, 2, box)
        assert box.allocations[1] == 5

    def test_assign_new_item(self, make_box):
        box = make_box(allocations={1: 1})
        assign_item(2, 1, box)
        assert box.allocations == {1: 1, 2: 1}

    def test_remove_partial(self, make_box):
        box = make_box(allocations={1: 5})
        remove_item(1, 2, box)
        assert box.allocations[1] == 3

    def test_remove_all(self, make_box):
        box = make_box(allocations={1: 3})
        remove_item(1, 3, box)
        assert 1 not in box.allocations

    def test_remove_more_than_available_clamps_to_zero(self, make_box):
        box = make_box(allocations={1: 2})
        remove_item(1, 5, box)
        assert 1 not in box.allocations

    def test_remove_nonexistent_item(self, make_box):
        box = make_box()
        remove_item(99, 1, box)
        assert 99 not in box.allocations


# ── box_deficit ─────────────────────────────────────────────────────────────


class TestBoxDeficit:
    def test_empty_box(self, make_box, make_result, make_item):
        box = make_box(tier="small")
        result = make_result(items=[make_item()], boxes=[box])
        assert box_deficit(box, result) == box.target_value

    def test_under_target(self, make_item, make_box, make_result):
        item = make_item(id=1, price=500)
        box = make_box(tier="small", allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        expected = box.target_value - 500
        assert box_deficit(box, result) == expected

    def test_over_target(self, make_item, make_box, make_result):
        item = make_item(id=1, price=5000)
        box = make_box(tier="small", allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        assert box_deficit(box, result) < 0


# ── compute_available_tags ──────────────────────────────────────────────────


class TestComputeAvailableTags:
    def test_collects_all_dimensions(self, make_item, make_result):
        items = [
            make_item(id=1, sub_category="tropical", usage_type="snacking",
                      colour="yellow", shape="long"),
            make_item(id=2, sub_category="root_veg", usage_type="cooking",
                      colour="orange", shape="long"),
        ]
        result = make_result(items=items)
        tags = compute_available_tags(result)
        assert tags["sub_category"] == {"tropical", "root_veg"}
        assert tags["usage"] == {"snacking", "cooking"}
        assert tags["colour"] == {"yellow", "orange"}
        assert tags["shape"] == {"long"}

    def test_empty_items(self, make_result):
        result = make_result(items=[])
        tags = compute_available_tags(result)
        for dim_tags in tags.values():
            assert len(dim_tags) == 0

    def test_empty_tags_ignored(self, make_item, make_result):
        item = make_item(id=1, sub_category="", usage_type="", colour="", shape="")
        result = make_result(items=[item])
        tags = compute_available_tags(result)
        for dim_tags in tags.values():
            assert len(dim_tags) == 0

    def test_preference_filters_items(self, make_item, make_result):
        """fruit_only preference should exclude veg items from available tags."""
        from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES
        items = [
            make_item(id=1, category_id=CATEGORY_FRUIT, sub_category="tropical"),
            make_item(id=2, category_id=CATEGORY_VEGETABLES, sub_category="root_veg"),
        ]
        result = make_result(items=items)
        fruit_tags = compute_available_tags(result, preference="fruit_only")
        assert "root_veg" not in fruit_tags["sub_category"]
        assert "tropical" in fruit_tags["sub_category"]


# ── compute_diversity_score ─────────────────────────────────────────────────


class TestComputeDiversityScore:
    def test_empty_box_zero(self, make_box, make_result, make_item):
        items = [make_item(id=1, sub_category="tropical")]
        box = make_box()
        result = make_result(items=items, boxes=[box])
        tags = compute_available_tags(result)
        score = compute_diversity_score(box, result, tags)
        assert score == 0.0

    def test_single_item_in_box(self, make_item, make_box, make_result):
        from allocator.config import DIVERSITY_WEIGHTS
        item = make_item(id=1, sub_category="tropical", usage_type="snacking",
                         colour="yellow", shape="long")
        box = make_box(allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        tags = compute_available_tags(result)
        score = compute_diversity_score(box, result, tags)
        # With only 1 available tag per dimension, coverage is full
        expected = sum(DIVERSITY_WEIGHTS.values())
        assert abs(score - expected) < 0.01

    def test_diverse_box_higher_score(self, make_item, make_box, make_result):
        """More diverse allocations → higher score."""
        items = [
            make_item(id=1, sub_category="tropical", usage_type="snacking",
                      colour="yellow", shape="long"),
            make_item(id=2, sub_category="root_veg", usage_type="cooking",
                      colour="orange", shape="chunky"),
            make_item(id=3, sub_category="pome_fruit", usage_type="snacking",
                      colour="red", shape="round"),
        ]
        diverse_box = make_box(name="diverse@test", allocations={1: 1, 2: 1, 3: 1})
        mono_box = make_box(name="mono@test", allocations={1: 3})
        result = make_result(items=items, boxes=[diverse_box, mono_box])
        tags = compute_available_tags(result)
        diverse_score = compute_diversity_score(diverse_box, result, tags)
        mono_score = compute_diversity_score(mono_box, result, tags)
        assert diverse_score > mono_score

    def test_no_available_tags_full_marks(self, make_item, make_box, make_result):
        """If no tags available in a dimension → full marks for that dimension."""
        from allocator.config import DIVERSITY_WEIGHTS
        item = make_item(id=1, sub_category="", usage_type="", colour="", shape="")
        box = make_box(allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        score = compute_diversity_score(box, result, tags)
        expected = sum(DIVERSITY_WEIGHTS.values())
        assert abs(score - expected) < 0.01

    def test_score_bounded(self, sample_items, make_box, make_result):
        from allocator.config import DIVERSITY_WEIGHTS
        box = make_box(allocations={i.id: 1 for i in sample_items})
        result = make_result(items=sample_items, boxes=[box])
        tags = compute_available_tags(result)
        score = compute_diversity_score(box, result, tags)
        assert 0.0 <= score <= sum(DIVERSITY_WEIGHTS.values()) + 0.01
