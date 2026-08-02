"""Tests for allocator/strategies/_scoring.py — value_penalty, same-item, group-concentration, box/total penalty."""

import pytest

from allocator.strategies._scoring import (
    _resolve_item_allowance,
    box_penalty,
    group_concentration_penalty_for_box,
    max_value_share_penalty_for_box,
    same_item_penalty_for_box,
    size_floor_penalty_for_box,
    total_penalty,
    value_penalty,
)
from allocator.strategies._helpers import compute_available_tags


# ── value_penalty ───────────────────────────────────────────────────────────


class TestValuePenalty:
    def test_sweet_spot_zero(self):
        """Values within [114, 117] should have zero penalty."""
        assert value_penalty(114.0) == 0.0
        assert value_penalty(115.5) == 0.0
        assert value_penalty(117.0) == 0.0

    def test_below_sweet_spot(self):
        """Below 114 → penalty increases."""
        p = value_penalty(110.0)
        assert p > 0.0
        # 114 - 110 = 4, 4^1.25 ≈ 5.66
        assert abs(p - 4.0 ** 1.25) < 0.01

    def test_above_sweet_spot(self):
        """Above 117 → penalty increases."""
        p = value_penalty(120.0)
        assert p > 0.0
        # 120 - 117 = 3, 3^1.25 ≈ 3.95
        assert abs(p - 3.0 ** 1.25) < 0.01

    def test_symmetry(self):
        """Equal distance from sweet spot edges → equal penalty."""
        assert abs(value_penalty(109.0) - value_penalty(122.0)) < 0.01

    def test_monotonically_increasing_below(self):
        """Further below sweet spot → higher penalty."""
        assert value_penalty(100.0) > value_penalty(110.0) > value_penalty(113.0) > 0.0

    def test_monotonically_increasing_above(self):
        """Further above sweet spot → higher penalty."""
        assert value_penalty(130.0) > value_penalty(125.0) > value_penalty(118.0) > 0.0

    def test_zero_value(self):
        """0% value → large penalty."""
        p = value_penalty(0.0)
        assert p > 100.0  # 114^1.25 is very large

    def test_boundary_just_outside(self):
        """Just outside sweet spot → small nonzero penalty."""
        assert value_penalty(113.9) > 0.0
        assert value_penalty(117.1) > 0.0
        assert value_penalty(113.9) < 1.0  # 0.1^1.25 is tiny


# ── same_item_penalty ───────────────────────────────────────────────────────


class TestSameItemPenalty:
    def test_within_allowance_zero(self, make_item, make_box, make_result):
        """Qty within per-item allowance → zero penalty."""
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        pen = same_item_penalty_for_box(box, result)
        assert pen == 0.0

    def test_above_allowance_penalised(self, make_item, make_box, make_result):
        """Qty > per-item allowance → positive penalty proportional to price."""
        # snack_piece small allowance = 2, so qty=4 → excess=2
        item = make_item(id=1, price=400, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 4})
        result = make_result(items=[item], boxes=[box])
        pen = same_item_penalty_for_box(box, result)
        assert pen > 0.0

    def test_no_fungible_group_conservative(self, make_item, make_box, make_result):
        """Item without fungible group uses usage-based allowance."""
        item = make_item(id=1, price=500, fungible_group=None, usage_type="cooking")
        box = make_box(tier="small", allocations={1: 3})
        result = make_result(items=[item], boxes=[box])
        pen = same_item_penalty_for_box(box, result)
        # cooking_piece small = 1, so excess = 2
        assert pen > 0.0


# ── group_concentration_penalty ─────────────────────────────────────────────


class TestGroupConcentrationPenalty:
    def test_within_group_allowance_zero(self, make_item, make_box, make_result):
        """Group load within allowance → zero penalty."""
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 2})
        result = make_result(items=[item], boxes=[box])
        pen = group_concentration_penalty_for_box(box, result)
        assert pen == 0.0

    def test_above_group_allowance_penalised(self, make_item, make_box, make_result):
        """Group load above allowance → positive penalty."""
        from allocator.config import GROUP_ALLOWANCES
        # apple small allowance = 3, snack_piece small = 2
        # Items within item allowance: min(qty, 2) per item
        # Need 4 different apple items each with qty=2 → group_load=4*2=8 > 3
        items = [
            make_item(id=i, fungible_group="apple", fungible_degree=0.7)
            for i in range(1, 5)
        ]
        box = make_box(tier="small", allocations={i: 2 for i in range(1, 5)})
        result = make_result(items=items, boxes=[box])
        pen = group_concentration_penalty_for_box(box, result)
        assert pen > 0.0

    def test_singleton_no_group_penalty(self, make_item, make_box, make_result):
        """Items without fungible group → no group concentration penalty."""
        item = make_item(id=1, fungible_group=None)
        box = make_box(tier="small", allocations={1: 10})
        result = make_result(items=[item], boxes=[box])
        pen = group_concentration_penalty_for_box(box, result)
        assert pen == 0.0  # singletons skip group penalty

    def test_min_caps_group_load(self, make_item, make_box, make_result):
        """min(qty, allowance) ensures excess qty doesn't inflate group load."""
        # snack_piece small allowance = 2
        # 1 item with qty=10: group_load = min(10, 2) = 2
        item = make_item(id=1, fungible_group="apple", fungible_degree=0.7)
        box = make_box(tier="small", allocations={1: 10})
        result = make_result(items=[item], boxes=[box])
        pen = group_concentration_penalty_for_box(box, result)
        # group_load = min(10, 2) = 2, apple small allowance = 3, 2 < 3 → no penalty
        assert pen == 0.0


# ── max_value_share_penalty ─────────────────────────────────────────────────


class TestMaxValueSharePenalty:
    def test_balanced_box_zero(self, make_item, make_box, make_result):
        """Multiple equal-priced items → max share below threshold."""
        items = [make_item(id=i, price=400) for i in range(1, 7)]
        box = make_box(tier="small", allocations={i: 1 for i in range(1, 7)})
        result = make_result(items=items, boxes=[box])
        pen = max_value_share_penalty_for_box(box, result)
        # 6 items, max share = 1/6 ≈ 0.167 < 0.20 threshold
        assert pen == 0.0

    def test_single_item_penalised(self, make_item, make_box, make_result):
        """Single item has 100% share → heavy penalty."""
        item = make_item(id=1, price=2300)
        box = make_box(tier="small", allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        pen = max_value_share_penalty_for_box(box, result)
        # share = 1.0, excess = 0.8, penalty = 0.8 * 15.0 = 12.0
        assert pen > 10.0

    def test_empty_box_zero(self, make_box, make_result, make_item):
        box = make_box()
        result = make_result(items=[make_item()], boxes=[box])
        pen = max_value_share_penalty_for_box(box, result)
        assert pen == 0.0


# ── size_floor_penalty ──────────────────────────────────────────────────────


class TestSizeFloorPenalty:
    def test_above_target_zero(self, make_item, make_box, make_result):
        """Total size above target → zero penalty."""
        items = [make_item(id=i, size=3) for i in range(1, 6)]
        box = make_box(tier="small", allocations={i: 1 for i in range(1, 6)})
        result = make_result(items=items, boxes=[box])
        pen = size_floor_penalty_for_box(box, result)
        # total_size = 15, target = 10 → no penalty
        assert pen == 0.0

    def test_below_target_penalised(self, make_item, make_box, make_result):
        """Total size below target → positive penalty."""
        item = make_item(id=1, size=1)
        box = make_box(tier="small", allocations={1: 2})
        result = make_result(items=[item], boxes=[box])
        pen = size_floor_penalty_for_box(box, result)
        # total_size = 2, target = 10, deficit = 8, pen = 8 * 0.5 = 4.0
        assert pen > 0.0


# ── box_penalty ─────────────────────────────────────────────────────────────


class TestBoxPenalty:
    def test_perfect_box_low_penalty(self, make_item, make_box, make_result):
        """Box at sweet spot with good diversity → low penalty."""
        items = [
            make_item(id=1, price=800, size=3, sub_category="tropical", usage_type="snacking",
                      colour="yellow", shape="long"),
            make_item(id=2, price=750, size=2, sub_category="root_veg", usage_type="cooking",
                      colour="orange", shape="long"),
            make_item(id=3, price=750, size=3, sub_category="pome_fruit", usage_type="snacking",
                      colour="red", shape="round"),
        ]
        box = make_box(allocations={1: 1, 2: 1, 3: 1})
        result = make_result(items=items, boxes=[box])
        tags = compute_available_tags(result)
        pen = box_penalty(box, result, tags)
        assert pen < 20.0

    def test_empty_box_has_penalty(self, make_box, make_result, make_item):
        items = [make_item(id=1)]
        box = make_box()
        result = make_result(items=items, boxes=[box])
        tags = compute_available_tags(result)
        pen = box_penalty(box, result, tags)
        # 0% value → high value penalty, no diversity → high diversity penalty
        assert pen > 50.0

    def test_includes_same_item_penalty(self, make_item, make_box, make_result):
        """box_penalty should include a same_item component."""
        # snack_piece small allowance = 2, qty=5 → excess=3
        item = make_item(id=1, price=460, fungible_group="apple", fungible_degree=0.7)
        box = make_box(allocations={1: 5})
        result = make_result(items=[item], boxes=[box])
        tags = compute_available_tags(result)
        pen_with = box_penalty(box, result, tags)
        pen_without = box_penalty(box, result, tags, params={"same_item_multiplier": 0.0})
        assert pen_with > pen_without


# ── total_penalty ───────────────────────────────────────────────────────────


class TestTotalPenalty:
    def test_zero_boxes(self, make_result):
        result = make_result(boxes=[])
        tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        assert total_penalty(result, tags) == 0.0

    def test_is_average_of_box_penalties(self, make_item, make_box, make_result):
        """total_penalty = avg(box_penalties), no fairness term."""
        items = [make_item(id=1, price=500), make_item(id=2, price=500)]
        box1 = make_box(name="a@test", allocations={1: 1})
        box2 = make_box(name="b@test", allocations={2: 1})
        result = make_result(items=items, boxes=[box1, box2])
        tags = compute_available_tags(result)
        pen = total_penalty(result, tags)
        bp1 = box_penalty(box1, result, tags)
        bp2 = box_penalty(box2, result, tags)
        avg_bp = (bp1 + bp2) / 2
        assert abs(pen - avg_bp) < 0.01

    def test_single_box(self, make_item, make_box, make_result):
        """Single box → total = box penalty."""
        item = make_item(id=1, price=500)
        box = make_box(allocations={1: 1})
        result = make_result(items=[item], boxes=[box])
        tags = compute_available_tags(result)
        pen = total_penalty(result, tags)
        bp = box_penalty(box, result, tags)
        assert abs(pen - bp) < 0.01


# ── params override ─────────────────────────────────────────────────────────


class TestParamsOverride:
    def test_value_penalty_params_override(self):
        """Params dict should override config defaults."""
        # Default sweet spot is 114-117
        assert value_penalty(115.0) == 0.0
        # Override to make 115 outside sweet spot
        params = {"value_sweet_from": 116, "value_sweet_to": 118}
        pen = value_penalty(115.0, params=params)
        assert pen > 0.0
