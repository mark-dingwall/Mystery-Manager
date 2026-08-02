"""Tests for allocator/scorer.py — prioritize_items_for_deal, score_topup_candidate."""

import pytest

from allocator.scorer import prioritize_items_for_deal, score_topup_candidate


# ── prioritize_items_for_deal ───────────────────────────────────────────────


class TestPrioritizeItemsForDeal:
    def test_non_fungible_before_fungible(self, make_item, make_result):
        items = [
            make_item(id=1, name="Broccoli", price=500, fungible_group=None, overage=3),
            make_item(id=2, name="Apples - Gala", price=400, fungible_group="apple",
                      fungible_degree=0.3, overage=3),
        ]
        result = make_result(items=items)
        ordered = prioritize_items_for_deal(result.items, result)
        assert ordered[0].id == 1  # non-fungible first

    def test_sorted_by_price_descending(self, make_item, make_result):
        items = [
            make_item(id=1, name="Cheap", price=100, fungible_group=None, overage=3),
            make_item(id=2, name="Expensive", price=900, fungible_group=None, overage=3),
            make_item(id=3, name="Middle", price=500, fungible_group=None, overage=3),
        ]
        result = make_result(items=items)
        ordered = prioritize_items_for_deal(result.items, result)
        prices = [i.price for i in ordered]
        assert prices == [900, 500, 100]

    def test_zero_overage_excluded(self, make_item, make_result):
        items = [
            make_item(id=1, price=500, overage=0),
            make_item(id=2, price=300, overage=3),
        ]
        result = make_result(items=items)
        ordered = prioritize_items_for_deal(result.items, result)
        assert len(ordered) == 1
        assert ordered[0].id == 2

    def test_high_degree_fungible_excluded(self, make_item, make_result):
        """Degree >= SLOT_DEGREE_THRESHOLD (0.7) → excluded from card deal."""
        items = [
            make_item(id=1, fungible_group="banana", fungible_degree=1.0, overage=3),
            make_item(id=2, fungible_group=None, overage=3),
        ]
        result = make_result(items=items)
        ordered = prioritize_items_for_deal(result.items, result)
        assert len(ordered) == 1
        assert ordered[0].id == 2

    def test_low_degree_fungible_included(self, make_item, make_result):
        """Degree < SLOT_DEGREE_THRESHOLD → included in card deal."""
        items = [
            make_item(id=1, fungible_group="apple", fungible_degree=0.3, overage=3),
        ]
        result = make_result(items=items)
        ordered = prioritize_items_for_deal(result.items, result)
        assert len(ordered) == 1

    def test_already_allocated_reduces_remaining(self, make_item, make_box, make_result):
        item = make_item(id=1, price=500, overage=2)
        box = make_box(allocations={1: 2})
        result = make_result(items=[item], boxes=[box])
        ordered = prioritize_items_for_deal(result.items, result)
        assert len(ordered) == 0


# ── score_topup_candidate ──────────────────────────────────────────────────


class TestScoreTopupCandidate:
    def test_excluded_item_neg_inf(self, make_item, make_box, make_result):
        from allocator.models import ExclusionRule
        item = make_item(id=1, name="Bananas", overage=5)
        rule = ExclusionRule(pattern="banana", source="note")
        box = make_box(exclusions=[rule])
        result = make_result(items=[item], boxes=[box])
        assert score_topup_candidate(item, 1, box, result) == float("-inf")

    def test_no_overage_neg_inf(self, make_item, make_box, make_result):
        item = make_item(id=1, overage=0)
        box = make_box()
        result = make_result(items=[item], boxes=[box])
        assert score_topup_candidate(item, 1, box, result) == float("-inf")

    def test_exceeds_ceiling_neg_inf(self, make_item, make_box, make_result):
        from allocator.config import BOX_TIERS, VALUE_CEILING_PCT
        ceiling = int(VALUE_CEILING_PCT * BOX_TIERS["small"]["price"])
        item = make_item(id=1, price=ceiling + 100, overage=5)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        assert score_topup_candidate(item, 1, box, result) == float("-inf")

    def test_hard_fungible_conflict_neg_inf(self, make_item, make_box, make_result):
        """Adding would exceed 2x per-item allowance → -inf."""
        # snack_piece small = 2, ceiling = 2*2 = 4
        item = make_item(id=1, fungible_group="banana", fungible_degree=1.0, overage=10)
        box = make_box(allocations={1: 4})  # at 2x per-item allowance
        result = make_result(items=[item], boxes=[box])
        # Adding the SAME item would exceed 2x per-item allowance (4+1 > 4)
        assert score_topup_candidate(item, 1, box, result) == float("-inf")

    def test_new_item_bonus(self, make_item, make_box, make_result):
        """New item (not in box) should score higher than existing item."""
        item = make_item(id=1, price=300, overage=10)
        empty_box = make_box(name="empty@test")
        has_item_box = make_box(name="has@test", allocations={1: 1})
        result = make_result(items=[item], boxes=[empty_box, has_item_box])
        new_score = score_topup_candidate(item, 1, empty_box, result)
        existing_score = score_topup_candidate(item, 1, has_item_box, result)
        assert new_score > existing_score

    def test_valid_candidate_positive_score(self, make_item, make_box, make_result):
        item = make_item(id=1, price=300, overage=5, fungible_group=None)
        box = make_box(tier="small")
        result = make_result(items=[item], boxes=[box])
        score = score_topup_candidate(item, 1, box, result)
        assert score > 0.0
