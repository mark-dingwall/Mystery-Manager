"""
Scoring functions for the variety-first allocation algorithm.

Two entry points:
- prioritize_items_for_deal(): sort items for the deal phase
- score_topup_candidate(): score an (item, qty) pair for the top-up phase
"""

from allocator.config import (
    BOX_TIERS,
    DIVERSITY_WEIGHTS,
    FUNGIBLE_NEUTRAL_SCORE,
    FUNGIBLE_NEW_GROUP_BONUS,
    GROUP_ALLOWANCES,
    SCORING_WEIGHTS,
    SLOT_DEGREE_THRESHOLD,
    VALUE_CEILING_PCT,
)
from allocator.models import AllocationResult, Item, MysteryBox
from allocator.strategies._helpers import _item_allowance

import logging
logger = logging.getLogger(__name__)


def prioritize_items_for_deal(
    items: dict[int, Item],
    result: AllocationResult,
) -> list[Item]:
    """
    Return items sorted in deal order for sub-phase 2 (card-deal at qty=1).

    Only includes non-fungible items and low-degree fungible groups
    (degree < SLOT_DEGREE_THRESHOLD). High-degree groups are handled
    separately by the slot-fill sub-phase.

    Priority tiers:
    1. Non-fungible items (unique items with no substitutes)
    2. One representative per low-degree fungible group

    Within each tier, secondary sort by price descending.
    """
    non_fungible: list[Item] = []
    fungible_groups: dict[str, list[Item]] = {}

    for item in items.values():
        if result.remaining_overage(item.id) <= 0:
            continue
        if item.fungible_group is None:
            non_fungible.append(item)
        elif item.fungible_degree < SLOT_DEGREE_THRESHOLD:
            # Only low-degree groups go through card-deal
            fungible_groups.setdefault(item.fungible_group, []).append(item)

    non_fungible.sort(key=lambda x: x.price, reverse=True)

    fungible_reps: list[Item] = []
    for group_items in fungible_groups.values():
        group_items.sort(key=lambda x: (x.overage, x.price), reverse=True)
        fungible_reps.extend(group_items)

    return non_fungible + fungible_reps


def score_topup_candidate(
    item: Item,
    qty: int,
    box: MysteryBox,
    result: AllocationResult,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Score adding `qty` of `item` to `box` during the top-up phase.

    Higher score = better fit. Returns -inf for hard constraint violations.
    """
    if weights is None:
        weights = SCORING_WEIGHTS

    # --- Hard constraints ---
    if box.is_excluded(item):
        return float("-inf")

    if result.remaining_overage(item.id) < qty:
        return float("-inf")

    current_value = result.box_value(box)
    add_value = item.price * qty
    new_value = current_value + add_value
    target = box.target_value

    # Would exceed ceiling
    if new_value > VALUE_CEILING_PCT * BOX_TIERS[box.tier]["price"]:
        return float("-inf")

    # --- new_item_bonus: strongly prefer items not yet in this box ---
    already_in_box = box.allocations.get(item.id, 0) > 0
    new_item_bonus = 0.0 if already_in_box else 1.0

    # --- fungible_spread: qty-based penalty for same-group items ---
    fungible_spread = 0.0
    item_allow = _item_allowance(item, box.tier)
    current_item_qty = box.allocations.get(item.id, 0)

    # Hard block: per-item qty would exceed 2x item allowance
    if current_item_qty + qty > 2 * item_allow:
        return float("-inf")

    # Hard block: group total would exceed 2x group allowance
    if item.fungible_group and item.fungible_group in GROUP_ALLOWANCES:
        group_allow = GROUP_ALLOWANCES[item.fungible_group].get(box.tier, 1)
        current_group_qty = sum(
            aq for aid, aq in box.allocations.items()
            if aq > 0 and aid in result.items
            and result.items[aid].fungible_group == item.fungible_group
        )
        if current_group_qty + qty > 2 * group_allow:
            return float("-inf")

    if item.fungible_group:
        current_group_qty = sum(
            aq for aid, aq in box.allocations.items()
            if aq > 0 and aid in result.items
            and result.items[aid].fungible_group == item.fungible_group
        )
        if current_group_qty > 0:
            # Same-item penalty term: penalise exceeding per-item allowance
            item_excess = max(0, current_item_qty + qty - item_allow)
            fungible_spread = -item.fungible_degree * item_excess
        else:
            fungible_spread = FUNGIBLE_NEW_GROUP_BONUS
    else:
        fungible_spread = FUNGIBLE_NEUTRAL_SCORE

    # --- diversity: reward items that improve effective species diversity ---
    # Accumulate qty-weighted tag counts per dimension
    existing_tag_counts: dict[str, dict[str, int]] = {
        "sub_category": {}, "usage": {}, "colour": {}, "shape": {},
    }
    _dim_attrs = {
        "sub_category": "sub_category",
        "usage": "usage_type",
        "colour": "colour",
        "shape": "shape",
    }
    for alloc_id, alloc_qty in box.allocations.items():
        if alloc_id in result.items and alloc_qty > 0:
            a = result.items[alloc_id]
            for dim, attr in _dim_attrs.items():
                tag = getattr(a, attr, "")
                if tag:
                    existing_tag_counts[dim][tag] = existing_tag_counts[dim].get(tag, 0) + alloc_qty

    diversity = 0.0
    for dim, weight in DIVERSITY_WEIGHTS.items():
        tag = getattr(item, _dim_attrs[dim], "")
        if not tag:
            continue
        tc = existing_tag_counts[dim]
        if tag not in tc:
            # New tag: full credit
            diversity += weight * 1.0
        else:
            # Existing tag: diminishing credit based on effective species gain
            total = sum(tc.values())
            if total > 0:
                hhi_before = sum((q / total) ** 2 for q in tc.values())
                new_total = total + qty
                hhi_after = sum(
                    ((q + (qty if t == tag else 0)) / new_total) ** 2
                    for t, q in tc.items()
                )
                eff_before = 1.0 / hhi_before
                eff_after = 1.0 / hhi_after
                # Normalise delta so it's in [~-1, 1] range like the binary version
                n_tags = len(tc)
                if n_tags > 0:
                    diversity += weight * (eff_after - eff_before) / n_tags

    # --- value_progress: gentle nudge toward target ---
    if target > 0:
        value_progress = add_value / target
    else:
        value_progress = 0.0

    return (
        weights.get("new_item_bonus", 5.0) * new_item_bonus
        + weights.get("fungible_spread", 3.0) * fungible_spread
        + weights.get("diversity", 2.0) * diversity
        + weights.get("value_progress", 1.0) * value_progress
    )
