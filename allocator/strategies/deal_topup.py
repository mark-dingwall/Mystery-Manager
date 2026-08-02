# STATUS: baseline — regression benchmark only; not the production direction (see CLAUDE.md § Project Direction). Do not extend.
"""
Deal + Top-up allocation strategy.

Two phases:
1. Deal round — deal qty=1 of each item to every eligible box (sampler style).
   High-degree fungible groups get slot-filled first with qty > 1.
2. Top-up round — fill under-target boxes with new items, then qty bumps.
"""

import logging

from allocator.config import (
    CHEAP_ITEM_THRESHOLD,
    GROUP_ALLOWANCES,
    MAX_SLOT_QTY,
    SLOT_DEGREE_THRESHOLD,
    TOPUP_MAX_PASSES,
    VALUE_CEILING_PCT,
)
from allocator.models import AllocationResult, Item, MysteryBox
from allocator.scorer import prioritize_items_for_deal, score_topup_candidate
from allocator.strategies._helpers import (
    _item_allowance,
    box_fungible_groups,
    would_exceed_ceiling,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(result: AllocationResult) -> None:
    """Deal + Top-up strategy: deal items then fill to target."""
    _deal_round(result)
    _topup_round(result)


# ---------------------------------------------------------------------------
# Phase 1: Deal round
# ---------------------------------------------------------------------------

def _deal_round(result: AllocationResult) -> None:
    """
    Two sub-phases:
    1. Slot-fill high-degree fungible groups (degree >= SLOT_DEGREE_THRESHOLD)
       with qty > 1, using varieties as drop-in replacements.
    2. Card-deal non-fungible + low-degree items at qty=1 for variety.
    """
    num_boxes = len(result.boxes)
    if num_boxes == 0:
        return

    _deal_slot_fill(result, num_boxes)
    _deal_card_deal(result, num_boxes)

    deal_stats = [
        sum(1 for q in box.allocations.values() if q > 0)
        for box in result.boxes
    ]
    logger.info(
        f"Deal round complete: items/box = "
        f"{min(deal_stats, default=0)}-{max(deal_stats, default=0)}"
    )


def _deal_slot_fill(result: AllocationResult, num_boxes: int) -> None:
    """
    Sub-phase 1: Fill high-degree fungible group slots with qty > 1.

    For each high-degree group, distribute items across boxes so each box
    gets a useful household quantity (e.g. 4 tomatoes, 6 bananas). When one
    variety runs out, the next variety subs in seamlessly.
    """
    # Collect high-degree groups and their items
    high_degree_groups: dict[str, list[Item]] = {}
    group_degrees: dict[str, float] = {}
    for item in result.items.values():
        if (
            item.fungible_group
            and item.fungible_degree >= SLOT_DEGREE_THRESHOLD
            and result.remaining_overage(item.id) > 0
        ):
            high_degree_groups.setdefault(item.fungible_group, []).append(item)
            group_degrees[item.fungible_group] = item.fungible_degree

    if not high_degree_groups:
        return

    # Sort groups by average price descending (fill value faster)
    def avg_price(group_name: str) -> float:
        items = high_degree_groups[group_name]
        return sum(i.price for i in items) / len(items)

    sorted_groups = sorted(high_degree_groups.keys(), key=avg_price, reverse=True)

    start_idx = 0  # Rotates across groups for fairness

    for group_name in sorted_groups:
        group_items = high_degree_groups[group_name]

        # Total overage across all items in this group
        total_group_overage = sum(
            result.remaining_overage(i.id) for i in group_items
        )
        if total_group_overage <= 0:
            continue

        group_avg_price = avg_price(group_name)

        # Count eligible boxes (not excluded from all items in group, not at target)
        eligible_indices = []
        for i in range(num_boxes):
            bi = (start_idx + i) % num_boxes
            box = result.boxes[bi]
            if result.box_value(box) >= box.target_value:
                continue
            # Box is eligible if at least one item in group is not excluded
            if any(not box.is_excluded(item) for item in group_items):
                eligible_indices.append(bi)

        if not eligible_indices:
            start_idx = (start_idx + 1) % num_boxes
            continue

        fair_share = max(1, total_group_overage // len(eligible_indices))

        for bi in eligible_indices:
            box = result.boxes[bi]

            # Recalculate remaining budget for this box
            remaining_budget = box.target_value - result.box_value(box)
            if remaining_budget <= 0:
                continue

            max_affordable = int(remaining_budget // max(group_avg_price, 1))
            max_slot = MAX_SLOT_QTY.get(box.tier, 4)

            # Recalculate fair share based on current remaining overage
            current_group_overage = sum(
                result.remaining_overage(i.id) for i in group_items
            )
            if current_group_overage <= 0:
                break

            slot_qty = min(fair_share, max_affordable, max_slot, current_group_overage)
            if slot_qty <= 0:
                continue

            # Fill the slot with available varieties (highest overage first)
            varieties = sorted(
                group_items,
                key=lambda i: result.remaining_overage(i.id),
                reverse=True,
            )

            filled = 0
            for variety in varieties:
                if filled >= slot_qty:
                    break
                if box.is_excluded(variety):
                    continue
                avail = result.remaining_overage(variety.id)
                if avail <= 0:
                    continue
                # Check ceiling for this chunk
                give = min(avail, slot_qty - filled)
                # Trim to stay under ceiling
                while give > 0 and would_exceed_ceiling(box, variety, give, result):
                    give -= 1
                if give <= 0:
                    continue
                box.allocations[variety.id] = box.allocations.get(variety.id, 0) + give
                filled += give

        start_idx = (start_idx + 1) % num_boxes

    slot_stats = {}
    for box in result.boxes:
        for aid, aq in box.allocations.items():
            if aq > 0 and aid in result.items:
                item = result.items[aid]
                if item.fungible_group and item.fungible_degree >= SLOT_DEGREE_THRESHOLD:
                    slot_stats.setdefault(item.fungible_group, []).append(aq)

    for group, qtys in slot_stats.items():
        logger.debug(
            f"  Slot {group}: avg qty={sum(qtys)/len(qtys):.1f}, "
            f"range={min(qtys)}-{max(qtys)}"
        )


def _deal_card_deal(result: AllocationResult, num_boxes: int) -> None:
    """
    Sub-phase 2: Deal non-fungible + low-degree items at qty=1.

    Same as the original deal logic: card-deal style, one per fungible group
    per box, rotating start index for fairness.
    """
    prioritized = prioritize_items_for_deal(result.items, result)

    start_idx = 0

    for item in prioritized:
        remaining = result.remaining_overage(item.id)
        if remaining <= 0:
            continue

        for attempt in range(num_boxes):
            if remaining <= 0:
                break

            bi = (start_idx + attempt) % num_boxes
            box = result.boxes[bi]

            if box.is_excluded(item):
                continue

            if result.box_value(box) >= box.target_value:
                continue

            # For low-degree fungible items, skip if per-item qty >= item allowance
            # or group qty >= group allowance
            if item.fungible_group:
                item_allow = _item_allowance(item, box.tier)
                current_item_qty = box.allocations.get(item.id, 0)
                if current_item_qty >= item_allow:
                    continue
                if item.fungible_group in GROUP_ALLOWANCES:
                    group_allow = GROUP_ALLOWANCES[item.fungible_group].get(box.tier, 1)
                    current_group_qty = sum(
                        aq for aid, aq in box.allocations.items()
                        if aq > 0 and aid in result.items
                        and result.items[aid].fungible_group == item.fungible_group
                    )
                    if current_group_qty >= group_allow:
                        continue

            remaining = result.remaining_overage(item.id)
            if remaining <= 0:
                break

            if would_exceed_ceiling(box, item, 1, result):
                continue

            box.allocations[item.id] = box.allocations.get(item.id, 0) + 1
            remaining = result.remaining_overage(item.id)

        start_idx = (start_idx + 1) % num_boxes


# ---------------------------------------------------------------------------
# Phase 2: Top-up round
# ---------------------------------------------------------------------------

def _topup_round(result: AllocationResult) -> None:
    """
    For boxes still below target value, add items to reach target.

    Strategy (in order of preference):
    a) Add new items (not yet in box) at qty=1, scored by score_topup_candidate
    b) Bump cheap items (price <= CHEAP_ITEM_THRESHOLD) from qty=1 to qty=2
    c) Last resort: bump any existing item by +1
    """
    for pass_num in range(TOPUP_MAX_PASSES):
        # Sort boxes by deficit descending
        boxes_with_deficit = []
        for box in result.boxes:
            value = result.box_value(box)
            deficit = box.target_value - value
            if deficit > 0:
                boxes_with_deficit.append((deficit, box))

        if not boxes_with_deficit:
            break

        boxes_with_deficit.sort(key=lambda x: x[0], reverse=True)
        made_progress = False

        for _deficit, box in boxes_with_deficit:
            current_value = result.box_value(box)
            if current_value >= box.target_value:
                continue

            # (a) Try adding a new item at qty=1
            best_score = float("-inf")
            best_item = None

            for item in result.items.values():
                if result.remaining_overage(item.id) <= 0:
                    continue
                s = score_topup_candidate(item, 1, box, result)
                if s > best_score:
                    best_score = s
                    best_item = item

            if best_item is not None and best_score > float("-inf"):
                box.allocations[best_item.id] = box.allocations.get(best_item.id, 0) + 1
                made_progress = True
                continue

            # (b) Bump a cheap existing item from qty=1 to qty=2
            bumped = False
            for alloc_id, alloc_qty in list(box.allocations.items()):
                if alloc_qty != 1 or alloc_id not in result.items:
                    continue
                item = result.items[alloc_id]
                if item.price > CHEAP_ITEM_THRESHOLD:
                    continue
                if result.remaining_overage(alloc_id) < 1:
                    continue
                if would_exceed_ceiling(box, item, 1, result):
                    continue
                box.allocations[alloc_id] = 2
                bumped = True
                made_progress = True
                break

            if bumped:
                continue

            # (c) Last resort: bump any existing item by +1
            for alloc_id, alloc_qty in list(box.allocations.items()):
                if alloc_id not in result.items:
                    continue
                item = result.items[alloc_id]
                if result.remaining_overage(alloc_id) < 1:
                    continue
                if would_exceed_ceiling(box, item, 1, result):
                    continue
                box.allocations[alloc_id] = alloc_qty + 1
                made_progress = True
                break

        if not made_progress:
            break

    logger.info(f"Top-up completed in {pass_num + 1} passes")
