# STATUS: baseline — regression benchmark only; not the production direction (see CLAUDE.md § Project Direction). Do not extend.
"""
Discard-worst allocation strategy.

Subtractive approach:
1. Seed: greedy draft — boxes take turns picking the item that most improves
   their diversity/dupe profile (ignore ceiling, respect exclusion + hard
   fungible).
2. Trim: for each box over ceiling, remove worst-scored items. Then for boxes
   above target, remove worst items until at/just-above target.

Removal scoring uses penalty-delta: remove the item whose removal most
reduces the box's composite penalty (matching compare.py's metric). A soft
sole-provider diversity guard reduces the removal score for items that are
the only source of a diversity tag in the box, making them harder (but not
impossible) to remove.
"""

import logging

from allocator.config import (
    BOX_TIERS,
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    GROUP_ALLOWANCES,
    GROUP_CONCENTRATION_MULTIPLIER,
    GROUP_QTY_EXPONENT,
    SAME_ITEM_MULTIPLIER,
    VALUE_CEILING_PCT,
)
from allocator.models import AllocationResult, Item, MysteryBox
from allocator.strategies._helpers import (
    _item_allowance,
    assign_item,
    compute_available_tags,
    has_hard_fungible_conflict,
    remove_item,
)
from allocator.strategies._scoring import box_penalty

logger = logging.getLogger(__name__)


def run(result: AllocationResult) -> None:
    """Discard-worst: seed all items then trim to target."""
    _seed_phase(result)
    _trim_phase(result)


def _seed_score(
    item: Item,
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    avail_counts: dict[str, int],
    box_tag_counts: dict[str, dict[str, int]],
    box_groups: dict[str, tuple[int, float]],
) -> float:
    """
    Score adding an item to a box during the draft seed phase.

    Considers diversity gain and dupe penalty only (value is irrelevant since
    we'll trim later). Diversity gain uses diminishing returns: new tags get
    full credit, existing tags get reduced credit proportional to how adding
    this item would improve the effective species count. Tiebreak by price.
    """
    score = 0.0

    dim_attrs = {
        "sub_category": "sub_category",
        "usage": "usage_type",
        "colour": "colour",
        "shape": "shape",
    }
    for dim, weight in DIVERSITY_WEIGHTS.items():
        tag = getattr(item, dim_attrs[dim], "")
        n_avail = avail_counts.get(dim, 0)
        if not tag or n_avail <= 0:
            continue
        tc = box_tag_counts[dim]
        if tag not in tc:
            # New tag: full diversity credit
            score += weight / n_avail * DIVERSITY_PENALTY_MULTIPLIER
        else:
            # Existing tag: diminishing credit based on effective species gain
            total = sum(tc.values())
            if total > 0:
                hhi_before = sum((q / total) ** 2 for q in tc.values())
                # Simulate adding 1 unit to this tag
                new_total = total + 1
                hhi_after = sum(
                    ((q + (1 if t == tag else 0)) / new_total) ** 2
                    for t, q in tc.items()
                )
                # Delta in effective species (can be negative if concentrating)
                eff_before = 1.0 / hhi_before
                eff_after = 1.0 / hhi_after
                delta = (eff_after - eff_before) / n_avail
                score += weight * delta * DIVERSITY_PENALTY_MULTIPLIER

    # Same-item cost: penalise exceeding per-item allowance
    current_item_qty = box.allocations.get(item.id, 0)
    item_allow = _item_allowance(item, box.tier)
    item_excess = max(0, current_item_qty + 1 - item_allow)
    if item_excess > 0:
        score -= item_excess * item.price * SAME_ITEM_MULTIPLIER / 100.0

    # Group concentration cost: penalise excess above group allowance
    if item.fungible_group and item.fungible_group in box_groups:
        current_qty, degree = box_groups[item.fungible_group]
        if item.fungible_group in GROUP_ALLOWANCES:
            ga = GROUP_ALLOWANCES[item.fungible_group].get(box.tier, current_qty + 1)
            # Use min(qty, item_allowance) for group load contribution
            capped_new = min(1, item_allow)
            new_load = current_qty + capped_new  # approximate: existing load + new capped qty
            excess = max(0, new_load - ga)
            if excess > 0:
                score -= (excess ** GROUP_QTY_EXPONENT) * degree * GROUP_CONCENTRATION_MULTIPLIER

    # Tiebreak: prefer placing expensive items (harder to place later)
    score += item.price * 1e-8

    return score


def _seed_phase(result: AllocationResult) -> None:
    """
    Greedy draft: boxes take turns picking the best available item.

    Each box selects the item that most improves its diversity / minimises
    dupe penalties. Ceiling is ignored (the trim phase handles value).
    """
    num_boxes = len(result.boxes)
    if num_boxes == 0:
        return

    available_tags = compute_available_tags(result)
    avail_counts = {dim: len(tags) for dim, tags in available_tags.items()}

    # Cache per-box tag counts and fungible groups (updated incrementally)
    dim_attrs = {
        "sub_category": "sub_category",
        "usage": "usage_type",
        "colour": "colour",
        "shape": "shape",
    }
    all_box_tag_counts: list[dict[str, dict[str, int]]] = []
    all_box_groups: list[dict[str, tuple[int, float]]] = []
    for box in result.boxes:
        btc: dict[str, dict[str, int]] = {d: {} for d in DIVERSITY_WEIGHTS}
        bg: dict[str, tuple[int, float]] = {}
        all_box_tag_counts.append(btc)
        all_box_groups.append(bg)

    box_idx = 0
    any_assigned = True
    while any_assigned:
        any_assigned = False
        for turn in range(num_boxes):
            bi = (box_idx + turn) % num_boxes
            box = result.boxes[bi]
            btc = all_box_tag_counts[bi]
            bg = all_box_groups[bi]

            best_score = -float("inf")
            best_item = None
            for item in result.items.values():
                if result.remaining_overage(item.id) <= 0:
                    continue
                if box.is_excluded(item):
                    continue
                if has_hard_fungible_conflict(item, box, result):
                    continue
                s = _seed_score(item, box, result, available_tags,
                                avail_counts, btc, bg)
                if s > best_score:
                    best_score = s
                    best_item = item

            if best_item is not None:
                assign_item(best_item.id, 1, box)
                any_assigned = True
                # Update cached tag counts
                for dim, attr in dim_attrs.items():
                    tag = getattr(best_item, attr, "")
                    if tag:
                        btc[dim][tag] = btc[dim].get(tag, 0) + 1
                # Update cached fungible groups
                if best_item.fungible_group:
                    if best_item.fungible_group in bg:
                        prev_count, deg = bg[best_item.fungible_group]
                        bg[best_item.fungible_group] = (prev_count + 1, deg)
                    else:
                        bg[best_item.fungible_group] = (1, best_item.fungible_degree)

        box_idx = (box_idx + 1) % num_boxes

    logger.info("Seed phase complete")


def _score_for_removal(
    item: Item,
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    avail_counts: dict[str, int],
    pen_before: float,
) -> float:
    """
    Score an item for removal via penalty-delta + sole-provider guard.

    Returns pen_before - pen_after - guard (positive = removal helps).
    Temporarily removes 1 unit, computes new penalty, then restores.

    The sole-provider guard reduces the score when the item is the only one
    in the box covering a diversity tag, making it harder (but not impossible)
    to remove.
    """
    old_qty = box.allocations.get(item.id, 0)
    if old_qty <= 0:
        return 0.0

    # Temporarily remove
    if old_qty == 1:
        del box.allocations[item.id]
    else:
        box.allocations[item.id] = old_qty - 1

    pen_after = box_penalty(box, result, available_tags)

    # Restore
    box.allocations[item.id] = old_qty

    penalty_delta = pen_before - pen_after

    # Sole-provider diversity guard: if removing this item would eliminate
    # a diversity tag from the box entirely, apply a soft penalty to the
    # removal score. Only applies when qty == 1 (otherwise the tag survives).
    guard = 0.0
    if old_qty == 1:
        dim_attrs = {
            "sub_category": "sub_category",
            "usage": "usage_type",
            "colour": "colour",
            "shape": "shape",
        }
        for dim, weight in DIVERSITY_WEIGHTS.items():
            tag = getattr(item, dim_attrs[dim], "")
            if not tag:
                continue
            n_avail = avail_counts.get(dim, 0)
            if n_avail <= 0:
                continue
            # Check if any other item in the box also provides this tag
            sole = True
            for other_id, other_qty in box.allocations.items():
                if other_id == item.id or other_qty <= 0:
                    continue
                if other_id not in result.items:
                    continue
                if getattr(result.items[other_id], dim_attrs[dim], "") == tag:
                    sole = False
                    break
            if sole:
                guard += weight / n_avail * DIVERSITY_PENALTY_MULTIPLIER

    return penalty_delta - guard


def _trim_phase(result: AllocationResult) -> None:
    """Trim boxes: first to ceiling, then toward target."""
    available_tags = compute_available_tags(result)
    avail_counts = {dim: len(tags) for dim, tags in available_tags.items()}
    for box in result.boxes:
        _trim_to_ceiling(box, result, available_tags, avail_counts)
        _trim_to_target(box, result, available_tags, avail_counts)


def _trim_to_ceiling(
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    avail_counts: dict[str, int],
) -> None:
    """Remove items until box is at or below ceiling."""
    ceiling = VALUE_CEILING_PCT * BOX_TIERS[box.tier]["price"]

    while result.box_value(box) > ceiling:
        pen_before = box_penalty(box, result, available_tags)

        candidates = []
        for item_id, qty in list(box.allocations.items()):
            if qty <= 0 or item_id not in result.items:
                continue
            item = result.items[item_id]
            score = _score_for_removal(
                item, box, result, available_tags, avail_counts, pen_before,
            )
            candidates.append((score, item_id))

        if not candidates:
            break

        # Remove the item whose removal most reduces penalty
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, remove_id = candidates[0]
        remove_item(remove_id, 1, box)


def _trim_to_target(
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    avail_counts: dict[str, int],
) -> None:
    """
    Remove items to bring box value toward target.

    Stop when no removal improves the box penalty (penalty-delta <= 0).
    """
    while True:
        current = result.box_value(box)
        if current <= box.target_value:
            break

        pen_before = box_penalty(box, result, available_tags)

        candidates = []
        for item_id, qty in list(box.allocations.items()):
            if qty <= 0 or item_id not in result.items:
                continue
            item = result.items[item_id]
            score = _score_for_removal(
                item, box, result, available_tags, avail_counts, pen_before,
            )
            candidates.append((score, item_id))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, remove_id = candidates[0]
        if best_score <= 0:
            break
        remove_item(remove_id, 1, box)

    logger.debug(
        f"  {box.name}: trimmed to ${result.box_value(box)/100:.2f} "
        f"(target ${box.target_value/100:.2f})"
    )
