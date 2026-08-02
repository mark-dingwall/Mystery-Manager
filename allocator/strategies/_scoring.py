"""
Shared penalty functions matching compare.py's composite scoring.

These let strategies optimise the same objective they're measured on.
All constants are imported from config.py (single source of truth).
"""

import logging

from allocator.config import (
    BOX_TIERS,
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    DIVERSITY_PENALTY_MULTIPLIER,
    FUNGIBLE_GROUPS,
    GROUP_ALLOWANCES,
    GROUP_CONCENTRATION_MULTIPLIER,
    GROUP_QTY_EXPONENT,
    ITEM_CLASSIFICATIONS,
    MAX_VALUE_SHARE_MULTIPLIER,
    MAX_VALUE_SHARE_THRESHOLD,
    QTY_CLASS_PRICE_THRESHOLDS,
    QUANTITY_CLASSES,
    SAME_ITEM_MULTIPLIER,
    SIZE_FLOOR_MULTIPLIER,
    SIZE_FLOOR_TARGETS,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.models import AllocationResult, MysteryBox
from allocator.strategies._helpers import compute_diversity_score

logger = logging.getLogger(__name__)


def _p(params: dict | None, key: str, default):
    """Look up a param from override dict, falling back to default."""
    if params and key in params:
        return params[key]
    return default


def value_penalty(vp: float, params: dict | None = None) -> float:
    """
    Power-function value penalty for a box at *vp* % of box price.

    Symmetric: penalty = distance_from_sweet_spot ** exponent.
    Small deviations are gentle, large deviations are harsh.
    """
    sweet_from = _p(params, "value_sweet_from", VALUE_SWEET_FROM)
    sweet_to = _p(params, "value_sweet_to", VALUE_SWEET_TO)
    exponent = _p(params, "value_penalty_exponent", VALUE_PENALTY_EXPONENT)

    if sweet_from <= vp <= sweet_to:
        return 0.0
    if vp < sweet_from:
        x = sweet_from - vp
    else:
        x = vp - sweet_to
    return x ** exponent


# ---------------------------------------------------------------------------
# Item allowance resolution
# ---------------------------------------------------------------------------

def _resolve_item_allowance(item, tier: str, params: dict | None = None) -> int:
    """
    Resolve per-item allowance for same-item penalty.

    Lookup chain:
    1. Fungible group quantity_class → tier allowance
    2. Usage-based default from item_classifications
    3. Fallback = 1 (conservative)
    """
    qty_classes = _p(params, "quantity_classes", QUANTITY_CLASSES)

    # 1. Check fungible group quantity_class
    if item.fungible_group and item.fungible_group in FUNGIBLE_GROUPS:
        _degree, _prefixes, qty_class = FUNGIBLE_GROUPS[item.fungible_group]
        if qty_class in qty_classes:
            return qty_classes[qty_class].get(tier, 1)

    # 2. Usage-based default
    usage = getattr(item, "usage_type", "")
    price = getattr(item, "price", 0)
    qty_class = _usage_to_qty_class(usage, price)
    if qty_class in qty_classes:
        return qty_classes[qty_class].get(tier, 1)

    # 3. Fallback
    return 1


def _usage_to_qty_class(usage: str, price: int) -> str:
    """Map item usage to quantity class, with price-based refinement."""
    if usage == "snacking":
        return "snack_piece" if price < QTY_CLASS_PRICE_THRESHOLDS["snacking_max"] else "cooking_piece"
    if usage == "cooking":
        return "cooking_piece" if price < QTY_CLASS_PRICE_THRESHOLDS["cooking_max"] else "portioned"
    if usage == "salad":
        return "cooking_piece"
    if usage == "garnish":
        return "garnish"
    return "portioned"


def _resolve_item_allowance_from_lookup(
    item_info: dict, tier: str, params: dict | None = None,
) -> int:
    """Resolve per-item allowance from an item_lookup dict entry (for compare.py)."""
    qty_classes = _p(params, "quantity_classes", QUANTITY_CLASSES)

    fg = item_info.get("fungible_group")
    if fg and fg in FUNGIBLE_GROUPS:
        _degree, _prefixes, qty_class = FUNGIBLE_GROUPS[fg]
        if qty_class in qty_classes:
            return qty_classes[qty_class].get(tier, 1)

    usage = item_info.get("usage", "")
    price = item_info.get("price", 0)
    qty_class = _usage_to_qty_class(usage, price)
    if qty_class in qty_classes:
        return qty_classes[qty_class].get(tier, 1)

    return 1


# ---------------------------------------------------------------------------
# Penalty functions
# ---------------------------------------------------------------------------

def same_item_penalty_for_box(
    box: MysteryBox, result: AllocationResult, params: dict | None = None,
) -> float:
    """
    Per-item excess penalty for a box.

    For each item, penalty = max(0, qty - item_allowance) * item.price * multiplier.
    Returns the total penalty (already includes multiplier).
    """
    multiplier = _p(params, "same_item_multiplier", SAME_ITEM_MULTIPLIER)

    penalty = 0.0
    for item_id, qty in box.allocations.items():
        if qty <= 0 or item_id not in result.items:
            continue
        item = result.items[item_id]
        allowance = _resolve_item_allowance(item, box.tier, params)
        excess = max(0, qty - allowance)
        if excess > 0:
            penalty += excess * item.price * multiplier / 100.0  # normalise cents
    return penalty


def group_concentration_penalty_for_box(
    box: MysteryBox, result: AllocationResult, params: dict | None = None,
) -> float:
    """
    Group concentration penalty using min() to avoid double-counting with same-item.

    group_load = sum(min(qty_i, item_allowance_i) for i in group)
    penalty = max(0, group_load - group_allowance) ^ exponent * degree * multiplier

    Returns the raw penalty (already includes multiplier).
    """
    exponent = _p(params, "group_qty_exponent", GROUP_QTY_EXPONENT)
    multiplier = _p(params, "group_concentration_multiplier", GROUP_CONCENTRATION_MULTIPLIER)
    group_allowances = _p(params, "group_allowances", GROUP_ALLOWANCES)

    # Build {group_key: (group_load, degree)}
    groups: dict[str, tuple[float, float]] = {}
    for item_id, qty in box.allocations.items():
        if qty <= 0 or item_id not in result.items:
            continue
        item = result.items[item_id]
        if item.fungible_group:
            key = item.fungible_group
            degree = item.fungible_degree
        else:
            key = f"__item_{item_id}"
            degree = 1.0

        # Use min(qty, item_allowance) for group load
        allowance = _resolve_item_allowance(item, box.tier, params)
        capped_qty = min(qty, allowance)

        if key in groups:
            prev_load, prev_degree = groups[key]
            groups[key] = (prev_load + capped_qty, prev_degree)
        else:
            groups[key] = (capped_qty, degree)

    penalty = 0.0
    for key, (group_load, degree) in groups.items():
        # Look up explicit group allowance
        if key in group_allowances:
            ga = group_allowances[key].get(box.tier, group_load)
        else:
            # No explicit allowance → no group penalty (singletons/unmapped)
            continue

        excess = max(0, group_load - ga)
        if excess > 0:
            penalty += (excess ** exponent) * degree * multiplier
    return penalty


def max_value_share_penalty_for_box(
    box: MysteryBox, result: AllocationResult, params: dict | None = None,
) -> float:
    """
    Penalty when any single item exceeds a threshold share of box value.

    penalty = max(0, max_share - threshold) * multiplier
    """
    threshold = _p(params, "max_value_share_threshold", MAX_VALUE_SHARE_THRESHOLD)
    multiplier = _p(params, "max_value_share_multiplier", MAX_VALUE_SHARE_MULTIPLIER)

    total_value = result.box_value(box)
    if total_value <= 0:
        return 0.0

    max_share = 0.0
    for item_id, qty in box.allocations.items():
        if qty <= 0 or item_id not in result.items:
            continue
        share = result.items[item_id].price * qty / total_value
        if share > max_share:
            max_share = share

    excess = max(0.0, max_share - threshold)
    return excess * multiplier


def size_floor_penalty_for_box(
    box: MysteryBox, result: AllocationResult, params: dict | None = None,
) -> float:
    """
    Weak penalty when total size points fall below tier target.

    penalty = max(0, target - total_size) * multiplier
    """
    targets = _p(params, "size_floor_targets", SIZE_FLOOR_TARGETS)
    multiplier = _p(params, "size_floor_multiplier", SIZE_FLOOR_MULTIPLIER)

    target = targets.get(box.tier, 0)
    if target <= 0:
        return 0.0

    total_size = 0
    for item_id, qty in box.allocations.items():
        if qty <= 0 or item_id not in result.items:
            continue
        total_size += result.items[item_id].size * qty

    deficit = max(0, target - total_size)
    return deficit * multiplier


def box_penalty(
    box: MysteryBox,
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    params: dict | None = None,
) -> float:
    """
    Total penalty for a single box.

    Sum of: value + same_item + group_concentration + diversity +
            max_value_share + size_floor.
    """
    div_mult = _p(params, "diversity_penalty_multiplier", DIVERSITY_PENALTY_MULTIPLIER)

    value = result.box_value(box)
    box_price = BOX_TIERS[box.tier]["price"]
    vp = value / box_price * 100 if box_price > 0 else 0.0

    val_pen = value_penalty(vp, params)
    si_pen = same_item_penalty_for_box(box, result, params)
    gc_pen = group_concentration_penalty_for_box(box, result, params)
    div_score = compute_diversity_score(box, result, available_tags)
    div_pen = (1.0 - div_score) * div_mult
    mvs_pen = max_value_share_penalty_for_box(box, result, params)
    sf_pen = size_floor_penalty_for_box(box, result, params)

    return val_pen + si_pen + gc_pen + div_pen + mvs_pen + sf_pen


def total_penalty(
    result: AllocationResult,
    available_tags: dict[str, set[str]],
    params: dict | None = None,
) -> float:
    """
    Full composite penalty across all boxes (lower = better).

    avg(box_penalties). No fairness term (dropped in v1 rework).
    """
    n = len(result.boxes)
    if n == 0:
        return 0.0

    box_pens = [box_penalty(box, result, available_tags, params) for box in result.boxes]
    return sum(box_pens) / n
