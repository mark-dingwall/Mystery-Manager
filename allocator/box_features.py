"""
Pure box-feature extraction.

Relocated from scripts/extract_features.py so that feature extraction can be
imported without a DB: that module imports `compare`, which imports
`allocator.db`, which loads queries.json at import time.

Hard constraint: no DB imports, no import-time side effects. `allocator.config`
imports are expected — a feature builder consumes the import-time-frozen config when called.
"""

from allocator.config import (
    BOX_TIERS,
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    GROUP_ALLOWANCES,
)
from allocator.strategies._helpers import _effective_species
from allocator.strategies._scoring import _resolve_item_allowance_from_lookup


def extract_box_features(
    box_name: str,
    allocations: dict[int, int],
    item_lookup: dict[int, dict],
    tier: str,
    available_tags: dict[str, set[str]],
    offer_id: int,
    source: str = "manual",
    preference: str | None = None,
) -> dict | None:
    """Extract raw features for a single box.

    Parallel to compare.compute_box_metrics but stores raw data for re-scoring.
    Returns feature dict or None if no items could be resolved.
    """
    box_price = BOX_TIERS[tier]["price"]

    total_value = 0
    resolved = 0

    # Per-item tracking: {item_id: [qty, price, item_allowance]}
    item_quantities_map: dict[int, list] = {}

    # Group tracking: {group_key: {item_id: capped_qty, ...}, degree, group_allowance}
    group_items: dict[str, dict[str, object]] = {}

    # Diversity tag counts
    tag_counts: dict[str, dict[str, int]] = {
        "sub_category": {}, "usage": {}, "colour": {}, "shape": {},
    }

    # Size tracking
    total_size_points = 0

    pref_violations = 0

    for item_id, qty in allocations.items():
        if qty <= 0 or item_id not in item_lookup:
            continue
        info = item_lookup[item_id]
        resolved += 1

        price = info["price"]
        total_value += price * qty

        # Per-item allowance
        item_allowance = _resolve_item_allowance_from_lookup(info, tier)
        item_quantities_map[item_id] = [qty, price, item_allowance]

        # Size points
        item_size = info.get("size", 1) or 1
        total_size_points += item_size * qty

        # Group tracking for group_concentration
        fg = info.get("fungible_group")
        if fg:
            key = fg
            degree = info.get("fungible_degree", 1.0)
        else:
            key = f"__item_{item_id}"
            degree = 1.0

        capped_qty = min(qty, item_allowance)

        if key not in group_items:
            group_items[key] = {"load": 0, "raw_load": 0, "degree": degree}
        group_items[key]["load"] += capped_qty
        group_items[key]["raw_load"] += qty

        # Diversity tags
        for dim, attr in [("sub_category", "sub_category"), ("usage", "usage"),
                          ("colour", "colour"), ("shape", "shape")]:
            tag = info.get(attr)
            if tag:
                tag_counts[dim][tag] = tag_counts[dim].get(tag, 0) + qty

        # Preference compliance
        if preference == "fruit_only" and info["category_id"] == CATEGORY_VEGETABLES:
            pref_violations += 1
        elif preference == "veg_only" and info["category_id"] == CATEGORY_FRUIT:
            pref_violations += 1

    if resolved == 0:
        return None

    value_pct = total_value / box_price * 100 if box_price > 0 else 0.0

    # Max value share
    max_value_share = 0.0
    if total_value > 0:
        for qty, price, _allow in item_quantities_map.values():
            share = price * qty / total_value
            if share > max_value_share:
                max_value_share = share

    # Build item_quantities list
    item_quantities = list(item_quantities_map.values())

    # Build group_totals: [group_load, degree, group_allowance] only for groups
    # that have explicit allowances
    group_totals = []
    for key, gdata in group_items.items():
        if key in GROUP_ALLOWANCES:
            ga = GROUP_ALLOWANCES[key].get(tier, gdata["load"])
            group_totals.append([gdata["load"], gdata["degree"], ga])

    # Keyed group loads — additional fields, never replacements. group_totals
    # stays exactly as rescore_box() consumes it (capped, positional, key-less).
    # Both dicts are filtered to GROUP_ALLOWANCES' key space, which excludes the
    # synthetic __item_{id} keys assigned to ungrouped items.
    raw_group_totals = {
        key: gdata["raw_load"]
        for key, gdata in group_items.items()
        if key in GROUP_ALLOWANCES
    }
    capped_group_totals = {
        key: gdata["load"]
        for key, gdata in group_items.items()
        if key in GROUP_ALLOWANCES
    }

    # Per-category value share, denominated in total_value exactly as specified.
    # Both keys always present; both 0.0 when the box has no value.
    # flatten() emits only the fruit column, which is sound *because* the pair
    # sums to 1 — and that holds only while every resolved item is fruit or veg.
    # An item in a third category would sit in the denominator while entering
    # neither numerator, quietly breaking the sum and leaving real signal in a
    # column that is never emitted. So refuse the box rather than encode it.
    fruit_value = 0
    veg_value = 0
    offending_items: list[tuple[int, object]] = []
    for item_id, qty in allocations.items():
        if qty <= 0 or item_id not in item_lookup:
            continue
        info = item_lookup[item_id]
        if info["category_id"] == CATEGORY_FRUIT:
            fruit_value += info["price"] * qty
        elif info["category_id"] == CATEGORY_VEGETABLES:
            veg_value += info["price"] * qty
        else:
            offending_items.append((item_id, info["category_id"]))
    if offending_items:
        offenders = ", ".join(
            f"item {item_id} category {category_id}"
            for item_id, category_id in offending_items
        )
        raise ValueError(
            f"Box {box_name!r} (offer {offer_id}) has resolved positive-quantity "
            f"{offenders}; category is neither CATEGORY_FRUIT ({CATEGORY_FRUIT}) "
            f"nor CATEGORY_VEGETABLES ({CATEGORY_VEGETABLES}). category_value_share "
            "is a two-key contract whose keys must sum to 1; a third category "
            "needs a third column and a re-derived multiple-testing family. "
            "Add the category to scoring_config.json and rerun, or exclude the item."
        )

    category_value_share = {"fruit": 0.0, "veg": 0.0}
    if total_value > 0:
        category_value_share = {
            "fruit": round(fruit_value / total_value, 6),
            "veg": round(veg_value / total_value, 6),
        }

    # Build dim_ratios and dim_available
    dim_ratios = {}
    dim_available = {}
    for dim in ["sub_category", "usage", "colour", "shape"]:
        n_avail = len(available_tags.get(dim, set()))
        dim_available[dim] = n_avail
        dim_counts = tag_counts.get(dim, {})
        if n_avail > 0 and dim_counts:
            eff = _effective_species(dim_counts)
            dim_ratios[dim] = eff / n_avail
        elif n_avail == 0:
            dim_ratios[dim] = 1.0
        else:
            dim_ratios[dim] = 0.0

    return {
        "offer_id": offer_id,
        "box_name": box_name,
        "tier": tier,
        "source": source,
        "value_pct": round(value_pct, 4),
        "item_quantities": item_quantities,
        "group_totals": [list(v) for v in group_totals],
        "max_value_share": round(max_value_share, 6),
        "total_size_points": total_size_points,
        "dim_ratios": {k: round(v, 6) for k, v in dim_ratios.items()},
        "dim_available": dim_available,
        "raw_group_totals": raw_group_totals,
        "capped_group_totals": capped_group_totals,
        "raw_tag_counts": {dim: dict(counts) for dim, counts in tag_counts.items()},
        "category_value_share": category_value_share,
        "pref_violations": pref_violations,
    }
