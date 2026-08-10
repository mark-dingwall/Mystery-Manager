"""
Pure box-feature extraction.

Relocated from scripts/extract_features.py so that feature extraction can be
imported without a DB: that module imports `compare`, which imports
`allocator.db`, which loads queries.json at import time.

Hard constraint: no DB imports, no import-time side effects. `allocator.config`
imports are expected — a feature builder consumes the import-time-frozen config when called.
"""

import copy
import hashlib
import json
import statistics

from allocator.config import (
    BOX_TARGET_PCT,
    BOX_TIERS,
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    CLASSIFICATION_FALLBACK,
    GROUP_ALLOWANCES,
    ITEM_CLASSIFICATIONS,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.categorizer import DEFAULT_CLASSIFICATION
from allocator.strategies._helpers import _effective_species
from allocator.strategies import _scoring


_DIMENSIONS = ("sub_category", "usage", "colour", "shape")
_TIERS = ("small", "medium", "large")
FEATURE_SCHEMA_VERSION = 2


class UnsupportedCategoryError(ValueError):
    """A resolved item cannot fit the two-category feature schema."""


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
        item_allowance = _scoring._resolve_item_allowance_from_lookup(info, tier)
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
        raise UnsupportedCategoryError(
            f"Box {box_name!r} (offer {offer_id}) has resolved positive-quantity "
            f"{offenders}; category is neither CATEGORY_FRUIT ({CATEGORY_FRUIT}) "
            f"nor CATEGORY_VEGETABLES ({CATEGORY_VEGETABLES}). category_value_share "
            "is a two-key contract whose keys must sum to 1; a third category "
            "needs a third column and a re-derived multiple-testing family. "
            "Add the category to scoring_config.json and rerun, or exclude the item."
        )

    category_value_share = {"fruit": 0.0, "veg": 0.0}
    if total_value > 0:
        fruit_share = round(fruit_value / total_value, 6)
        category_value_share = {
            "fruit": fruit_share,
            "veg": 1.0 - fruit_share,
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


def tag_vocabulary() -> list[str]:
    """Return sorted, dimension-qualified tag names for the feature schema.

    The union includes configured classifications, category-specific fallbacks,
    and the categorizer's canonical default for unknown category IDs.
    """
    seen: set[str] = set()
    for _prefixes, *tags in ITEM_CLASSIFICATIONS.values():
        for dimension, tag in zip(_DIMENSIONS, tags):
            seen.add(f"{dimension}.{tag}")
    for tags in list(CLASSIFICATION_FALLBACK.values()) + [DEFAULT_CLASSIFICATION]:
        for dimension, tag in zip(_DIMENSIONS, tags):
            seen.add(f"{dimension}.{tag}")
    return sorted(seen)


def flatten(record: dict) -> dict[str, float]:
    """Map one feature record to globally name-sorted numeric matrix columns.

    Positional ``item_quantities`` and ``group_totals`` never become columns:
    they have variable length and embed scoring configuration parameters.
    """
    columns: dict[str, float] = {}

    if record["tier"] not in _TIERS:
        raise ValueError(f"unsupported tier for feature matrix: {record['tier']!r}")

    for tier in _TIERS:
        columns[f"value_pct_{tier}"] = (
            float(record["value_pct"]) if record["tier"] == tier else 0.0
        )

    for dimension in _DIMENSIONS:
        columns[f"dim_ratios.{dimension}"] = float(
            record["dim_ratios"].get(dimension, 0.0)
        )
        columns[f"dim_available.{dimension}"] = float(
            record["dim_available"].get(dimension, 0)
        )

    columns["max_value_share"] = float(record["max_value_share"])
    columns["total_size_points"] = float(record["total_size_points"])
    columns["pref_violations"] = float(record["pref_violations"])

    prices = [price for _quantity, price, _allowance in record["item_quantities"]]
    columns["n_unique_items"] = float(len(prices))
    columns["total_qty"] = float(
        sum(quantity for quantity, _price, _allowance in record["item_quantities"])
    )
    columns["price_mean"] = float(statistics.fmean(prices)) if prices else 0.0
    columns["price_sd"] = float(statistics.pstdev(prices)) if len(prices) > 1 else 0.0
    columns["price_max"] = float(max(prices)) if prices else 0.0

    columns["fruit_value_share"] = float(record["category_value_share"]["fruit"])

    for group in sorted(GROUP_ALLOWANCES):
        columns[f"capped_group_totals.{group}"] = float(
            record["capped_group_totals"].get(group, 0)
        )
        columns[f"raw_group_totals.{group}"] = float(
            record["raw_group_totals"].get(group, 0)
        )

    for entry in tag_vocabulary():
        dimension, tag = entry.split(".", 1)
        columns[f"raw_tag_counts.{entry}"] = float(
            record["raw_tag_counts"].get(dimension, {}).get(tag, 0)
        )

    return {name: columns[name] for name in sorted(columns)}


def stable_hash(obj: object) -> str:
    """Return the established 16-hex configuration digest for JSON-like data."""
    payload = json.dumps(obj, sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _ordered_rules(rules: dict) -> list[tuple[object, object]]:
    """Encode a first-match mapping without losing its insertion order."""
    return list(rules.items())


def _hash_inputs() -> dict:
    """Return the effective schema/scoring inputs used by feature extraction."""
    return {
        "BOX_TIERS": BOX_TIERS,
        "GROUP_ALLOWANCES": GROUP_ALLOWANCES,
        "ITEM_CLASSIFICATIONS": _ordered_rules(ITEM_CLASSIFICATIONS),
        "FUNGIBLE_GROUPS": _ordered_rules(_scoring.FUNGIBLE_GROUPS),
        "CLASSIFICATION_FALLBACK": CLASSIFICATION_FALLBACK,
        "QUANTITY_CLASSES": _scoring.QUANTITY_CLASSES,
        "QTY_CLASS_PRICE_THRESHOLDS": _scoring.QTY_CLASS_PRICE_THRESHOLDS,
        "VALUE_SWEET_FROM": VALUE_SWEET_FROM,
        "VALUE_SWEET_TO": VALUE_SWEET_TO,
        "VALUE_PENALTY_EXPONENT": VALUE_PENALTY_EXPONENT,
        "CATEGORY_FRUIT": CATEGORY_FRUIT,
        "CATEGORY_VEGETABLES": CATEGORY_VEGETABLES,
        "DEFAULT_CLASSIFICATION": DEFAULT_CLASSIFICATION,
    }


def config_hash() -> str:
    """Stamp over thirteen effective schema/scoring inputs.

    BOX_TARGET_PCT is represented by BOX_TIERS' derived target_value entries;
    config_snapshot() carries the frozen percentage directly.
    """
    return stable_hash(_hash_inputs())


def config_snapshot() -> dict:
    """Stamp for scenario files, compared for equality by the ordinal analyser.

    Enumerated rather than derived. The classification and fungible structures
    enter as digests to keep the written file readable.
    """
    return {
        "box_tiers": {
            tier: {"price": entry["price"], "target_value": entry["target_value"]}
            for tier, entry in BOX_TIERS.items()
        },
        "box_target_pct": BOX_TARGET_PCT,
        "category_fruit": CATEGORY_FRUIT,
        "category_vegetables": CATEGORY_VEGETABLES,
        "value_sweet_from": VALUE_SWEET_FROM,
        "value_sweet_to": VALUE_SWEET_TO,
        "value_penalty_exponent": VALUE_PENALTY_EXPONENT,
        "group_allowances": copy.deepcopy(GROUP_ALLOWANCES),
        "quantity_classes": copy.deepcopy(_scoring.QUANTITY_CLASSES),
        "qty_class_price_thresholds": copy.deepcopy(
            _scoring.QTY_CLASS_PRICE_THRESHOLDS
        ),
        "item_classifications_hash": stable_hash(_ordered_rules(ITEM_CLASSIFICATIONS)),
        "fungible_groups_hash": stable_hash(_ordered_rules(_scoring.FUNGIBLE_GROUPS)),
        "classification_fallback_hash": stable_hash(CLASSIFICATION_FALLBACK),
        "default_classification_hash": stable_hash(DEFAULT_CLASSIFICATION),
    }
