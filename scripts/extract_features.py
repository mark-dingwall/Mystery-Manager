#!/usr/bin/env python3
"""
Extract precomputed box features for Optuna parameter tuning.

Reads historical CSVs + DB item data, computes raw box features (value_pct,
item_quantities, group_totals, dim_ratios, max_value_share, total_size_points,
pref_violations) for each manual box, then generates synthetic bad boxes per offer.

Output: diagnostics/tuning_features.json

Usage:
    python3 scripts/extract_features.py                        # all Tier A
    python3 scripts/extract_features.py --only-offers 85-106   # post-85 only
    python3 scripts/extract_features.py --no-synthetics        # manual only
"""

import json
import logging
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from allocator.box_parser import infer_box_tier
from allocator.config import (
    BOX_TIERS,
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    DONATION_IDENTIFIERS,
    FUNGIBLE_GROUPS,
    GROUP_ALLOWANCES,
    QUANTITY_CLASSES,
    SKIP_COLUMN_IDENTIFIERS,
    STAFF_IDENTIFIERS,
)
from allocator.strategies._scoring import _resolve_item_allowance_from_lookup
from compare import (
    build_item_lookup,
    compute_available_tags,
    load_historical_csv,
    load_summary,
    read_xlsx_pack_overrides,
    _build_offer_ids,
    _discover_cleaned_offer_ids,
    _offer_tier,
)
from allocator.db import fetch_mystery_box_buyers


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _effective_species(tag_counts: dict[str, int]) -> float:
    """Effective number of species (1/HHI) from tag->qty counts."""
    total = sum(tag_counts.values())
    if total == 0:
        return 0.0
    hhi = sum((q / total) ** 2 for q in tag_counts.values())
    return 1.0 / hhi


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
            group_items[key] = {"load": 0, "degree": degree}
        group_items[key]["load"] += capped_qty

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
        "pref_violations": pref_violations,
    }


# ---------------------------------------------------------------------------
# Synthetic box generation
# ---------------------------------------------------------------------------

def _items_by_value(item_lookup: dict[int, dict]) -> list[tuple[int, dict]]:
    """Return items sorted by price ascending."""
    return sorted(item_lookup.items(), key=lambda x: x[1]["price"])


def _fill_to_target(items: list[tuple[int, dict]], target_value: int,
                     rng: Random | None = None) -> dict[int, int]:
    """Greedily fill allocations to reach target_value."""
    allocs: dict[int, int] = {}
    current = 0
    if rng:
        items = list(items)
        rng.shuffle(items)
    for item_id, info in items:
        while current < target_value:
            current += info["price"]
            allocs[item_id] = allocs.get(item_id, 0) + 1
        if current >= target_value:
            break
    return allocs


def generate_synthetic_boxes(
    offer_id: int,
    item_lookup: dict[int, dict],
    available_tags: dict[str, set[str]],
) -> list[dict]:
    """Generate 5 synthetic bad boxes per tier for a single offer."""
    synthetics = []
    sorted_items = _items_by_value(item_lookup)
    if not sorted_items:
        return []

    rng = Random(offer_id)

    for tier in ["small", "medium", "large"]:
        target = BOX_TIERS[tier]["target_value"]

        # 1. Monoculture: cheapest item, fill to ~115% target
        cheapest_id, cheapest_info = sorted_items[0]
        mono_target = int(target * 1.15)
        mono_qty = max(1, mono_target // cheapest_info["price"])
        mono_allocs = {cheapest_id: mono_qty}
        f = extract_box_features(
            f"synth_mono_{tier}", mono_allocs, item_lookup, tier,
            available_tags, offer_id, source="synth_monoculture",
        )
        if f:
            synthetics.append(f)

        # 2. Random: uniform sample to target value
        random_allocs = _fill_to_target(sorted_items, int(target * 1.15), rng=rng)
        f = extract_box_features(
            f"synth_random_{tier}", random_allocs, item_lookup, tier,
            available_tags, offer_id, source="synth_random",
        )
        if f:
            synthetics.append(f)

        # 3. Over-fungible: max items from largest fungible groups
        fg_items: dict[str, list] = {}
        for iid, info in item_lookup.items():
            fg = info.get("fungible_group")
            if fg:
                fg_items.setdefault(fg, []).append((iid, info))
        overfg_allocs: dict[int, int] = {}
        overfg_value = 0
        for fg_name in sorted(fg_items, key=lambda g: len(fg_items[g]), reverse=True):
            for iid, info in fg_items[fg_name]:
                while overfg_value < int(target * 1.15):
                    overfg_allocs[iid] = overfg_allocs.get(iid, 0) + 1
                    overfg_value += info["price"]
                if overfg_value >= int(target * 1.15):
                    break
            if overfg_value >= int(target * 1.15):
                break
        if overfg_allocs:
            f = extract_box_features(
                f"synth_overfg_{tier}", overfg_allocs, item_lookup, tier,
                available_tags, offer_id, source="synth_over_fungible",
            )
            if f:
                synthetics.append(f)

        # 4. Value low: items to ~70% target
        low_allocs = _fill_to_target(sorted_items, int(target * 0.70))
        f = extract_box_features(
            f"synth_low_{tier}", low_allocs, item_lookup, tier,
            available_tags, offer_id, source="synth_value_low",
        )
        if f:
            synthetics.append(f)

        # 5. Value high: items to ~160% target
        high_allocs = _fill_to_target(sorted_items, int(target * 1.60))
        f = extract_box_features(
            f"synth_high_{tier}", high_allocs, item_lookup, tier,
            available_tags, offer_id, source="synth_value_high",
        )
        if f:
            synthetics.append(f)

    return synthetics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract box features for Optuna parameter tuning"
    )
    parser.add_argument("--only-offers", type=str, default=None,
                        help="Comma-separated offer IDs/ranges (e.g. '85-106')")
    parser.add_argument("--no-synthetics", action="store_true",
                        help="Skip synthetic bad box generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: diagnostics/tuning_features.json)")
    args = parser.parse_args()

    summary = load_summary()

    # Default to Tier A offers
    offer_ids = _build_offer_ids(summary, only_offers=args.only_offers)
    print(f"Processing {len(offer_ids)} offers...")

    all_features: list[dict] = []
    skipped = 0

    for offer_id in offer_ids:
        item_lookup = build_item_lookup(offer_id)
        if not item_lookup:
            print(f"  [SKIP] Offer {offer_id}: no items in DB")
            skipped += 1
            continue

        # Build manual item lookup with XLSX pack-price overrides
        pack_overrides = read_xlsx_pack_overrides(offer_id)
        manual_lookup = (
            build_item_lookup(offer_id, price_overrides=pack_overrides)
            if pack_overrides else item_lookup
        )

        box_names, hist_allocs = load_historical_csv(offer_id)
        if not box_names:
            print(f"  [SKIP] Offer {offer_id}: no historical CSV")
            skipped += 1
            continue

        # Filter non-customer boxes
        box_names = [bn for bn in box_names
                     if bn not in DONATION_IDENTIFIERS
                     and bn not in SKIP_COLUMN_IDENTIFIERS
                     and bn not in STAFF_IDENTIFIERS]

        # Load preferences
        buyers_db = fetch_mystery_box_buyers(offer_id)
        buyer_prefs = {}
        for buyer in buyers_db:
            email = buyer["user_email"]
            opt = buyer.get("selected_option") or ""
            if "no veg" in opt.lower():
                buyer_prefs[email] = "fruit_only"
            elif "no fruit" in opt.lower():
                buyer_prefs[email] = "veg_only"

        avail_tags = compute_available_tags(item_lookup)

        n_manual = 0
        for bn in box_names:
            tier = infer_box_tier(offer_id, bn, summary)
            if tier is None:
                continue

            box_allocs = {}
            for item_id, per_box in hist_allocs.items():
                qty = per_box.get(bn, 0)
                if qty > 0:
                    box_allocs[item_id] = int(qty)

            pref = buyer_prefs.get(bn)
            f = extract_box_features(
                bn, box_allocs, manual_lookup, tier, avail_tags,
                offer_id, source="manual", preference=pref,
            )
            if f:
                all_features.append(f)
                n_manual += 1

        # Synthetic bad boxes
        n_synth = 0
        if not args.no_synthetics:
            synths = generate_synthetic_boxes(offer_id, item_lookup, avail_tags)
            all_features.extend(synths)
            n_synth = len(synths)

        print(f"  Offer {offer_id}: {n_manual} manual, {n_synth} synthetic")

    # Write output
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "diagnostics" / "tuning_features.json"
    )
    output_path.parent.mkdir(exist_ok=True)

    manual_count = sum(1 for f in all_features if f["source"] == "manual")
    synth_count = len(all_features) - manual_count

    output = {
        "n_offers": len(offer_ids) - skipped,
        "n_manual": manual_count,
        "n_synthetic": synth_count,
        "features": all_features,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(all_features)} features ({manual_count} manual, {synth_count} synthetic) "
          f"to {output_path}")
    if skipped:
        print(f"  ({skipped} offers skipped)")


if __name__ == "__main__":
    main()
