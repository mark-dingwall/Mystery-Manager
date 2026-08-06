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
    DONATION_IDENTIFIERS,
    SKIP_COLUMN_IDENTIFIERS,
    STAFF_IDENTIFIERS,
)
from allocator.box_features import UnsupportedCategoryError
from allocator.box_features import extract_box_features


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_or_skip(
    box_name: str,
    allocations: dict[int, int],
    item_lookup: dict[int, dict],
    tier: str,
    available_tags: dict[str, set[str]],
    offer_id: int,
    source: str = "manual",
    preference: str | None = None,
) -> dict | None:
    try:
        return extract_box_features(
            box_name,
            allocations,
            item_lookup,
            tier,
            available_tags,
            offer_id,
            source=source,
            preference=preference,
        )
    except UnsupportedCategoryError as exc:
        print(f"  [SKIP] {exc}")
        return None


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
        f = _extract_or_skip(
            f"synth_mono_{tier}", mono_allocs, item_lookup, tier,
            available_tags, offer_id, source="synth_monoculture",
        )
        if f:
            synthetics.append(f)

        # 2. Random: uniform sample to target value
        random_allocs = _fill_to_target(sorted_items, int(target * 1.15), rng=rng)
        f = _extract_or_skip(
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
            f = _extract_or_skip(
                f"synth_overfg_{tier}", overfg_allocs, item_lookup, tier,
                available_tags, offer_id, source="synth_over_fungible",
            )
            if f:
                synthetics.append(f)

        # 4. Value low: items to ~70% target
        low_allocs = _fill_to_target(sorted_items, int(target * 0.70))
        f = _extract_or_skip(
            f"synth_low_{tier}", low_allocs, item_lookup, tier,
            available_tags, offer_id, source="synth_value_low",
        )
        if f:
            synthetics.append(f)

        # 5. Value high: items to ~160% target
        high_allocs = _fill_to_target(sorted_items, int(target * 1.60))
        f = _extract_or_skip(
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

    from compare import (
        _build_offer_ids,
        build_item_lookup,
        compute_available_tags,
        load_historical_csv,
        load_summary,
        read_xlsx_pack_overrides,
    )
    from allocator.db import fetch_mystery_box_buyers

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
            f = _extract_or_skip(
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
