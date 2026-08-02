#!/usr/bin/env python3
"""
Generate packer survey scenarios from historical boxes + overage.

Constructs two tiers of scenarios:
  Tier 1 (~80): random (offer, box) pairs, items removed to 50-75% fill,
    random candidates from remaining overage. General calibration.
  Tier 2 (~40-60): deliberately constructed to isolate specific scoring
    dimensions (group-qty saturation, diversity, value budget, max value
    share, desirability vs context).

Output: diagnostics/survey_scenarios.json

Usage:
    python3 scripts/generate_survey_scenarios.py                    # default
    python3 scripts/generate_survey_scenarios.py --seed 42          # reproducible
    python3 scripts/generate_survey_scenarios.py --tier1 80 --tier2 50
    python3 scripts/generate_survey_scenarios.py --only-offers 85-106
"""

import json
import logging
import sys
from datetime import datetime, timezone
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
    SKIP_COLUMN_IDENTIFIERS,
    STAFF_IDENTIFIERS,
)
from allocator.desirability import get_item_desirability
from compare import (
    _build_offer_ids,
    _discover_cleaned_offer_ids,
    build_item_lookup,
    compute_available_tags,
    load_historical_csv,
    load_summary,
    read_xlsx_pack_overrides,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_allocations_from_csv(box_name: str, hist_allocs: dict) -> dict[int, int]:
    """Extract {item_id: qty} for a single box from the CSV allocation dict.

    Historical data occasionally records a fractional share of a bulk item
    (e.g. 0.5 of a 1kg bag split across two boxes). Scenarios are integer-only,
    so any positive qty is coerced to a whole unit (floored at 1) here at the
    ingest boundary — no fractional quantity propagates downstream.
    """
    allocs = {}
    for item_id, per_box in hist_allocs.items():
        qty = per_box.get(box_name, 0)
        if qty > 0:
            allocs[item_id] = max(1, round(qty))
    return allocs


def _box_value(allocs: dict[int, int], item_lookup: dict) -> int:
    """Total value in cents for a box's allocations."""
    return sum(
        item_lookup[iid]["price"] * qty
        for iid, qty in allocs.items()
        if iid in item_lookup
    )


def _remove_items_to_fill(
    allocs: dict[int, int],
    item_lookup: dict,
    target_value: int,
    fill_pct: float,
    rng: Random,
) -> dict[int, int]:
    """Remove random items from a box until it reaches the target fill level."""
    target = int(target_value * fill_pct)
    result = dict(allocs)
    item_ids = list(result.keys())

    while _box_value(result, item_lookup) > target and item_ids:
        rng.shuffle(item_ids)
        removed = False
        for iid in list(item_ids):
            if iid not in result:
                item_ids.remove(iid)
                continue
            result[iid] -= 1
            if result[iid] <= 0:
                del result[iid]
                item_ids.remove(iid)
            removed = True
            break
        if not removed:
            break

    return result


def _format_item(item_id: int, info: dict, qty: int = 0) -> dict:
    """Format an item for the scenario JSON.

    Box items pass an integer qty >= 1; candidates pass qty=0 and stay
    qty-less (the omitted-qty key is the placed-vs-candidate discriminator).
    """
    d = {
        "item_id": item_id,
        "name": info["name"],
        "price_cents": info["price"],
        "size": info.get("size", 1) or 1,
        "fungible_group": info.get("fungible_group"),
        "sub_category": info.get("sub_category"),
        "usage": info.get("usage"),
        "colour": info.get("colour"),
        "shape": info.get("shape"),
        "category_name": info.get("category_name", ""),
    }
    if qty > 0:
        d["qty"] = qty
    return d


def _format_candidate(item_id: int, info: dict, overage: int) -> dict:
    """Format a candidate item for the scenario JSON."""
    d = _format_item(item_id, info)
    d["qty_available"] = overage
    return d


def _build_scenario(
    scenario_id: str,
    scenario_type: str,
    target_dimension: str | None,
    offer_id: int,
    box_name: str,
    reduced_allocs: dict[int, int],
    candidates: list[dict],
    item_lookup: dict,
    tier: str,
) -> dict:
    """Build a complete scenario dict."""
    target_value = BOX_TIERS[tier]["target_value"]
    current_value = _box_value(reduced_allocs, item_lookup)
    value_pct = current_value / target_value * 100 if target_value > 0 else 0

    current_items = []
    for iid, qty in sorted(reduced_allocs.items()):
        if iid in item_lookup:
            current_items.append(_format_item(iid, item_lookup[iid], qty))

    return {
        "id": scenario_id,
        "type": scenario_type,
        "target_dimension": target_dimension,
        "source_offer_id": offer_id,
        "source_box_name": box_name,
        "box": {
            "tier": tier,
            "target_value_cents": target_value,
            "current_items": current_items,
            "current_value_cents": current_value,
            "current_value_pct": round(value_pct, 1),
        },
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Offer data loading
# ---------------------------------------------------------------------------

def _load_offer_data(offer_id: int, summary: dict) -> dict | None:
    """Load all data needed for an offer. Returns None if unavailable."""
    item_lookup = build_item_lookup(offer_id)
    if not item_lookup:
        return None

    pack_overrides = read_xlsx_pack_overrides(offer_id)
    manual_lookup = (
        build_item_lookup(offer_id, price_overrides=pack_overrides)
        if pack_overrides else item_lookup
    )

    box_names, hist_allocs = load_historical_csv(offer_id)
    if not box_names:
        return None

    box_names = [
        bn for bn in box_names
        if bn not in DONATION_IDENTIFIERS
        and bn not in SKIP_COLUMN_IDENTIFIERS
        and bn not in STAFF_IDENTIFIERS
    ]

    # Build per-box data
    boxes = []
    for bn in box_names:
        tier = infer_box_tier(offer_id, bn, summary)
        if tier is None:
            continue
        allocs = _box_allocations_from_csv(bn, hist_allocs)
        # Drop item_ids missing from the lookup (e.g. soft-deleted after the
        # offer); otherwise they carry qty but zero value, understating the box
        # and — if a box is entirely such items — yielding empty current_items.
        allocs = {iid: q for iid, q in allocs.items() if iid in item_lookup}
        if not allocs:
            continue
        boxes.append({"name": bn, "tier": tier, "allocs": allocs})

    if not boxes:
        return None

    avail_tags = compute_available_tags(item_lookup)

    return {
        "offer_id": offer_id,
        "item_lookup": item_lookup,
        "manual_lookup": manual_lookup,
        "boxes": boxes,
        "avail_tags": avail_tags,
    }


# ---------------------------------------------------------------------------
# Tier 1: Random calibration scenarios
# ---------------------------------------------------------------------------

def generate_tier1_scenarios(
    offer_data_list: list[dict],
    n_scenarios: int,
    rng: Random,
    n_candidates: int = 6,
) -> list[dict]:
    """Generate Tier 1 random calibration scenarios."""
    scenarios = []
    counter = 0

    # Build pool of (offer_data, box) pairs
    pool = []
    for od in offer_data_list:
        for box in od["boxes"]:
            pool.append((od, box))

    if not pool:
        return []

    rng.shuffle(pool)

    for i in range(n_scenarios):
        od, box = pool[i % len(pool)]
        offer_id = od["offer_id"]
        item_lookup = od["manual_lookup"]
        tier = box["tier"]
        target_value = BOX_TIERS[tier]["target_value"]

        # Remove items to 50-75% fill
        fill_pct = rng.triangular(0.50, 0.75, 0.62)
        reduced = _remove_items_to_fill(
            box["allocs"], item_lookup, target_value, fill_pct, rng
        )

        if not reduced:
            continue

        # Select candidates from items NOT in the reduced box
        candidate_pool = [
            (iid, info) for iid, info in item_lookup.items()
            if iid not in reduced
        ]
        if len(candidate_pool) < 3:
            continue

        n_pick = min(n_candidates, len(candidate_pool))
        chosen = rng.sample(candidate_pool, n_pick)

        candidates = []
        for iid, info in chosen:
            # Estimate available overage (not critical for survey, use 5)
            candidates.append(_format_candidate(iid, info, overage=5))

        counter += 1
        scenario_id = f"t1_offer{offer_id}_{counter:03d}"
        scenarios.append(_build_scenario(
            scenario_id, "tier1_random", None,
            offer_id, box["name"], reduced, candidates, item_lookup, tier,
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Tier 2: Dimension-targeted scenarios
# ---------------------------------------------------------------------------

def _generate_group_qty_scenarios(
    offer_data_list: list[dict], rng: Random, n_target: int = 10,
) -> list[dict]:
    """Scenarios testing group-qty saturation.

    For boxes with a fungible group present, vary the group count
    and present another variety from the same group as a candidate.
    """
    scenarios = []
    counter = 0

    for od in offer_data_list:
        item_lookup = od["manual_lookup"]
        for box in od["boxes"]:
            if counter >= n_target:
                break

            # Find fungible groups present in the box
            group_items_in_box = {}
            for iid, qty in box["allocs"].items():
                if iid not in item_lookup:
                    continue
                fg = item_lookup[iid].get("fungible_group")
                if fg:
                    group_items_in_box.setdefault(fg, []).append((iid, qty))

            if not group_items_in_box:
                continue

            # Pick a group with multiple items available in the offer
            for group_name, box_items in group_items_in_box.items():
                # Find other items from same group NOT in box
                same_group_candidates = [
                    (iid, info) for iid, info in item_lookup.items()
                    if info.get("fungible_group") == group_name
                    and iid not in box["allocs"]
                ]
                if not same_group_candidates:
                    continue

                # Also get some non-group candidates for contrast
                non_group = [
                    (iid, info) for iid, info in item_lookup.items()
                    if info.get("fungible_group") != group_name
                    and iid not in box["allocs"]
                ]

                if not non_group:
                    continue

                # Build reduced box (keep group items to test saturation)
                tier = box["tier"]
                target_value = BOX_TIERS[tier]["target_value"]
                fill_pct = rng.uniform(0.55, 0.75)
                reduced = _remove_items_to_fill(
                    box["allocs"], item_lookup, target_value, fill_pct, rng
                )

                if not reduced:
                    continue

                # Candidates: 1-2 same-group + 3-4 non-group
                n_same = min(2, len(same_group_candidates))
                n_other = min(4, len(non_group))
                candidates = []
                for iid, info in rng.sample(same_group_candidates, n_same):
                    candidates.append(_format_candidate(iid, info, 5))
                for iid, info in rng.sample(non_group, n_other):
                    candidates.append(_format_candidate(iid, info, 5))

                rng.shuffle(candidates)
                counter += 1
                scenario_id = f"t2_gq_offer{od['offer_id']}_{counter:03d}"
                scenarios.append(_build_scenario(
                    scenario_id, "tier2_targeted", "group_qty",
                    od["offer_id"], box["name"], reduced, candidates,
                    item_lookup, tier,
                ))
                break  # one scenario per box

        if counter >= n_target:
            break

    return scenarios


def _generate_diversity_scenarios(
    offer_data_list: list[dict], rng: Random, n_target: int = 10,
) -> list[dict]:
    """Scenarios testing diversity weight sensitivity.

    Box heavy on one sub_category. Candidates that fill a gap vs duplicate.
    """
    scenarios = []
    counter = 0

    for od in offer_data_list:
        item_lookup = od["manual_lookup"]
        for box in od["boxes"]:
            if counter >= n_target:
                break

            # Count sub_categories in box
            subcat_counts = {}
            for iid, qty in box["allocs"].items():
                if iid not in item_lookup:
                    continue
                sc = item_lookup[iid].get("sub_category")
                if sc:
                    subcat_counts[sc] = subcat_counts.get(sc, 0) + qty

            if not subcat_counts:
                continue

            # Find the dominant sub_category
            dominant = max(subcat_counts, key=subcat_counts.get)
            total_items = sum(subcat_counts.values())
            dom_share = subcat_counts[dominant] / total_items

            if dom_share < 0.3:  # not concentrated enough
                continue

            # Find candidates: items from dominant subcat (bad) vs gap-fillers (good)
            dominant_candidates = [
                (iid, info) for iid, info in item_lookup.items()
                if info.get("sub_category") == dominant and iid not in box["allocs"]
            ]
            gap_candidates = [
                (iid, info) for iid, info in item_lookup.items()
                if info.get("sub_category") and info["sub_category"] != dominant
                and iid not in box["allocs"]
            ]

            if not dominant_candidates or not gap_candidates:
                continue

            tier = box["tier"]
            target_value = BOX_TIERS[tier]["target_value"]
            fill_pct = rng.uniform(0.55, 0.70)
            reduced = _remove_items_to_fill(
                box["allocs"], item_lookup, target_value, fill_pct, rng
            )
            if not reduced:
                continue

            # 2 dominant + 4 gap-fillers
            n_dom = min(2, len(dominant_candidates))
            n_gap = min(4, len(gap_candidates))
            candidates = []
            for iid, info in rng.sample(dominant_candidates, n_dom):
                candidates.append(_format_candidate(iid, info, 5))
            for iid, info in rng.sample(gap_candidates, n_gap):
                candidates.append(_format_candidate(iid, info, 5))

            rng.shuffle(candidates)
            counter += 1
            scenario_id = f"t2_div_offer{od['offer_id']}_{counter:03d}"
            scenarios.append(_build_scenario(
                scenario_id, "tier2_targeted", "diversity",
                od["offer_id"], box["name"], reduced, candidates,
                item_lookup, tier,
            ))

        if counter >= n_target:
            break

    return scenarios


def _generate_value_budget_scenarios(
    offer_data_list: list[dict], rng: Random, n_target: int = 10,
) -> list[dict]:
    """Scenarios testing value budget awareness.

    Box at different fill levels. Cheap vs expensive candidates.
    """
    scenarios = []
    counter = 0
    fill_targets = [0.85, 0.95, 1.05]  # near/at/over value target
    # Each fill level gets its own quota; a shared counter would let the first
    # level (0.85) exhaust n_target and starve the at/over-target cases.
    per_fill = max(1, -(-n_target // len(fill_targets)))

    for fill_factor in fill_targets:
        made = 0
        for od in offer_data_list:
            if made >= per_fill or counter >= n_target:
                break
            item_lookup = od["manual_lookup"]
            for box in od["boxes"]:
                if made >= per_fill or counter >= n_target:
                    break

                tier = box["tier"]
                target_value = BOX_TIERS[tier]["target_value"]
                reduced = _remove_items_to_fill(
                    box["allocs"], item_lookup, target_value, fill_factor, rng
                )
                if not reduced:
                    continue

                # Find cheap and expensive candidates
                available = [
                    (iid, info) for iid, info in item_lookup.items()
                    if iid not in reduced
                ]
                if len(available) < 4:
                    continue

                sorted_by_price = sorted(available, key=lambda x: x[1]["price"])
                cheap = sorted_by_price[:len(sorted_by_price) // 3]
                expensive = sorted_by_price[-(len(sorted_by_price) // 3):]

                if not cheap or not expensive:
                    continue

                n_cheap = min(3, len(cheap))
                n_exp = min(3, len(expensive))
                candidates = []
                for iid, info in rng.sample(cheap, n_cheap):
                    candidates.append(_format_candidate(iid, info, 5))
                for iid, info in rng.sample(expensive, n_exp):
                    candidates.append(_format_candidate(iid, info, 5))

                rng.shuffle(candidates)
                counter += 1
                made += 1
                scenario_id = f"t2_val_offer{od['offer_id']}_{counter:03d}"
                scenarios.append(_build_scenario(
                    scenario_id, "tier2_targeted", "value_budget",
                    od["offer_id"], box["name"], reduced, candidates,
                    item_lookup, tier,
                ))

    return scenarios


def _generate_mvs_scenarios(
    offer_data_list: list[dict], rng: Random, n_target: int = 10,
) -> list[dict]:
    """Scenarios testing max value share sensitivity.

    Box where one item dominates value. More of dominant vs something new.
    """
    scenarios = []
    counter = 0

    for od in offer_data_list:
        item_lookup = od["manual_lookup"]
        for box in od["boxes"]:
            if counter >= n_target:
                break

            total_val = _box_value(box["allocs"], item_lookup)
            if total_val == 0:
                continue

            # Find item with highest value share
            max_share_id = None
            max_share = 0
            for iid, qty in box["allocs"].items():
                if iid not in item_lookup:
                    continue
                share = item_lookup[iid]["price"] * qty / total_val
                if share > max_share:
                    max_share = share
                    max_share_id = iid

            if max_share < 0.15 or max_share_id is None:
                continue

            tier = box["tier"]
            target_value = BOX_TIERS[tier]["target_value"]
            fill_pct = rng.uniform(0.55, 0.75)
            reduced = _remove_items_to_fill(
                box["allocs"], item_lookup, target_value, fill_pct, rng
            )
            if not reduced:
                continue

            # Candidates: the dominant item (if available) + diverse alternatives
            candidates = []
            dom_info = item_lookup.get(max_share_id)
            if dom_info and max_share_id not in reduced:
                candidates.append(_format_candidate(max_share_id, dom_info, 5))

            alternatives = [
                (iid, info) for iid, info in item_lookup.items()
                if iid not in reduced and iid != max_share_id
            ]
            n_alt = min(5, len(alternatives))
            if n_alt < 2:
                continue
            for iid, info in rng.sample(alternatives, n_alt):
                candidates.append(_format_candidate(iid, info, 5))

            rng.shuffle(candidates)
            counter += 1
            scenario_id = f"t2_mvs_offer{od['offer_id']}_{counter:03d}"
            scenarios.append(_build_scenario(
                scenario_id, "tier2_targeted", "max_value_share",
                od["offer_id"], box["name"], reduced, candidates,
                item_lookup, tier,
            ))

        if counter >= n_target:
            break

    return scenarios


def _generate_desirability_scenarios(
    offer_data_list: list[dict], rng: Random, n_target: int = 10,
) -> list[dict]:
    """Scenarios testing desirability vs contextual fit.

    High-desirability item that's a bad fit vs low-desirability gap-filler.
    """
    scenarios = []
    counter = 0

    for od in offer_data_list:
        item_lookup = od["manual_lookup"]
        for box in od["boxes"]:
            if counter >= n_target:
                break

            # Score desirability for all available items
            available = [
                (iid, info, get_item_desirability(info["name"]))
                for iid, info in item_lookup.items()
                if iid not in box["allocs"]
            ]
            if len(available) < 6:
                continue

            # Sort by desirability
            available.sort(key=lambda x: x[2], reverse=True)
            high_des = available[:len(available) // 3]
            low_des = available[-(len(available) // 3):]

            if not high_des or not low_des:
                continue

            tier = box["tier"]
            target_value = BOX_TIERS[tier]["target_value"]
            fill_pct = rng.uniform(0.55, 0.70)
            reduced = _remove_items_to_fill(
                box["allocs"], item_lookup, target_value, fill_pct, rng
            )
            if not reduced:
                continue

            # Mix high and low desirability candidates
            n_high = min(3, len(high_des))
            n_low = min(3, len(low_des))
            candidates = []
            for iid, info, _ in rng.sample(high_des, n_high):
                candidates.append(_format_candidate(iid, info, 5))
            for iid, info, _ in rng.sample(low_des, n_low):
                candidates.append(_format_candidate(iid, info, 5))

            rng.shuffle(candidates)
            counter += 1
            scenario_id = f"t2_des_offer{od['offer_id']}_{counter:03d}"
            scenarios.append(_build_scenario(
                scenario_id, "tier2_targeted", "desirability",
                od["offer_id"], box["name"], reduced, candidates,
                item_lookup, tier,
            ))

        if counter >= n_target:
            break

    return scenarios


# ---------------------------------------------------------------------------
# Test-retest repeats
# ---------------------------------------------------------------------------

def generate_repeats(
    scenarios: list[dict],
    n_repeats: int,
    rng: Random,
) -> list[dict]:
    """Select scenarios for test-retest reliability and create shuffled copies.

    Samples proportionally from tier1 and tier2 scenarios. Each repeat
    gets a new ID and re-randomised candidate order, with a ``repeat_of``
    field linking to the original.
    """
    if n_repeats <= 0 or not scenarios:
        return []

    n_repeats = min(n_repeats, len(scenarios))

    tier1 = [s for s in scenarios if s["type"] == "tier1_random"]
    tier2 = [s for s in scenarios if s["type"] != "tier1_random"]

    # Proportional sampling across tiers
    n_t1 = round(n_repeats * len(tier1) / len(scenarios)) if scenarios else 0
    n_t2 = n_repeats - n_t1

    selected = []
    if tier1:
        selected += rng.sample(tier1, min(n_t1, len(tier1)))
    if tier2:
        selected += rng.sample(tier2, min(n_t2, len(tier2)))

    repeats = []
    for s in selected:
        repeat = json.loads(json.dumps(s))  # deep copy
        repeat["id"] = f"r_{s['id']}"
        repeat["repeat_of"] = s["id"]
        # Re-randomise candidate presentation order
        rng.shuffle(repeat["candidates"])
        repeats.append(repeat)

    return repeats


# ---------------------------------------------------------------------------
# Contract validation (matches the Jointly.Shop consumer's requirements)
# ---------------------------------------------------------------------------

def validate_scenarios(scenarios: list[dict]) -> list[str]:
    """Check every scenario against the consumer contract.

    Returns a list of human-readable violations (empty == valid). Box items
    must carry an integer qty >= 1; candidate item_ids must be unique within a
    scenario; required keys must be present.
    """
    errors = []
    seen_ids = set()
    for s in scenarios:
        sid = s.get("id")
        if sid is None:
            errors.append("scenario with no id")
        elif sid in seen_ids:
            errors.append(f"duplicate scenario id: {sid}")
        seen_ids.add(sid)

        for key in ("type", "source_offer_id"):
            if key not in s:
                errors.append(f"{sid}: missing {key}")

        box = s.get("box") or {}
        for key in ("tier", "target_value_cents", "current_value_cents",
                    "current_value_pct", "current_items"):
            if key not in box:
                errors.append(f"{sid}: missing box.{key}")

        items = box.get("current_items", [])
        if "current_items" in box and not items:
            errors.append(f"{sid}: empty current_items")
        for it in items:
            name = it.get("name", "?")
            if "name" not in it:
                errors.append(f"{sid}: box item missing name")
            if "price_cents" not in it:
                errors.append(f"{sid}: box item {name} missing price_cents")
            qty = it.get("qty")
            if not isinstance(qty, int) or qty < 1:
                errors.append(f"{sid}: box item {name} has bad qty {qty!r} (need int >= 1)")

        candidates = s.get("candidates") or []
        if not candidates:
            errors.append(f"{sid}: no candidates")
        cand_ids = [c.get("item_id") for c in candidates]
        if len(cand_ids) != len(set(cand_ids)):
            errors.append(f"{sid}: duplicate candidate item_id")
        for c in candidates:
            for key in ("item_id", "name", "price_cents"):
                if key not in c:
                    errors.append(f"{sid}: candidate missing {key}")
            # Omitted-qty is the placed-vs-candidate discriminator; a candidate
            # carrying qty would render as a placed item on the consumer.
            if "qty" in c:
                errors.append(f"{sid}: candidate {c.get('name', '?')} must not carry qty")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate packer survey scenarios from historical data"
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--tier1", type=int, default=80,
                        help="Number of Tier 1 (random) scenarios (default: 80)")
    parser.add_argument("--tier2", type=int, default=50,
                        help="Total Tier 2 (targeted) scenarios (default: 50)")
    parser.add_argument("--only-offers", type=str, default="85-109",
                        help="Offer range (default: 85-109, post-85 Tier A)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: diagnostics/survey_scenarios.json)")
    parser.add_argument("--candidates", type=int, default=6,
                        help="Candidates per Tier 1 scenario (default: 6)")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repeated scenarios for test-retest reliability (default: 10)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else 2026
    rng = Random(seed)

    summary = load_summary()
    offer_ids = _build_offer_ids(summary, only_offers=args.only_offers)
    print(f"Loading {len(offer_ids)} offers...")

    # Load all offer data
    offer_data_list = []
    for oid in offer_ids:
        od = _load_offer_data(oid, summary)
        if od:
            offer_data_list.append(od)
            n_boxes = len(od["boxes"])
            print(f"  Offer {oid}: {n_boxes} boxes, {len(od['item_lookup'])} items")
        else:
            print(f"  [SKIP] Offer {oid}: no data")

    if not offer_data_list:
        print("No offer data available. Check DB connection and cleaned CSVs.")
        sys.exit(1)

    rng.shuffle(offer_data_list)

    # Generate Tier 1
    print(f"\nGenerating {args.tier1} Tier 1 scenarios...")
    tier1 = generate_tier1_scenarios(offer_data_list, args.tier1, rng, args.candidates)
    print(f"  Generated {len(tier1)} Tier 1 scenarios")

    # Generate Tier 2 (split target across 5 dimensions)
    per_dim = max(1, args.tier2 // 5)
    print(f"\nGenerating Tier 2 scenarios (~{per_dim} per dimension)...")

    tier2_gq = _generate_group_qty_scenarios(offer_data_list, rng, per_dim)
    print(f"  Group-qty saturation: {len(tier2_gq)}")

    tier2_div = _generate_diversity_scenarios(offer_data_list, rng, per_dim)
    print(f"  Diversity coverage: {len(tier2_div)}")

    tier2_val = _generate_value_budget_scenarios(offer_data_list, rng, per_dim)
    print(f"  Value budget: {len(tier2_val)}")

    tier2_mvs = _generate_mvs_scenarios(offer_data_list, rng, per_dim)
    print(f"  Max value share: {len(tier2_mvs)}")

    tier2_des = _generate_desirability_scenarios(offer_data_list, rng, per_dim)
    print(f"  Desirability vs context: {len(tier2_des)}")

    tier2 = tier2_gq + tier2_div + tier2_val + tier2_mvs + tier2_des

    all_scenarios = tier1 + tier2

    # Generate test-retest repeats
    print(f"\nGenerating {args.repeats} test-retest repeat scenarios...")
    repeats = generate_repeats(all_scenarios, args.repeats, rng)
    print(f"  Generated {len(repeats)} repeats "
          f"({sum(1 for r in repeats if not r.get('target_dimension'))} tier1, "
          f"{sum(1 for r in repeats if r.get('target_dimension'))} tier2)")

    # Interleave repeats into the second half so packer doesn't see a
    # cluster of "same boxes again" at the very end.
    midpoint = len(all_scenarios) // 2
    second_half = all_scenarios[midpoint:] + repeats
    rng.shuffle(second_half)
    all_scenarios = all_scenarios[:midpoint] + second_half

    # Validate against the consumer contract before writing — never ship an
    # out-of-spec file that could 500 the survey page.
    errors = validate_scenarios(all_scenarios)
    if errors:
        print(f"\nContract validation FAILED ({len(errors)} issues):")
        for e in errors[:50]:
            print(f"  - {e}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        sys.exit(1)
    print("\nContract validation passed.")

    # Write output
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "diagnostics" / "survey_scenarios.json"
    )
    output_path.parent.mkdir(exist_ok=True)

    output = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_scenarios": len(all_scenarios),
        "n_tier1": len(tier1),
        "n_tier2": len(tier2),
        "n_repeats": len(repeats),
        "repeat_pairs": [
            {"original": r["repeat_of"], "repeat": r["id"]}
            for r in repeats
        ],
        "scenarios": all_scenarios,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(all_scenarios)} scenarios ({len(tier1)} tier1, {len(tier2)} tier2, "
          f"{len(repeats)} repeats) to {output_path}")


if __name__ == "__main__":
    main()
