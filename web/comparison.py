"""
Bridge between compare.py data functions and web templates.

Builds structured comparison data for side-by-side box rendering.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from allocator.allocator import allocate
from allocator.box_parser import infer_box_tier
from allocator.config import (
    DIVERSITY_PENALTY_MULTIPLIER,
    DONATION_IDENTIFIERS,
    MAX_COMPOSITE_SCORE,
    PREF_VIOLATION_PENALTY,
    SKIP_COLUMN_IDENTIFIERS,
    STAFF_IDENTIFIERS,
)
from allocator.db import fetch_mystery_box_buyers
from allocator.strategies._scoring import value_penalty
from compare import (
    _discover_cleaned_offer_ids,
    _find_xlsx_path,
    build_item_lookup,
    compute_available_tags,
    compute_box_metrics,
    compute_composite_score,
    load_historical_csv,
    load_summary,
    read_xlsx_pack_overrides,
)
from allocator.strategies import list_strategies


def _annotate_box_score(m: dict) -> None:
    """Add a per-box score to a metrics dict (in place)."""
    vp = value_penalty(m["value_pct"])
    si = m.get("same_item_penalty", 0.0)
    gc = m.get("group_concentration_penalty", 0.0)
    div = (1.0 - m["diversity_score"]) * DIVERSITY_PENALTY_MULTIPLIER
    mvs = m.get("max_value_share_penalty", 0.0)
    sf = m.get("size_floor_penalty", 0.0)
    pref = m["pref_violations"] * PREF_VIOLATION_PENALTY
    m["box_score"] = MAX_COMPOSITE_SCORE - vp - si - gc - div - mvs - sf - pref


def compute_item_diff(manual_items: dict[int, int], algo_items: dict[int, int]) -> dict:
    """
    Compute diff between two {item_id: qty} dicts.

    Returns dict with keys: added, removed, changed, unchanged.
    """
    all_ids = set(manual_items.keys()) | set(algo_items.keys())
    diff = {"added": {}, "removed": {}, "changed": {}, "unchanged": {}}

    for iid in all_ids:
        mq = manual_items.get(iid, 0)
        aq = algo_items.get(iid, 0)
        if mq == 0 and aq > 0:
            diff["added"][iid] = aq
        elif mq > 0 and aq == 0:
            diff["removed"][iid] = mq
        elif mq != aq:
            diff["changed"][iid] = (mq, aq)
        else:
            diff["unchanged"][iid] = mq

    return diff


def build_box_pairs(
    manual_metrics: list[dict],
    algo_metrics: list[dict],
    manual_per_box: dict[str, dict[int, int]],
    algo_per_box: dict[str, dict[int, int]],
) -> list[dict]:
    """
    Match manual and algo boxes by name and build pair dicts with diffs.

    Pure function — no DB or file access.
    """
    manual_by_name = {m["box_name"]: m for m in manual_metrics}
    algo_by_name = {m["box_name"]: m for m in algo_metrics}
    all_names = list(dict.fromkeys(
        list(manual_by_name.keys()) + list(algo_by_name.keys())
    ))

    pairs = []
    for bn in all_names:
        m_metrics = manual_by_name.get(bn)
        a_metrics = algo_by_name.get(bn)
        m_items = manual_per_box.get(bn, {})
        a_items = algo_per_box.get(bn, {})
        tier = (m_metrics or a_metrics or {}).get("tier", "?")

        pairs.append({
            "box_name": bn,
            "tier": tier,
            "manual": {"metrics": m_metrics, "items": m_items},
            "algo": {"metrics": a_metrics, "items": a_items},
            "diff": compute_item_diff(m_items, a_items),
        })

    return pairs


def build_comparison_data(offer_id: int, algorithm: str) -> dict:
    """
    Build full comparison data for an offer: manual vs algorithm.

    Requires DB access (for item lookup, buyers, and allocate()).
    Returns structured dict ready for template rendering.
    """
    t0 = time.time()

    summary = load_summary()
    item_lookup = build_item_lookup(offer_id)
    if not item_lookup:
        raise ValueError(f"No items found for offer {offer_id}")

    avail_tags = compute_available_tags(item_lookup)

    # Manual pack-price overrides
    pack_overrides = read_xlsx_pack_overrides(offer_id)
    manual_item_lookup = (
        build_item_lookup(offer_id, price_overrides=pack_overrides)
        if pack_overrides else item_lookup
    )

    # --- Manual side ---
    box_names, hist_allocs = load_historical_csv(offer_id)
    if not box_names:
        raise ValueError(f"No cleaned CSV found for offer {offer_id}")

    box_names = [
        bn for bn in box_names
        if bn not in DONATION_IDENTIFIERS
        and bn not in SKIP_COLUMN_IDENTIFIERS
        and bn not in STAFF_IDENTIFIERS
    ]

    buyers_db = fetch_mystery_box_buyers(offer_id)
    buyer_prefs = {}
    for buyer in buyers_db:
        email = buyer["user_email"]
        opt = buyer.get("selected_option") or ""
        if "no veg" in opt.lower():
            buyer_prefs[email] = "fruit_only"
        elif "no fruit" in opt.lower():
            buyer_prefs[email] = "veg_only"

    # Invert hist_allocs: {item_id: {box: qty}} → {box: {item_id: qty}}
    manual_per_box: dict[str, dict[int, int]] = {}
    for item_id, per_box in hist_allocs.items():
        for bn, qty in per_box.items():
            if qty > 0:
                manual_per_box.setdefault(bn, {})[item_id] = int(qty)

    manual_metrics = []
    for bn in box_names:
        tier = infer_box_tier(offer_id, bn, summary)
        if tier is None:
            continue
        box_items = manual_per_box.get(bn, {})
        pref = buyer_prefs.get(bn)
        m = compute_box_metrics(
            bn, box_items, manual_item_lookup, tier,
            preference=pref, available_tags=avail_tags,
        )
        if m:
            m["source"] = "manual"
            _annotate_box_score(m)
            manual_metrics.append(m)

    # --- Algorithm side ---
    xlsx_path = _find_xlsx_path(offer_id)
    if xlsx_path is None:
        raise ValueError(f"No XLSX found for offer {offer_id}")

    result = allocate(offer_id, xlsx_path, strategy=algorithm)

    algo_per_box: dict[str, dict[int, int]] = {}
    algo_metrics = []
    for box in result.boxes:
        algo_per_box[box.name] = dict(box.allocations)
        m = compute_box_metrics(
            box.name, box.allocations, item_lookup, box.tier,
            preference=box.preference, available_tags=avail_tags,
        )
        if m:
            m["source"] = "algorithm"
            _annotate_box_score(m)
            algo_metrics.append(m)

    # --- Composites + pairs ---
    manual_composite = compute_composite_score(manual_metrics)
    algo_composite = compute_composite_score(algo_metrics)
    box_pairs = build_box_pairs(manual_metrics, algo_metrics, manual_per_box, algo_per_box)

    elapsed = time.time() - t0

    return {
        "offer_id": offer_id,
        "algorithm": algorithm,
        "item_lookup": item_lookup,
        "manual_composite": manual_composite,
        "algo_composite": algo_composite,
        "box_pairs": box_pairs,
        "elapsed": round(elapsed, 1),
    }


def get_available_offers() -> list[int]:
    """Return sorted list of offer IDs that have cleaned CSVs."""
    return sorted(_discover_cleaned_offer_ids())


def get_available_algorithms() -> list[str]:
    """Return list of available strategy names."""
    return list_strategies(include_baselines=True)
