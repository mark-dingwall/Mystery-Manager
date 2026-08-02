#!/usr/bin/env python3
"""
Scoring diagnostic analysis for historical mystery box allocations.

Processes all available historical offers (Tiers A-D), computes per-box and
per-offer penalty breakdowns, detects pricing anomalies, generates visualisations,
and writes a structured JSON report for downstream consumption.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Suppress noise from allocator library and paramiko
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("allocator.categorizer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from allocator.box_parser import classify_box, infer_box_tier
from allocator.config import (
    BOX_TIERS,
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    DONATION_IDENTIFIERS,
    GROUP_CONCENTRATION_MULTIPLIER,
    GROUP_QTY_EXPONENT,
    MAX_COMPOSITE_SCORE,
    MAX_VALUE_SHARE_MULTIPLIER,
    MAX_VALUE_SHARE_THRESHOLD,
    PREF_VIOLATION_PENALTY,
    SAME_ITEM_MULTIPLIER,
    SIZE_FLOOR_MULTIPLIER,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.strategies._scoring import value_penalty
from compare import (
    build_item_lookup,
    compute_available_tags,
    compute_box_metrics,
    compute_composite_score,
    load_historical_csv,
    load_summary,
    read_xlsx_pack_overrides,
)

CLEANED_DIR = Path(__file__).parent.parent / "cleaned"
DIAGNOSTICS_DIR = Path(__file__).parent.parent / "diagnostics"

# ANSI colours
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[90m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Offer discovery and classification
# ---------------------------------------------------------------------------

def offer_tier(offer_id: int) -> str:
    if offer_id >= 64:
        return "A"
    if offer_id >= 55:
        return "B"
    if offer_id >= 49:
        return "C"
    return "D"


def discover_offer_ids() -> list[int]:
    """Scan cleaned/ for offer_*_mystery.csv files and return sorted IDs."""
    ids = set()
    for f in CLEANED_DIR.glob("offer_*_mystery.csv"):
        try:
            num = int(f.stem.split("_")[1])
            ids.add(num)
        except (ValueError, IndexError):
            pass
    return sorted(ids)


def _parse_only_offers(raw: str) -> set[int]:
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


_SKIP_BOX_NAMES = {"Unallocated", "unallocated", "Stock", "stock", "Buffer", "buffer",
                    "Volunteers", "volunteers", "Overs", "overs"}


def _should_exclude_box(box_name: str, offer_meta: dict) -> bool:
    """Check if a box should be excluded from scoring (matches compare.py behavior)."""
    # Explicit donation/charity names from summary.json
    donation_names = set(offer_meta.get("donation_names", []))
    charity_names = set(offer_meta.get("charity_names", []))
    if box_name in (donation_names | charity_names | DONATION_IDENTIFIERS):
        return True

    # Common non-box column names
    if box_name in _SKIP_BOX_NAMES:
        return True

    # Use box_parser to check type
    _, _, box_type = classify_box(box_name)
    if box_type in ("donation", "charity", "staff", "skip"):
        return True

    return False


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_offer(offer_id: int, summary: dict) -> dict | None:
    """
    Collect scoring data for a single offer.

    Returns dict with offer-level and per-box data, or None on failure.
    """
    box_names, hist_allocs = load_historical_csv(offer_id)
    if not box_names:
        return None

    try:
        pack_overrides = read_xlsx_pack_overrides(offer_id)
    except Exception:
        pack_overrides = {}

    try:
        item_lookup = build_item_lookup(offer_id, price_overrides=pack_overrides or None)
    except Exception:
        return None

    if not item_lookup:
        return None

    avail_tags = compute_available_tags(item_lookup)

    offer_meta = summary.get("offers", {}).get(str(offer_id), {})

    # Filter out donation/charity boxes
    box_names = [bn for bn in box_names if not _should_exclude_box(bn, offer_meta)]

    # Count missing items
    all_item_ids = set()
    for item_id, per_box in hist_allocs.items():
        for bn in box_names:
            if per_box.get(bn, 0) > 0:
                all_item_ids.add(item_id)
    missing_items = all_item_ids - set(item_lookup.keys())
    missing_pct = len(missing_items) / len(all_item_ids) * 100 if all_item_ids else 0

    # Compute per-box metrics
    box_metrics = []
    for bn in box_names:
        tier = infer_box_tier(offer_id, bn, summary)
        if tier is None:
            continue
        box_allocs = {}
        for item_id, per_box in hist_allocs.items():
            qty = per_box.get(bn, 0)
            if qty > 0:
                box_allocs[item_id] = qty

        m = compute_box_metrics(bn, box_allocs, item_lookup, tier, available_tags=avail_tags)
        if m is None:
            continue

        # Add per-box penalty breakdown. same_item, group_concentration,
        # max_value_share and size_floor penalties are already on m from
        # compute_box_metrics; derive the value/diversity/pref components here.
        vp = m["value_pct"]
        m["value_penalty"] = value_penalty(vp)
        m["diversity_penalty"] = (1.0 - m["diversity_score"]) * DIVERSITY_PENALTY_MULTIPLIER
        m["pref_penalty"] = m["pref_violations"] * PREF_VIOLATION_PENALTY
        box_metrics.append(m)

    if not box_metrics:
        return None

    # Compute offer-level composite score
    composite = compute_composite_score(box_metrics)

    # Pricing plausibility
    pricing = _assess_pricing(box_metrics)

    data_tier = offer_tier(offer_id)

    return {
        "offer_id": offer_id,
        "data_tier": data_tier,
        "box_count": len(box_metrics),
        "missing_items": len(missing_items),
        "missing_item_pct": round(missing_pct, 1),
        "composite": composite,
        "pricing": pricing,
        "boxes": box_metrics,
    }


def _assess_pricing(box_metrics: list[dict]) -> dict:
    """Check if box values are plausible for their detected tiers."""
    anomalies = []
    for m in box_metrics:
        tier = m["tier"]
        vp = m["value_pct"]
        total = m["total_value"]

        # Check if value falls within 100-130% of its tier's price
        in_range = 100 <= vp <= 130

        if not in_range:
            # Check if it fits a *different* tier
            fits_other = None
            for other_tier, info in BOX_TIERS.items():
                if other_tier == tier:
                    continue
                other_target = info["target_value"]
                other_pct = total / other_target * 100 if other_target > 0 else 0
                if 100 <= other_pct <= 130:
                    fits_other = other_tier
                    break

            anomalies.append({
                "box_name": m["box_name"],
                "tier": tier,
                "value_pct": round(vp, 1),
                "fits_other_tier": fits_other,
            })

    plausible = len(anomalies) <= len(box_metrics) * 0.25  # tolerate up to 25% anomalies
    return {
        "plausible": plausible,
        "anomaly_count": len(anomalies),
        "total_boxes": len(box_metrics),
        "anomalies": anomalies[:5],  # keep report manageable
    }


def collect_all_offers(offer_ids: list[int], summary: dict) -> list[dict]:
    """Collect data for all offers, with progress."""
    results = []
    total = len(offer_ids)
    for i, offer_id in enumerate(offer_ids, 1):
        tier = offer_tier(offer_id)
        sys.stdout.write(f"\r  Processing offer {offer_id} (Tier {tier}) [{i}/{total}]")
        sys.stdout.flush()
        data = collect_offer(offer_id, summary)
        if data:
            results.append(data)
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _all_boxes(offers: list[dict]) -> list[dict]:
    """Flatten all box metrics from all offers."""
    boxes = []
    for o in offers:
        for b in o["boxes"]:
            b["offer_id"] = o["offer_id"]
            b["data_tier"] = o["data_tier"]
            boxes.append(b)
    return boxes


def _percentiles(values: list[float], pcts: list[int]) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in pcts}
    sv = sorted(values)
    n = len(sv)
    result = {}
    for p in pcts:
        idx = max(0, min(n - 1, int(p / 100 * n)))
        result[p] = sv[idx]
    return result


def compute_aggregate_stats(boxes: list[dict]) -> dict:
    """Mean/std/percentiles for each metric across all boxes."""
    if not boxes:
        return {}
    n = len(boxes)
    vp_vals = [b["value_pct"] for b in boxes]
    div_vals = [b["diversity_score"] for b in boxes]
    val_pen = [b["value_penalty"] for b in boxes]
    si_pen = [b["same_item_penalty"] for b in boxes]
    gc_pen = [b["group_concentration_penalty"] for b in boxes]
    div_pen = [b["diversity_penalty"] for b in boxes]
    mvs_pen = [b["max_value_share_penalty"] for b in boxes]
    sf_pen = [b["size_floor_penalty"] for b in boxes]

    mean_vp = sum(vp_vals) / n
    std_vp = (sum((v - mean_vp) ** 2 for v in vp_vals) / n) ** 0.5

    return {
        "box_count": n,
        "value_pct": {
            "mean": round(mean_vp, 2),
            "std": round(std_vp, 2),
            "percentiles": {str(k): round(v, 1) for k, v in
                            _percentiles(vp_vals, [5, 25, 50, 75, 95]).items()},
            "in_sweet_spot": sum(1 for v in vp_vals if VALUE_SWEET_FROM <= v <= VALUE_SWEET_TO),
            "below_100": sum(1 for v in vp_vals if v < 100),
            "above_130": sum(1 for v in vp_vals if v > 130),
        },
        "diversity": {
            "mean": round(sum(div_vals) / n, 3),
            "std": round((sum((v - sum(div_vals) / n) ** 2 for v in div_vals) / n) ** 0.5, 3),
            "percentiles": {str(k): round(v, 3) for k, v in
                            _percentiles(div_vals, [5, 25, 50, 75, 95]).items()},
        },
        "penalties": {
            "value": {"mean": round(sum(val_pen) / n, 2), "total": round(sum(val_pen), 1)},
            "same_item": {"mean": round(sum(si_pen) / n, 2), "total": round(sum(si_pen), 1)},
            "group_concentration": {"mean": round(sum(gc_pen) / n, 2), "total": round(sum(gc_pen), 1)},
            "diversity": {"mean": round(sum(div_pen) / n, 2), "total": round(sum(div_pen), 1)},
            "max_value_share": {"mean": round(sum(mvs_pen) / n, 2), "total": round(sum(mvs_pen), 1)},
            "size_floor": {"mean": round(sum(sf_pen) / n, 2), "total": round(sum(sf_pen), 1)},
        },
        "unique_items": {
            "mean": round(sum(b["unique_items"] for b in boxes) / n, 1),
        },
        "fungible_dupes": {
            "mean": round(sum(b["fungible_dupes"] for b in boxes) / n, 2),
            "boxes_with_dupes": sum(1 for b in boxes if b["fungible_dupes"] > 0),
        },
    }


def compute_tier_breakdown(boxes: list[dict], key: str) -> dict[str, dict]:
    """Group boxes by a key (e.g. 'tier' for size, 'data_tier' for A/B/C/D) and compute stats."""
    groups = defaultdict(list)
    for b in boxes:
        groups[b[key]].append(b)
    return {k: compute_aggregate_stats(v) for k, v in sorted(groups.items())}


def compute_what_if_sweet_spots(boxes: list[dict]) -> list[dict]:
    """Re-compute value penalties under alternative sweet spot ranges."""
    vp_vals = [b["value_pct"] for b in boxes]
    n = len(vp_vals)
    if n == 0:
        return []

    # Pre-compute base constants
    current_score = _composite_from_penalties(boxes)

    ranges = [
        (114, 117, "current"),
        (112, 119, "112-119"),
        (110, 120, "110-120"),
        (108, 122, "108-122"),
        (105, 125, "105-125"),
        (112, 117, "112-117 (shifted low)"),
        (114, 120, "114-120 (shifted high)"),
        (110, 117, "110-117 (wider low)"),
        (114, 125, "114-125 (wider high)"),
        (110, 125, "110-125 (wide)"),
    ]

    results = []
    for lo, hi, label in ranges:
        total_val_pen = 0.0
        for vp in vp_vals:
            total_val_pen += _value_penalty_custom(vp, lo, hi)
        avg_val_pen = total_val_pen / n

        # Recompute score with new value penalty, keeping other penalties same
        other_pens = current_score["si_pen"] + current_score["gc_pen"] + \
                     current_score["diversity_pen"] + current_score["mvs_pen"] + \
                     current_score["sf_pen"] + current_score["pref_pen"]
        score = MAX_COMPOSITE_SCORE - avg_val_pen - other_pens

        in_sweet = sum(1 for v in vp_vals if lo <= v <= hi)

        results.append({
            "label": label,
            "low": lo,
            "high": hi,
            "avg_value_penalty": round(avg_val_pen, 2),
            "composite_score": round(score, 1),
            "score_delta": round(score - current_score["score"], 1),
            "boxes_in_sweet_spot": in_sweet,
            "pct_in_sweet_spot": round(in_sweet / n * 100, 1),
        })

    return results


def _value_penalty_custom(vp: float, sweet_low: float, sweet_high: float) -> float:
    """Value penalty with custom sweet spot bounds, same power function."""
    if sweet_low <= vp <= sweet_high:
        return 0.0
    if vp < sweet_low:
        x = sweet_low - vp
    else:
        x = vp - sweet_high
    return x ** VALUE_PENALTY_EXPONENT


def _composite_from_penalties(boxes: list[dict]) -> dict:
    """Compute composite from per-box penalties (already computed)."""
    return compute_composite_score(boxes)


def compute_correlations(boxes: list[dict]) -> dict[str, dict[str, float]]:
    """Compute correlation between penalty components."""
    if len(boxes) < 3:
        return {}

    fields = ["value_penalty", "same_item_penalty", "group_concentration_penalty", "diversity_penalty", "max_value_share_penalty", "size_floor_penalty", "value_pct", "diversity_score"]
    data = {f: [b.get(f, 0) for b in boxes] for f in fields}
    n = len(boxes)

    corr = {}
    for f1 in fields:
        corr[f1] = {}
        for f2 in fields:
            mean1 = sum(data[f1]) / n
            mean2 = sum(data[f2]) / n
            cov = sum((data[f1][i] - mean1) * (data[f2][i] - mean2) for i in range(n)) / n
            std1 = (sum((v - mean1) ** 2 for v in data[f1]) / n) ** 0.5
            std2 = (sum((v - mean2) ** 2 for v in data[f2]) / n) ** 0.5
            if std1 > 0 and std2 > 0:
                corr[f1][f2] = round(cov / (std1 * std2), 3)
            else:
                corr[f1][f2] = 0.0
    return corr


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------

def print_summary(offers: list[dict], boxes: list[dict], what_if: list[dict]):
    """Print concise diagnostic summary to stdout."""
    n_offers = len(offers)
    n_boxes = len(boxes)

    # Header
    print(f"\n{_BOLD}Scoring Diagnostic Report{_RESET}")
    print(f"  {n_offers} offers, {n_boxes} boxes\n")

    # Offer tier counts
    tier_counts = defaultdict(int)
    for o in offers:
        tier_counts[o["data_tier"]] += 1
    tier_str = ", ".join(f"Tier {t}: {c}" for t, c in sorted(tier_counts.items()))
    print(f"  Tiers: {tier_str}\n")

    # Global composite score (all boxes pooled — matches compare.py leaderboard)
    global_composite = compute_composite_score(boxes)
    print(f"  {_BOLD}Global Composite Score (all boxes pooled){_RESET}")
    print(f"    Score: {_BOLD}{global_composite['score']:.1f}{_RESET} / {MAX_COMPOSITE_SCORE:.0f}")
    print(f"    Value: -{global_composite['value_pen']:.1f}  SameItem: -{global_composite['si_pen']:.1f}  "
          f"Group: -{global_composite['gc_pen']:.1f}  Diversity: -{global_composite['diversity_pen']:.1f}  "
          f"MaxVal: -{global_composite['mvs_pen']:.1f}  Size: -{global_composite['sf_pen']:.1f}  "
          f"Pref: -{global_composite['pref_pen']:.1f}\n")

    # Composite score distribution per-offer
    offer_scores = [o["composite"]["score"] for o in offers]
    if offer_scores:
        pcts = _percentiles(offer_scores, [5, 25, 50, 75, 95])
        mean_score = sum(offer_scores) / len(offer_scores)
        print(f"  {_BOLD}Per-Offer Composite Score Distribution{_RESET}")
        print(f"    Mean: {mean_score:.1f}  |  P5: {pcts[5]:.1f}  P25: {pcts[25]:.1f}  "
              f"P50: {pcts[50]:.1f}  P75: {pcts[75]:.1f}  P95: {pcts[95]:.1f}\n")

    # Penalty breakdown (global composite)
    gc = global_composite
    total_pen = (gc["value_pen"] + gc["si_pen"] + gc["gc_pen"] + gc["diversity_pen"]
                 + gc["mvs_pen"] + gc["sf_pen"] + gc["pref_pen"])
    if total_pen > 0:
        print(f"  {_BOLD}Penalty Breakdown (global){_RESET}")
        print(f"    {'Dimension':<15} {'Penalty':>8} {'% of total':>10}")
        print(f"    {'─' * 35}")
        for name, pen in [("Value", gc["value_pen"]), ("Same-item", gc["si_pen"]),
                          ("Group", gc["gc_pen"]), ("Diversity", gc["diversity_pen"]),
                          ("Max-val-share", gc["mvs_pen"]), ("Size-floor", gc["sf_pen"]),
                          ("Preference", gc["pref_pen"])]:
            pct = pen / total_pen * 100 if total_pen > 0 else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {name:<15} {pen:>7.1f}  {pct:>6.1f}%  {bar}")
        print(f"    {'─' * 35}")
        print(f"    {'Total':<15} {total_pen:>7.1f}\n")

    # Value% distribution
    vp_vals = [b["value_pct"] for b in boxes]
    sf = VALUE_SWEET_FROM
    st = VALUE_SWEET_TO
    if vp_vals:
        below_100 = sum(1 for v in vp_vals if v < 100)
        in_100_sf = sum(1 for v in vp_vals if 100 <= v < sf)
        in_sweet = sum(1 for v in vp_vals if sf <= v <= st)
        in_st_130 = sum(1 for v in vp_vals if st < v <= 130)
        above_130 = sum(1 for v in vp_vals if v > 130)

        print(f"  {_BOLD}Value% Distribution ({n_boxes} boxes){_RESET}")
        print(f"    < 100%:     {below_100:>4}  ({below_100/n_boxes*100:>5.1f}%)  {_RED}penalty{_RESET}")
        print(f"    100-{sf}%:  {in_100_sf:>4}  ({in_100_sf/n_boxes*100:>5.1f}%)  {_YELLOW}penalty{_RESET}")
        print(f"    {sf}-{st}%:  {in_sweet:>4}  ({in_sweet/n_boxes*100:>5.1f}%)  {_GREEN}sweet spot{_RESET}")
        print(f"    {st}-130%:  {in_st_130:>4}  ({in_st_130/n_boxes*100:>5.1f}%)  {_YELLOW}penalty{_RESET}")
        print(f"    > 130%:     {above_130:>4}  ({above_130/n_boxes*100:>5.1f}%)  {_RED}penalty{_RESET}")
        print()

    # What-if sweet spot comparison
    if what_if:
        print(f"  {_BOLD}What-if Sweet Spot Analysis{_RESET}")
        print(f"    {'Range':<25} {'Avg Val Pen':>10} {'Score':>7} {'Delta':>7} {'In SS':>6} {'% SS':>6}")
        print(f"    {'─' * 65}")
        for w in what_if:
            delta_str = f"+{w['score_delta']:.1f}" if w['score_delta'] > 0 else f"{w['score_delta']:.1f}"
            hl = _GREEN if w['score_delta'] > 0 else (_DIM if w['score_delta'] == 0 else "")
            rst = _RESET if hl else ""
            print(f"    {hl}{w['label']:<25} {w['avg_value_penalty']:>10.2f} "
                  f"{w['composite_score']:>7.1f} {delta_str:>7} "
                  f"{w['boxes_in_sweet_spot']:>6} {w['pct_in_sweet_spot']:>5.1f}%{rst}")
        print()

    # Size tier comparison
    by_size = defaultdict(list)
    for o in offers:
        for b in o["boxes"]:
            by_size[b["tier"]].append(o["composite"]["score"])
    if by_size:
        # Per-box composite per size tier
        size_box_metrics = defaultdict(list)
        for b in boxes:
            size_box_metrics[b["tier"]].append(b)
        print(f"  {_BOLD}Size Tier Comparison{_RESET}")
        print(f"    {'Tier':<8} {'Boxes':>6} {'Mean V%':>8} {'Std V%':>7} {'AvgDiv':>7} {'AvgDup':>7}")
        print(f"    {'─' * 45}")
        for tier in ["small", "medium", "large"]:
            bxs = size_box_metrics.get(tier, [])
            if not bxs:
                continue
            nb = len(bxs)
            mvp = sum(b["value_pct"] for b in bxs) / nb
            svp = (sum((b["value_pct"] - mvp) ** 2 for b in bxs) / nb) ** 0.5
            mdiv = sum(b["diversity_score"] for b in bxs) / nb
            mdup = sum(b["fungible_dupes"] for b in bxs) / nb
            print(f"    {tier:<8} {nb:>6} {mvp:>7.1f}% {svp:>7.1f} {mdiv:>7.3f} {mdup:>7.2f}")
        print()

    # Pricing plausibility
    implausible = [o for o in offers if not o["pricing"]["plausible"]]
    if implausible:
        print(f"  {_BOLD}Pricing Anomalies{_RESET} ({len(implausible)} offers flagged)")
        for o in implausible[:10]:
            p = o["pricing"]
            print(f"    Offer {o['offer_id']} (Tier {o['data_tier']}): "
                  f"{p['anomaly_count']}/{p['total_boxes']} boxes outside 100-130%")
            for a in p["anomalies"][:3]:
                fits = f" (fits {a['fits_other_tier']})" if a["fits_other_tier"] else ""
                print(f"      {a['box_name']}: {a['value_pct']:.1f}% as {a['tier']}{fits}")
        print()
    else:
        print(f"  {_GREEN}Pricing: all offers plausible{_RESET}\n")

    # Data quality
    high_missing = [o for o in offers if o["missing_item_pct"] > 10]
    if high_missing:
        print(f"  {_BOLD}Data Quality Notes{_RESET}")
        for o in sorted(high_missing, key=lambda x: -x["missing_item_pct"]):
            print(f"    Offer {o['offer_id']} (Tier {o['data_tier']}): "
                  f"{o['missing_item_pct']:.0f}% missing items ({o['missing_items']} items)")
        print()

    # Per-offer table
    print(f"  {_BOLD}Per-Offer Summary{_RESET}")
    print(f"    {'Offer':>5} {'Tier':>4} {'Boxes':>5} {'Score':>6} "
          f"{'ValPen':>7} {'SIPen':>7} {'GCPen':>7} {'DivPen':>7} {'MVSPen':>7} {'SFPen':>7} {'Miss%':>6} {'Price':>5}")
    print(f"    {'─' * 92}")
    for o in offers:
        c = o["composite"]
        price_ok = "ok" if o["pricing"]["plausible"] else "FLAG"
        color = _GREEN if c["score"] >= 70 else (_YELLOW if c["score"] >= 50 else _RED)
        print(f"    {o['offer_id']:>5} {o['data_tier']:>4} {o['box_count']:>5} "
              f"{color}{c['score']:>6.1f}{_RESET} "
              f"{c['value_pen']:>7.1f} {c['si_pen']:>7.1f} {c['gc_pen']:>7.1f} "
              f"{c['diversity_pen']:>7.1f} {c['mvs_pen']:>7.1f} {c['sf_pen']:>7.1f} "
              f"{o['missing_item_pct']:>5.1f}% {price_ok:>5}")
    print()


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def write_json_report(offers: list[dict], boxes: list[dict], what_if: list[dict],
                      correlations: dict, by_size: dict, by_data_tier: dict,
                      aggregate: dict):
    """Write structured JSON report to diagnostics/report.json."""
    config = {
        "sweet_spot": [VALUE_SWEET_FROM, VALUE_SWEET_TO],
        "penalty_exponent": VALUE_PENALTY_EXPONENT,
        "same_item_multiplier": SAME_ITEM_MULTIPLIER,
        "group_concentration_multiplier": GROUP_CONCENTRATION_MULTIPLIER,
        "group_qty_exponent": GROUP_QTY_EXPONENT,
        "max_value_share_threshold": MAX_VALUE_SHARE_THRESHOLD,
        "max_value_share_multiplier": MAX_VALUE_SHARE_MULTIPLIER,
        "size_floor_multiplier": SIZE_FLOOR_MULTIPLIER,
        "diversity_penalty_multiplier": DIVERSITY_PENALTY_MULTIPLIER,
        "pref_violation_penalty": PREF_VIOLATION_PENALTY,
        "max_composite_score": MAX_COMPOSITE_SCORE,
        "diversity_weights": DIVERSITY_WEIGHTS,
        "box_tiers": {k: {"price": v["price"], "target": v["target_value"]}
                      for k, v in BOX_TIERS.items()},
    }

    # Simplify per-offer data for JSON
    per_offer = []
    for o in offers:
        offer_boxes = []
        for b in o["boxes"]:
            offer_boxes.append({
                "box_name": b["box_name"],
                "tier": b["tier"],
                "value_pct": round(b["value_pct"], 1),
                "total_value": b["total_value"],
                "target_value": b["target_value"],
                "unique_items": b["unique_items"],
                "diversity_score": round(b["diversity_score"], 3),
                "fungible_dupes": b["fungible_dupes"],
                "penalties": {
                    "value": round(b["value_penalty"], 2),
                    "same_item": round(b["same_item_penalty"], 2),
                    "group_concentration": round(b["group_concentration_penalty"], 2),
                    "diversity": round(b["diversity_penalty"], 2),
                    "max_value_share": round(b["max_value_share_penalty"], 2),
                    "size_floor": round(b["size_floor_penalty"], 2),
                    "pref": round(b.get("pref_penalty", 0), 2),
                },
            })
        per_offer.append({
            "offer_id": o["offer_id"],
            "data_tier": o["data_tier"],
            "box_count": o["box_count"],
            "missing_item_pct": o["missing_item_pct"],
            "composite": {k: round(v, 2) for k, v in o["composite"].items()},
            "pricing_plausible": o["pricing"]["plausible"],
            "pricing_anomalies": o["pricing"]["anomaly_count"],
            "boxes": offer_boxes,
        })

    report = {
        "config": config,
        "aggregate": aggregate,
        "by_size_tier": by_size,
        "by_data_tier": by_data_tier,
        "what_if_sweet_spots": what_if,
        "correlations": correlations,
        "offers": per_offer,
    }

    out_path = DIAGNOSTICS_DIR / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report: {out_path}")


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def generate_plots(offers: list[dict], boxes: list[dict], what_if: list[dict],
                   correlations: dict):
    """Generate diagnostic PNG plots to diagnostics/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [SKIP] matplotlib not installed, skipping plots")
        return

    plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (10, 6)})

    _plot_penalty_breakdown(offers, plt)
    _plot_composite_distribution(offers, plt)
    _plot_value_distribution(boxes, plt)
    _plot_diversity_distribution(boxes, plt)
    _plot_dupe_analysis(boxes, plt)
    _plot_correlation_heatmap(correlations, plt)
    _plot_size_tier_comparison(boxes, plt)
    _plot_offer_timeline(offers, plt, mpatches)
    _plot_what_if_sweet_spot(what_if, plt)
    _plot_pricing_scatter(offers, plt, mpatches)

    print(f"  Plots saved to {DIAGNOSTICS_DIR}/")


def _plot_penalty_breakdown(offers, plt):
    """1. Horizontal stacked bar showing avg penalty by dimension."""
    if not offers:
        return
    dims = ["Value", "Same-item", "Group", "Diversity", "Max-val", "Size", "Preference"]
    keys = ["value_pen", "si_pen", "gc_pen", "diversity_pen", "mvs_pen", "sf_pen", "pref_pen"]
    n = len(offers)
    avgs = [sum(o["composite"].get(k, 0) for o in offers) / n for k in keys]
    colors = ["#e74c3c", "#e67e22", "#d35400", "#3498db", "#f39c12", "#16a085", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 3))
    left = 0
    for dim, avg, color in zip(dims, avgs, colors):
        ax.barh(0, avg, left=left, color=color, label=f"{dim}: {avg:.1f}")
        if avg > 1:
            ax.text(left + avg / 2, 0, f"{avg:.1f}", ha="center", va="center", fontsize=9, color="white")
        left += avg
    ax.set_xlim(0, max(left * 1.1, 1))
    ax.set_yticks([])
    ax.set_xlabel("Penalty points")
    ax.set_title("Average Penalty Breakdown by Dimension")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "penalty_breakdown.png")
    plt.close()


def _plot_composite_distribution(offers, plt):
    """2. Histogram of per-offer composite scores."""
    scores = [o["composite"]["score"] for o in offers]
    if not scores:
        return
    fig, ax = plt.subplots()
    ax.hist(scores, bins=20, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(sum(scores) / len(scores), color="#e74c3c", linestyle="--", label=f"Mean: {sum(scores)/len(scores):.1f}")
    for p in [25, 50, 75]:
        val = _percentiles(scores, [p])[p]
        ax.axvline(val, color="#2ecc71", linestyle=":", alpha=0.7)
    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Count")
    ax.set_title("Composite Score Distribution (per-offer)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "composite_distribution.png")
    plt.close()


def _plot_value_distribution(boxes, plt):
    """3. Value% histogram with sweet spot region shaded and penalty curve."""
    vp_vals = [b["value_pct"] for b in boxes]
    if not vp_vals:
        return
    fig, ax1 = plt.subplots()
    ax1.hist(vp_vals, bins=40, range=(80, 160), color="#3498db", edgecolor="white", alpha=0.7)
    ax1.axvspan(VALUE_SWEET_FROM, VALUE_SWEET_TO, alpha=0.2, color="#2ecc71", label="Sweet spot")
    ax1.set_xlabel("Value % of target")
    ax1.set_ylabel("Box count")
    ax1.set_title("Value% Distribution with Penalty Curve")

    # Overlay penalty curve
    ax2 = ax1.twinx()
    xs = [x / 2 for x in range(160, 340)]
    ys = [value_penalty(x) for x in xs]
    ax2.plot(xs, ys, color="#e74c3c", linewidth=2, alpha=0.7, label="Penalty")
    ax2.set_ylabel("Penalty", color="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "value_distribution.png")
    plt.close()


def _plot_diversity_distribution(boxes, plt):
    """4. Diversity score histogram split by size tier."""
    fig, ax = plt.subplots()
    tier_colors = {"small": "#e74c3c", "medium": "#3498db", "large": "#2ecc71"}
    for tier in ["small", "medium", "large"]:
        vals = [b["diversity_score"] for b in boxes if b["tier"] == tier]
        if vals:
            ax.hist(vals, bins=20, range=(0, 1), alpha=0.5, color=tier_colors[tier],
                    label=f"{tier} ({len(vals)})")
    ax.set_xlabel("Diversity Score")
    ax.set_ylabel("Count")
    ax.set_title("Diversity Score Distribution by Size Tier")
    ax.legend()
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "diversity_distribution.png")
    plt.close()


def _plot_dupe_analysis(boxes, plt):
    """5. Bar chart of dupe frequency."""
    dupe_counts = defaultdict(int)
    for b in boxes:
        dupes = b["fungible_dupes"]
        dupe_counts[dupes] += 1

    if not dupe_counts:
        return

    keys_sorted = sorted(dupe_counts.keys())
    fig, ax = plt.subplots()
    ax.bar([str(k) for k in keys_sorted], [dupe_counts[k] for k in keys_sorted],
           color="#e67e22", edgecolor="white")
    ax.set_xlabel("Fungible Dupes per Box")
    ax.set_ylabel("Number of Boxes")
    ax.set_title("Fungible Duplicate Distribution")
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "dupe_analysis.png")
    plt.close()


def _plot_correlation_heatmap(correlations, plt):
    """6. Correlation heatmap of penalty dimensions."""
    if not correlations:
        return
    fields = list(correlations.keys())
    n = len(fields)
    matrix = [[correlations[f1][f2] for f2 in fields] for f1 in fields]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [f.replace("_penalty", "").replace("_", " ") for f in fields]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Penalty Component Correlations")
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "correlation_heatmap.png")
    plt.close()


def _plot_size_tier_comparison(boxes, plt):
    """7. Box plots of value% by size tier."""
    tier_data = defaultdict(list)
    for b in boxes:
        tier_data[b["tier"]].append(b["value_pct"])

    tiers = ["small", "medium", "large"]
    data = [tier_data.get(t, []) for t in tiers]
    if not any(data):
        return

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, labels=tiers, patch_artist=True)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhspan(VALUE_SWEET_FROM, VALUE_SWEET_TO, alpha=0.15, color="#2ecc71")
    ax.set_ylabel("Value % of target")
    ax.set_title("Value% by Size Tier")
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "size_tier_comparison.png")
    plt.close()


def _plot_offer_timeline(offers, plt, mpatches):
    """8. Per-offer composite score vs offer_id, coloured by data tier."""
    if not offers:
        return
    tier_colors = {"A": "#3498db", "B": "#2ecc71", "C": "#e67e22", "D": "#e74c3c"}
    fig, ax = plt.subplots()
    for o in offers:
        color = tier_colors.get(o["data_tier"], "#999999")
        ax.scatter(o["offer_id"], o["composite"]["score"], color=color, s=40, zorder=3)

    handles = [mpatches.Patch(color=c, label=f"Tier {t}")
               for t, c in tier_colors.items()
               if any(o["data_tier"] == t for o in offers)]
    ax.legend(handles=handles)
    ax.set_xlabel("Offer ID")
    ax.set_ylabel("Composite Score")
    ax.set_title("Composite Score by Offer (coloured by data tier)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "offer_timeline.png")
    plt.close()


def _plot_what_if_sweet_spot(what_if, plt):
    """9. Bar chart comparing composite scores under different sweet spot ranges."""
    if not what_if:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [w["label"] for w in what_if]
    scores = [w["composite_score"] for w in what_if]
    colors = ["#2ecc71" if w["score_delta"] > 0 else ("#e74c3c" if w["score_delta"] < 0 else "#3498db")
              for w in what_if]
    bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Composite Score")
    ax.set_title("What-if Sweet Spot Analysis")
    for bar, w in zip(bars, what_if):
        delta = w["score_delta"]
        label = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                label, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "what_if_sweet_spot.png")
    plt.close()


def _plot_pricing_scatter(offers, plt, mpatches):
    """10. Per-offer median value% vs offer_id, flagging implausible offers."""
    if not offers:
        return
    fig, ax = plt.subplots()
    for o in offers:
        vp_vals = [b["value_pct"] for b in o["boxes"]]
        median_vp = sorted(vp_vals)[len(vp_vals) // 2] if vp_vals else 0
        color = "#e74c3c" if not o["pricing"]["plausible"] else "#3498db"
        marker = "x" if not o["pricing"]["plausible"] else "o"
        ax.scatter(o["offer_id"], median_vp, color=color, marker=marker, s=40, zorder=3)

    ax.axhspan(100, 130, alpha=0.1, color="#2ecc71", label="100-130% range")
    ax.axhspan(VALUE_SWEET_FROM, VALUE_SWEET_TO, alpha=0.2, color="#2ecc71", label="Sweet spot")
    handles = [
        mpatches.Patch(color="#3498db", label="Plausible"),
        mpatches.Patch(color="#e74c3c", label="Flagged"),
    ]
    ax.legend(handles=handles)
    ax.set_xlabel("Offer ID")
    ax.set_ylabel("Median Value %")
    ax.set_title("Pricing Plausibility by Offer")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DIAGNOSTICS_DIR / "pricing_scatter.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scoring diagnostic analysis for historical allocations")
    parser.add_argument("--only-offers", type=str, default=None,
                        help="Comma-separated offer IDs/ranges (e.g. '55-63,64-106')")
    parser.add_argument("--only-tier", type=str, default=None,
                        help="Only process offers from this tier (A/B/C/D)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating PNG plots")
    args = parser.parse_args()

    # Discover offers
    all_ids = discover_offer_ids()

    if args.only_offers:
        requested = _parse_only_offers(args.only_offers)
        offer_ids = sorted(set(all_ids) & requested)
    elif args.only_tier:
        tier = args.only_tier.upper()
        offer_ids = [oid for oid in all_ids if offer_tier(oid) == tier]
    else:
        offer_ids = all_ids

    if not offer_ids:
        print("No matching offers found.")
        sys.exit(1)

    tier_counts = defaultdict(int)
    for oid in offer_ids:
        tier_counts[offer_tier(oid)] += 1
    tier_summary = ", ".join(f"{t}:{c}" for t, c in sorted(tier_counts.items()))
    print(f"  Discovered {len(offer_ids)} offers ({tier_summary})")

    # Create output directory
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load summary.json for metadata
    summary = load_summary()

    # Collect data
    print("  Collecting offer data...")
    offers = collect_all_offers(offer_ids, summary)
    print(f"  Processed {len(offers)} offers successfully")

    if not offers:
        print("No offers could be processed.")
        sys.exit(1)

    # Flatten boxes
    boxes = _all_boxes(offers)

    # Analysis
    aggregate = compute_aggregate_stats(boxes)
    by_size = compute_tier_breakdown(boxes, "tier")
    by_data_tier = compute_tier_breakdown(boxes, "data_tier")
    what_if = compute_what_if_sweet_spots(boxes)
    correlations = compute_correlations(boxes)

    # Print summary
    print_summary(offers, boxes, what_if)

    # Write JSON report
    write_json_report(offers, boxes, what_if, correlations, by_size, by_data_tier, aggregate)

    # Generate plots
    if not args.no_plots:
        generate_plots(offers, boxes, what_if, correlations)

    print(f"\n  Done. {len(offers)} offers, {len(boxes)} boxes analysed.")


if __name__ == "__main__":
    main()
