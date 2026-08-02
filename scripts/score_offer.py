#!/usr/bin/env python3
"""
Run all strategies against a single offer and print a leaderboard.

Usage:
    python3 scripts/score_offer.py 106 historical/offer_106_shopping_list.xlsx
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from allocator.allocator import allocate, build_boxes_from_db
from allocator.config import (
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.strategies import list_strategies
from compare import (
    build_item_lookup,
    compute_available_tags,
    compute_box_metrics,
    compute_composite_score,
)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 score_offer.py <offer_id> <xlsx_path>")
        sys.exit(1)

    offer_id = int(sys.argv[1])
    xlsx_path = Path(sys.argv[2])

    strategies = list_strategies(include_baselines=True)
    item_lookup = build_item_lookup(offer_id)
    avail_tags = compute_available_tags(item_lookup)

    print(f"Offer {offer_id}: {len(item_lookup)} items in DB")
    print(f"XLSX: {xlsx_path.name}")
    print(f"Strategies: {', '.join(strategies)}")
    print()

    results = {}
    all_box_details = {}
    dw_allocations = None  # for local-search bootstrap

    # Run strategies in a sensible order: discard-worst before local-search
    ordered = [s for s in strategies if s not in ("local-search",)]
    ordered.append("local-search")

    for strat in ordered:
        t0 = time.monotonic()
        try:
            kwargs = {"strategy": strat}
            if strat == "local-search" and dw_allocations is not None:
                kwargs["bootstrap_allocations"] = dw_allocations
            result = allocate(offer_id, xlsx_path, **kwargs)
        except Exception as e:
            print(f"  {strat}: FAILED — {e}")
            continue
        elapsed = time.monotonic() - t0

        # Save discard-worst allocations for local-search bootstrap
        if strat == "discard-worst":
            dw_allocations = [dict(box.allocations) for box in result.boxes]

        # Compute metrics for each box
        metrics = []
        box_details = []
        for box in result.boxes:
            m = compute_box_metrics(
                box.name, box.allocations, item_lookup, box.tier,
                preference=box.preference, available_tags=avail_tags,
            )
            if m:
                metrics.append(m)
                box_details.append((box, m))

        comp = compute_composite_score(metrics)
        results[strat] = (comp, metrics, box_details, result, elapsed)
        print(f"  {strat:<20} score={comp['score']:>6.1f}  ({elapsed:.1f}s)")

    # Leaderboard
    print(f"\n{'='*80}")
    print(f"  OFFER {offer_id} — STRATEGY LEADERBOARD")
    print(f"{'='*80}")
    print(f"  {'Rank':<6} {'Strategy':<20} {'Score':>7} {'Value':>8} {'GrpQty':>8} "
          f"{'Diver':>8} {'Fair':>8} {'Pref':>8} {'Desir':>8}")
    print(f"  {'-'*82}")

    ranked = sorted(results.items(), key=lambda x: x[1][0]["score"], reverse=True)
    for i, (name, (comp, metrics, _, _, elapsed)) in enumerate(ranked, 1):
        print(f"  {i:<6} {name:<20} {comp['score']:>7.1f} "
              f"{-comp['value_pen']:>+8.1f} {-comp['gq_pen']:>+8.1f} "
              f"{-comp['diversity_pen']:>+8.1f} {-comp['fair_pen']:>+8.1f} "
              f"{-comp['pref_pen']:>+8.1f} {-comp.get('desir_pen', 0):>+8.1f}")

    print()
    print(f"  Score = 100 - penalties.  Higher is better.")
    print(f"  Value: {VALUE_SWEET_FROM}-{VALUE_SWEET_TO}% of price sweet spot.  Dupes: weighted fungible overlap.")
    print(f"  Diver: coverage of sub-cat/usage/colour/shape.  Fair: stddev of value%.")

    # Per-box detail for the top strategy
    best_name, (best_comp, best_metrics, best_details, best_result, _) = ranked[0]
    print(f"\n{'='*80}")
    print(f"  BEST STRATEGY: {best_name} (score {best_comp['score']:.1f})")
    print(f"{'='*80}")
    print(f"  {'Box':<35} {'Tier':<7} {'Value':>8} {'Target':>8} {'%Tgt':>6} "
          f"{'Items':>5} {'Fr%':>5} {'Diver':>5} {'FDup':>4} {'BDup':>4} {'Pref':>4}")
    print(f"  {'-'*97}")

    for box, m in best_details:
        print(f"  {m['box_name'][:34]:<35} {m['tier']:<7} "
              f"${m['total_value']/100:>7.2f} ${m['target_value']/100:>7.2f} "
              f"{m['value_pct']:>5.1f}% {m['unique_items']:>5} "
              f"{m['fruit_pct']:>4.1f}% {m['diversity_score']:>5.2f} "
              f"{m['fungible_dupes']:>4} {m['bad_dupes']:>4} {m['pref_violations']:>4}")

    # Stock and charity summary for best strategy
    stock_value = sum(
        best_result.items[iid].price * qty
        for iid, qty in best_result.stock.items()
        if iid in best_result.items
    )
    stock_items = sum(1 for q in best_result.stock.values() if q > 0)
    print(f"\n  Stock: {stock_items} items, ${stock_value/100:.2f}")
    for charity in best_result.charity:
        cv = sum(best_result.items[iid].price * qty
                 for iid, qty in charity.allocations.items()
                 if iid in best_result.items)
        ci = sum(1 for q in charity.allocations.values() if q > 0)
        print(f"  {charity.name}: {ci} items, ${cv/100:.2f} (target ${charity.target_value/100:.2f})")

    # Value distribution for best strategy
    sf, st = VALUE_SWEET_FROM, VALUE_SWEET_TO
    _ss = f"{sf}-{st}%"
    buckets = [
        (f"<{sf}%",
         lambda v, lo=sf: v < lo),
        (_ss,
         lambda v, lo=sf, hi=st: lo <= v < hi),
        (f"{st}-130%",
         lambda v, lo=st: lo <= v < 130),
        (">=130%",
         lambda v: v >= 130),
    ]

    n_total = len(best_metrics)
    print(f"\n  Value distribution ({n_total} boxes):")
    for label, test in buckets:
        count = sum(1 for m in best_metrics if test(m["value_pct"]))
        marker = " <-- sweet spot" if label == _ss else ""
        print(f"    {label:>10}: {count:>2} / {n_total}{marker}")

    print()


if __name__ == "__main__":
    main()
