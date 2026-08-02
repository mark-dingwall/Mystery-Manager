#!/usr/bin/env python3
"""
Analyze packer survey responses against the scoring function.

Loads scenarios JSON + responses JSON, computes marginal penalty deltas
for each candidate, and measures Spearman rank correlation between the
scoring function's item ranking and the packer's ratings.

Usage:
    python3 scripts/process_survey_results.py responses.json
    python3 scripts/process_survey_results.py responses.json --detail
    python3 scripts/process_survey_results.py responses.json --scenarios path/to/scenarios.json
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from allocator.config import BOX_TIERS, GROUP_ALLOWANCES
from allocator.tuning import compute_marginal_deltas, default_params
from scripts.extract_features import extract_box_features
from compare import build_item_lookup, compute_available_tags, read_xlsx_pack_overrides


# ---------------------------------------------------------------------------
# Spearman rank correlation (pure Python, no scipy dependency)
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Assign ranks with average tie-breaking."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation coefficient. Returns None if < 3 items."""
    n = len(x)
    if n < 3 or n != len(y):
        return None
    rx = _rank(x)
    ry = _rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = (sum((r - mean_rx) ** 2 for r in rx)) ** 0.5
    std_y = (sum((r - mean_ry) ** 2 for r in ry)) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


# ---------------------------------------------------------------------------
# Feature recomputation for marginal deltas
# ---------------------------------------------------------------------------

def _rebuild_features_for_scenario(scenario: dict, item_lookup: dict, avail_tags: dict):
    """Recompute base and candidate features for a scenario.

    Returns (base_features, candidate_features_list) or (None, None) if
    item data is missing.
    """
    tier = scenario["box"]["tier"]
    offer_id = scenario["source_offer_id"]

    # Build base box allocations
    base_allocs = {}
    for item in scenario["box"]["current_items"]:
        iid = item["item_id"]
        if iid in item_lookup:
            base_allocs[iid] = item["qty"]

    if not base_allocs:
        return None, None

    base_feat = extract_box_features(
        "survey_base", base_allocs, item_lookup, tier, avail_tags, offer_id
    )
    if not base_feat:
        return None, None

    candidate_feats = []
    for c in scenario["candidates"]:
        cid = c["item_id"]
        if cid not in item_lookup:
            candidate_feats.append(None)
            continue
        # Add 1 unit of candidate to base box
        augmented = dict(base_allocs)
        augmented[cid] = augmented.get(cid, 0) + 1
        cf = extract_box_features(
            "survey_candidate", augmented, item_lookup, tier, avail_tags, offer_id
        )
        candidate_feats.append(cf)

    return base_feat, candidate_feats


# ---------------------------------------------------------------------------
# Test-retest reliability
# ---------------------------------------------------------------------------

def _analyze_test_retest(
    scenarios_data: dict,
    responses: dict,
) -> dict:
    """Measure packer consistency from repeated scenarios.

    For each repeat pair where both original and repeat were answered,
    compares the ratings on the same candidate items. Reports mean absolute
    rating difference and within-pair rank correlation.
    """
    repeat_pairs = scenarios_data.get("repeat_pairs", [])
    if not repeat_pairs:
        return {"n_pairs": 0, "n_completed": 0}

    pair_results = []
    for pair in repeat_pairs:
        orig_id = pair["original"]
        repeat_id = pair["repeat"]

        if orig_id not in responses or repeat_id not in responses:
            continue

        orig_resp = responses[orig_id]
        repeat_resp = responses[repeat_id]

        # Both marked none_fit — agreement on escape hatch
        if orig_resp.get("none_fit") and repeat_resp.get("none_fit"):
            pair_results.append({
                "original": orig_id,
                "repeat": repeat_id,
                "none_fit_agree": True,
            })
            continue

        # One none_fit, one not — disagreement
        if orig_resp.get("none_fit") != repeat_resp.get("none_fit"):
            pair_results.append({
                "original": orig_id,
                "repeat": repeat_id,
                "none_fit_agree": False,
            })
            continue

        orig_ratings = orig_resp.get("ratings", {})
        repeat_ratings = repeat_resp.get("ratings", {})

        # Match candidates rated in both presentations
        common_ids = sorted(set(orig_ratings.keys()) & set(repeat_ratings.keys()))
        if len(common_ids) < 2:
            continue

        orig_vals = [float(orig_ratings[cid]) for cid in common_ids]
        repeat_vals = [float(repeat_ratings[cid]) for cid in common_ids]
        diffs = [abs(o - r) for o, r in zip(orig_vals, repeat_vals)]
        rho = spearman_rho(orig_vals, repeat_vals)

        pair_results.append({
            "original": orig_id,
            "repeat": repeat_id,
            "n_common": len(common_ids),
            "mean_abs_diff": sum(diffs) / len(diffs),
            "max_diff": max(diffs),
            "rank_correlation": rho,
        })

    valid = [p for p in pair_results if "mean_abs_diff" in p]
    valid_rhos = [p["rank_correlation"] for p in valid if p["rank_correlation"] is not None]

    return {
        "n_pairs": len(repeat_pairs),
        "n_completed": len(pair_results),
        "n_valid": len(valid),
        "mean_abs_rating_diff": sum(p["mean_abs_diff"] for p in valid) / len(valid) if valid else None,
        "mean_rank_correlation": sum(valid_rhos) / len(valid_rhos) if valid_rhos else None,
        "pair_details": pair_results,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(scenarios_path: str, responses_path: str, detail: bool = False):
    """Run the full analysis pipeline."""
    with open(scenarios_path) as f:
        scenarios_data = json.load(f)
    with open(responses_path) as f:
        responses_data = json.load(f)

    scenarios = {s["id"]: s for s in scenarios_data["scenarios"]}
    responses = responses_data.get("responses", {})

    # Repeat scenario IDs — excluded from main analysis, handled separately
    repeat_ids = {p["repeat"] for p in scenarios_data.get("repeat_pairs", [])}

    params = default_params()

    # Cache item lookups per offer
    offer_lookups: dict[int, tuple] = {}

    results = []
    skipped = 0

    for sid, resp in responses.items():
        if sid not in scenarios:
            skipped += 1
            continue

        # Skip repeats from main analysis (handled by test-retest)
        if sid in repeat_ids:
            continue

        scenario = scenarios[sid]

        # Skip none_fit scenarios (no meaningful ranking)
        if resp.get("none_fit", False):
            results.append({
                "scenario_id": sid,
                "type": scenario["type"],
                "dimension": scenario.get("target_dimension"),
                "rho": None,
                "none_fit": True,
                "n_rated": 0,
            })
            continue

        ratings = resp.get("ratings", {})
        if len(ratings) < 3:
            skipped += 1
            continue

        # Load offer data (cached)
        offer_id = scenario["source_offer_id"]
        if offer_id not in offer_lookups:
            try:
                pack_overrides = read_xlsx_pack_overrides(offer_id)
                il = build_item_lookup(offer_id, price_overrides=pack_overrides) if pack_overrides else build_item_lookup(offer_id)
                at = compute_available_tags(il)
                offer_lookups[offer_id] = (il, at)
            except Exception:
                offer_lookups[offer_id] = (None, None)

        item_lookup, avail_tags = offer_lookups[offer_id]
        if not item_lookup:
            skipped += 1
            continue

        base_feat, candidate_feats = _rebuild_features_for_scenario(
            scenario, item_lookup, avail_tags
        )
        if not base_feat:
            skipped += 1
            continue

        # Filter to candidates that have both ratings and valid features
        valid_indices = []
        for i, c in enumerate(scenario["candidates"]):
            cid = str(c["item_id"])
            if cid in ratings and candidate_feats[i] is not None:
                valid_indices.append(i)

        if len(valid_indices) < 3:
            skipped += 1
            continue

        # Compute marginal deltas
        valid_feats = [candidate_feats[i] for i in valid_indices]
        deltas = compute_marginal_deltas(base_feat, valid_feats, params)

        # Packer ratings (higher = better fit)
        packer_ratings = [
            ratings[str(scenario["candidates"][i]["item_id"])]
            for i in valid_indices
        ]

        # Correlation: positive delta = score improves = should correlate with high rating
        rho = spearman_rho(deltas, [float(r) for r in packer_ratings])

        results.append({
            "scenario_id": sid,
            "type": scenario["type"],
            "dimension": scenario.get("target_dimension"),
            "rho": rho,
            "none_fit": False,
            "n_rated": len(valid_indices),
            "deltas": deltas if detail else None,
            "ratings": packer_ratings if detail else None,
        })

    # Aggregate statistics
    valid_rhos = [r["rho"] for r in results if r["rho"] is not None]
    none_fit_count = sum(1 for r in results if r["none_fit"])

    overall = {
        "n_scenarios": len(results),
        "n_valid": len(valid_rhos),
        "n_none_fit": none_fit_count,
        "n_skipped": skipped,
        "mean_rho": sum(valid_rhos) / len(valid_rhos) if valid_rhos else None,
        "median_rho": sorted(valid_rhos)[len(valid_rhos) // 2] if valid_rhos else None,
    }

    # Per-dimension breakdown
    per_dimension = {}
    for r in results:
        dim = r["dimension"] or "general"
        if dim not in per_dimension:
            per_dimension[dim] = []
        if r["rho"] is not None:
            per_dimension[dim].append(r["rho"])

    dim_summary = {}
    for dim, rhos in per_dimension.items():
        if rhos:
            dim_summary[dim] = {
                "n": len(rhos),
                "mean_rho": sum(rhos) / len(rhos),
                "median_rho": sorted(rhos)[len(rhos) // 2],
            }

    # Disagreement scenarios (negative correlation)
    disagreements = [r["scenario_id"] for r in results if r["rho"] is not None and r["rho"] < -0.1]

    # Test-retest reliability
    test_retest = _analyze_test_retest(scenarios_data, responses)

    output = {
        "overall": overall,
        "per_dimension": dim_summary,
        "test_retest": test_retest,
        "disagreement_scenarios": disagreements,
    }
    if detail:
        output["scenario_details"] = results

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze packer survey responses vs scoring function"
    )
    parser.add_argument("responses", type=str, help="Path to responses JSON file")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Path to scenarios JSON (default: diagnostics/survey_scenarios.json)")
    parser.add_argument("--detail", action="store_true",
                        help="Include per-scenario deltas and ratings in output")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: diagnostics/survey_analysis.json)")
    args = parser.parse_args()

    scenarios_path = args.scenarios or str(
        Path(__file__).resolve().parent.parent / "diagnostics" / "survey_scenarios.json"
    )

    print(f"Loading scenarios from {scenarios_path}")
    print(f"Loading responses from {args.responses}")

    output = analyze(scenarios_path, args.responses, detail=args.detail)

    # Print summary
    o = output["overall"]
    print(f"\n{'=' * 60}")
    print(f"  SURVEY ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Scenarios: {o['n_scenarios']}  (valid: {o['n_valid']}, "
          f"none-fit: {o['n_none_fit']}, skipped: {o['n_skipped']})")
    if o["mean_rho"] is not None:
        print(f"  Mean Spearman rho:   {o['mean_rho']:.3f}")
        print(f"  Median Spearman rho: {o['median_rho']:.3f}")
    else:
        print("  No valid correlations computed.")

    if output["per_dimension"]:
        print(f"\n  Per-dimension breakdown:")
        for dim, stats in sorted(output["per_dimension"].items()):
            print(f"    {dim:20s}  n={stats['n']:3d}  mean={stats['mean_rho']:.3f}  "
                  f"median={stats['median_rho']:.3f}")

    tr = output.get("test_retest", {})
    if tr.get("n_completed", 0) > 0:
        print(f"\n  Test-retest reliability ({tr['n_completed']}/{tr['n_pairs']} pairs):")
        if tr.get("mean_abs_rating_diff") is not None:
            print(f"    Mean |Δrating|:    {tr['mean_abs_rating_diff']:.2f}  (on 1-7 scale)")
        if tr.get("mean_rank_correlation") is not None:
            print(f"    Mean rank corr:    {tr['mean_rank_correlation']:.3f}")
    elif tr.get("n_pairs", 0) > 0:
        print(f"\n  Test-retest: {tr['n_pairs']} pairs in scenarios but no completed pairs in responses")

    if output["disagreement_scenarios"]:
        print(f"\n  Disagreement scenarios ({len(output['disagreement_scenarios'])}):")
        for sid in output["disagreement_scenarios"][:10]:
            print(f"    {sid}")

    # Write output
    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "diagnostics" / "survey_analysis.json"
    )
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Written to {output_path}")


if __name__ == "__main__":
    main()
