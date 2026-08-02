#!/usr/bin/env python3
"""
Optuna-based parameter tuning for the composite scoring function.

Loads precomputed features from JSON (no DB needed), runs cross-validated
Bayesian optimisation to find scoring parameters where manual boxes score
high and synthetic bad boxes score low.

Usage:
    python3 scripts/tune_scoring.py                                # full run
    python3 scripts/tune_scoring.py --trials 200 --folds 3         # quick
    python3 scripts/tune_scoring.py --trials 3000 --repeats 25     # overnight stability
    python3 scripts/tune_scoring.py --features path/to/features.json
"""

import functools
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

try:
    import optuna
except ImportError:
    print("ERROR: optuna is required. Install with: pip install optuna", file=sys.stderr)
    sys.exit(1)

from allocator.tuning import compute_objective, default_params, rescore_offer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DEFAULT_FEATURES = (
    Path(__file__).resolve().parent.parent / "diagnostics" / "tuning_features.json"
)

_TUNING_BOUNDS_PATH = Path(__file__).resolve().parent.parent / "tuning_bounds.json"


@functools.lru_cache(maxsize=1)
def load_tuning_bounds() -> dict:
    """Load the Optuna search-space bounds from the gitignored tuning_bounds.json.

    The bounds are business-tuned ranges and are kept out of the tracked tree
    (see tuning_bounds.json.example for the structure and to bootstrap a copy).
    """
    if not _TUNING_BOUNDS_PATH.exists():
        raise FileNotFoundError(
            f"{_TUNING_BOUNDS_PATH.name} not found. Copy tuning_bounds.json.example to "
            f"{_TUNING_BOUNDS_PATH.name} and set your Optuna search ranges."
        )
    return json.loads(_TUNING_BOUNDS_PATH.read_text())


def load_features(path: Path) -> tuple[list[dict], list[dict]]:
    """Load features JSON, split into manual and synthetic lists."""
    with open(path) as f:
        data = json.load(f)

    features = data["features"]
    manual = [f for f in features if f["source"] == "manual"]
    synthetic = [f for f in features if f["source"] != "manual"]
    return manual, synthetic


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------

def make_cv_folds(manual: list[dict], k: int = 5) -> list[tuple[list[dict], list[dict]]]:
    """Split manual features into k folds at the offer level.

    Returns list of (train, test) feature lists.
    Stratified approximately by tier distribution.
    """
    # Group by offer
    by_offer: dict[int, list[dict]] = defaultdict(list)
    for f in manual:
        by_offer[f["offer_id"]].append(f)

    # Sort offers by dominant tier for approximate stratification
    def _dominant_tier(boxes):
        tier_counts = defaultdict(int)
        for b in boxes:
            tier_counts[b["tier"]] += 1
        return max(tier_counts, key=tier_counts.get)

    offer_ids = sorted(by_offer.keys())
    # Interleave by tier for stratification
    tier_groups: dict[str, list[int]] = defaultdict(list)
    for oid in offer_ids:
        tier_groups[_dominant_tier(by_offer[oid])].append(oid)

    # Round-robin assign to folds
    fold_offers: list[list[int]] = [[] for _ in range(k)]
    idx = 0
    for tier in sorted(tier_groups):
        for oid in tier_groups[tier]:
            fold_offers[idx % k].append(oid)
            idx += 1

    # Build train/test splits
    folds = []
    for fold_idx in range(k):
        test_ids = set(fold_offers[fold_idx])
        train = [f for f in manual if f["offer_id"] not in test_ids]
        test = [f for f in manual if f["offer_id"] in test_ids]
        folds.append((train, test))

    return folds


# ---------------------------------------------------------------------------
# Optuna trial
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """Suggest a complete params dict for one Optuna trial.

    Search-space bounds come from the gitignored tuning_bounds.json (business
    ranges), not hardcoded here.
    """
    b = load_tuning_bounds()
    sweet_from = trial.suggest_int("value_sweet_from", *b["value_sweet_from"])
    sweet_to = trial.suggest_int("value_sweet_to", sweet_from + 1, b["value_sweet_to_max"])

    w_subcat = trial.suggest_float("w_subcat", *b["w_subcat"])
    w_usage = trial.suggest_float("w_usage", *b["w_usage"])
    w_colour = trial.suggest_float("w_colour", *b["w_colour"])

    return {
        "value_sweet_from": sweet_from,
        "value_sweet_to": sweet_to,
        "value_penalty_exponent": trial.suggest_float("value_penalty_exponent", *b["value_penalty_exponent"]),
        "same_item_multiplier": trial.suggest_float("same_item_multiplier", *b["same_item_multiplier"]),
        "group_concentration_multiplier": trial.suggest_float("group_concentration_multiplier", *b["group_concentration_multiplier"]),
        "group_qty_exponent": trial.suggest_float("group_qty_exponent", *b["group_qty_exponent"]),
        "diversity_penalty_multiplier": trial.suggest_float("diversity_penalty_multiplier", *b["diversity_penalty_multiplier"]),
        "w_subcat": w_subcat,
        "w_usage": w_usage,
        "w_colour": w_colour,
        "max_value_share_threshold": trial.suggest_float("max_value_share_threshold", *b["max_value_share_threshold"]),
        "max_value_share_multiplier": trial.suggest_float("max_value_share_multiplier", *b["max_value_share_multiplier"]),
        "size_floor_multiplier": trial.suggest_float("size_floor_multiplier", *b["size_floor_multiplier"]),
        "pref_violation_penalty": trial.suggest_float("pref_violation_penalty", *b["pref_violation_penalty"]),
        "max_composite_score": 100.0,
        # Pass through from defaults — not tuned, but needed by rescore functions
        "group_allowances": default_params()["group_allowances"],
        "size_floor_targets": default_params()["size_floor_targets"],
    }


def make_objective(train: list[dict], bad: list[dict]):
    """Return an Optuna objective function closed over train/bad data."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = compute_objective(train, bad, params)
        if result is None:
            raise optuna.TrialPruned()
        return result
    return objective


# ---------------------------------------------------------------------------
# Median helper
# ---------------------------------------------------------------------------

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


# Keys that are passed through (not tuned, not aggregated)
_PASSTHROUGH_KEYS = {"group_allowances", "size_floor_targets", "max_composite_score"}


def _median_params_from_folds(fold_results: list[dict]) -> dict:
    """Compute median params across fold results."""
    param_keys = [k for k in fold_results[0]["best_params"]
                  if k not in _PASSTHROUGH_KEYS]
    median_p = {}
    for key in param_keys:
        median_p[key] = _median([fr["best_params"][key] for fr in fold_results])

    # Pass through non-tuned dict params from first fold
    for key in _PASSTHROUGH_KEYS:
        if key in fold_results[0]["best_params"]:
            median_p[key] = fold_results[0]["best_params"][key]

    for key in ["value_sweet_from", "value_sweet_to"]:
        if key in median_p:
            median_p[key] = int(round(median_p[key]))

    return median_p


# ---------------------------------------------------------------------------
# Single CV run
# ---------------------------------------------------------------------------

def run_cv_study(
    folds: list[tuple[list[dict], list[dict]]],
    synthetic: list[dict],
    n_trials: int,
    base_seed: int,
    quiet: bool = True,
) -> list[dict]:
    """Run one complete CV study across all folds. Returns fold_results list."""
    fold_results = []

    for fold_idx, (train, test) in enumerate(folds):
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=base_seed + fold_idx),
        )
        study.optimize(
            make_objective(train, synthetic),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best = study.best_trial
        best_params = suggest_params(best)

        train_obj = compute_objective(train, synthetic, best_params)
        test_obj = compute_objective(test, synthetic, best_params)

        test_offers: dict[int, list[dict]] = defaultdict(list)
        for f in test:
            test_offers[f["offer_id"]].append(f)
        test_offer_scores = {
            oid: rescore_offer(boxes, best_params)["score"]
            for oid, boxes in test_offers.items()
        }

        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train),
            "n_test": len(test),
            "best_value": best.value,
            "train_objective": train_obj,
            "test_objective": test_obj,
            "best_params": best_params,
            "test_offer_scores": {str(k): v for k, v in test_offer_scores.items()},
            "n_trials": len(study.trials),
            "n_pruned": len([t for t in study.trials
                            if t.state == optuna.trial.TrialState.PRUNED]),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune scoring parameters via Optuna")
    parser.add_argument("--features", type=str, default=None,
                        help=f"Features JSON path (default: {_DEFAULT_FEATURES})")
    parser.add_argument("--trials", type=int, default=800,
                        help="Trials per fold (default: 800)")
    parser.add_argument("--folds", type=int, default=5,
                        help="CV folds (default: 5)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Independent repeats with different seeds (default: 1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: diagnostics/tuning_results.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress Optuna progress logs")
    args = parser.parse_args()

    # Suppress Optuna's per-trial logs (very noisy with many trials/repeats)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.quiet:
        logging.getLogger(__name__).setLevel(logging.WARNING)

    features_path = Path(args.features) if args.features else _DEFAULT_FEATURES
    if not features_path.exists():
        print(f"ERROR: Features file not found: {features_path}", file=sys.stderr)
        print("Run scripts/extract_features.py first.", file=sys.stderr)
        sys.exit(1)

    manual, synthetic = load_features(features_path)
    log.info(f"Loaded {len(manual)} manual + {len(synthetic)} synthetic features")

    folds = make_cv_folds(manual, k=args.folds)
    log.info(f"Created {args.folds} CV folds")

    n_repeats = args.repeats
    total_tasks = n_repeats * args.folds * args.trials
    log.info(f"Running {n_repeats} repeat(s) x {args.folds} folds x {args.trials} trials "
             f"= {total_tasks:,} total trials")

    t0 = time.monotonic()
    repeat_results = []

    for rep_idx in range(n_repeats):
        rep_t0 = time.monotonic()
        base_seed = rep_idx * 1000

        fold_results = run_cv_study(folds, synthetic, args.trials, base_seed)

        median_p = _median_params_from_folds(fold_results)

        train_scores = [fr["train_objective"] for fr in fold_results
                        if fr["train_objective"] is not None]
        test_scores = [fr["test_objective"] for fr in fold_results
                       if fr["test_objective"] is not None]

        rep_summary = {
            "repeat": rep_idx,
            "base_seed": base_seed,
            "median_params": median_p,
            "mean_train_objective": sum(train_scores) / len(train_scores) if train_scores else None,
            "mean_test_objective": sum(test_scores) / len(test_scores) if test_scores else None,
            "fold_results": fold_results,
            "elapsed_s": round(time.monotonic() - rep_t0, 1),
        }
        repeat_results.append(rep_summary)

        # Progress line
        elapsed = time.monotonic() - t0
        remaining = elapsed / (rep_idx + 1) * (n_repeats - rep_idx - 1)
        train_str = f"{rep_summary['mean_train_objective']:.0f}" if rep_summary['mean_train_objective'] is not None else "n/a"
        test_str = f"{rep_summary['mean_test_objective']:.0f}" if rep_summary['mean_test_objective'] is not None else "n/a"
        print(f"  Repeat {rep_idx + 1:>3}/{n_repeats}: "
              f"train={train_str:>7}  test={test_str:>7}  "
              f"({rep_summary['elapsed_s']:.0f}s, ~{remaining:.0f}s remaining)")

    total_elapsed = time.monotonic() - t0

    # -----------------------------------------------------------------------
    # Aggregate across repeats
    # -----------------------------------------------------------------------

    # Collect tuned param keys (exclude passthrough)
    all_param_keys = [k for k in repeat_results[0]["median_params"]
                      if k not in _PASSTHROUGH_KEYS]

    # Compute overall median + IQR for each parameter
    param_stability = {}
    final_params = {}
    for key in all_param_keys:
        vals = [r["median_params"][key] for r in repeat_results]
        med = _median(vals)
        q25 = _percentile(vals, 25)
        q75 = _percentile(vals, 75)
        iqr = q75 - q25
        # Relative IQR: IQR / |median| (0 = perfectly stable)
        rel_iqr = iqr / abs(med) if abs(med) > 1e-9 else 0.0
        final_params[key] = med
        param_stability[key] = {
            "median": med, "q25": q25, "q75": q75, "iqr": iqr,
            "rel_iqr": round(rel_iqr, 4),
            "min": min(vals), "max": max(vals),
        }

    # Passthrough params
    for key in _PASSTHROUGH_KEYS:
        if key in repeat_results[0]["median_params"]:
            final_params[key] = repeat_results[0]["median_params"][key]

    # Integer params
    for key in ["value_sweet_from", "value_sweet_to"]:
        if key in final_params:
            final_params[key] = int(round(final_params[key]))

    # Aggregate scores
    all_train = [r["mean_train_objective"] for r in repeat_results
                 if r["mean_train_objective"] is not None]
    all_test = [r["mean_test_objective"] for r in repeat_results
                if r["mean_test_objective"] is not None]

    output = {
        "n_repeats": n_repeats,
        "n_folds": args.folds,
        "n_trials_per_fold": args.trials,
        "total_trials": n_repeats * args.folds * args.trials,
        "total_elapsed_s": round(total_elapsed, 1),
        "mean_train_objective": sum(all_train) / len(all_train) if all_train else None,
        "mean_test_objective": sum(all_test) / len(all_test) if all_test else None,
        "overfit_gap": (
            (sum(all_train) / len(all_train) - sum(all_test) / len(all_test))
            if all_train and all_test else None
        ),
        "final_params": final_params,
        "param_stability": param_stability,
        "default_params": default_params(),
        "repeat_results": repeat_results if n_repeats <= 10 else [
            # For large repeat counts, only store summaries (not full fold data)
            {k: v for k, v in r.items() if k != "fold_results"}
            for r in repeat_results
        ],
    }

    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "diagnostics" / "tuning_results.json"
    )
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(output, fp, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------

    print(f"\n{'='*70}")
    print(f"  TUNING RESULTS")
    print(f"{'='*70}")
    print(f"  Repeats: {n_repeats}, Folds: {args.folds}, Trials/fold: {args.trials}")
    print(f"  Total trials: {output['total_trials']:,}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    if output["mean_train_objective"] is not None:
        print(f"  Mean train objective: {output['mean_train_objective']:.2f}")
    if output["mean_test_objective"] is not None:
        print(f"  Mean test objective:  {output['mean_test_objective']:.2f}")
    if output["overfit_gap"] is not None:
        print(f"  Overfit gap:          {output['overfit_gap']:.2f}")

    # Parameter table with stability
    print(f"\n  {'Parameter':<35} {'Median':>10} {'IQR':>10} {'Rel IQR':>8}  Stability")
    print(f"  {'-'*78}")

    for key in sorted(param_stability):
        s = param_stability[key]
        # Classify stability
        if s["rel_iqr"] < 0.05:
            grade = "strong"
        elif s["rel_iqr"] < 0.15:
            grade = "good"
        elif s["rel_iqr"] < 0.30:
            grade = "moderate"
        else:
            grade = "WEAK"

        if isinstance(s["median"], float):
            med_str = f"{s['median']:.4f}"
            iqr_str = f"{s['iqr']:.4f}"
        else:
            med_str = f"{s['median']}"
            iqr_str = f"{s['iqr']}"

        print(f"  {key:<35} {med_str:>10} {iqr_str:>10} {s['rel_iqr']:>7.1%}  {grade}")

    print(f"\n  Final params (median across {n_repeats} repeats):")
    for k, v in sorted(final_params.items()):
        if isinstance(v, dict):
            print(f"    {k}: {json.dumps(v)}")
        elif isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    print(f"\n  Results written to {output_path}")
    print()


if __name__ == "__main__":
    main()
