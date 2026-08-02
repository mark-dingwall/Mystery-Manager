"""
Re-scoring module for Optuna parameter tuning.

Pure functions, no DB imports, no config imports, no side effects.
All parameters passed explicitly via a params dict.

Precomputed box features (from extract_features.py) + params dict -> composite score.
This avoids rebuilding Item/MysteryBox/AllocationResult objects during tuning trials.
"""

from __future__ import annotations


def default_params() -> dict:
    """Return current config values as a params dict (for parity tests).

    Imports config lazily so the module itself stays pure.
    """
    from allocator.config import (
        DIVERSITY_PENALTY_MULTIPLIER,
        DIVERSITY_WEIGHTS,
        GROUP_ALLOWANCES,
        GROUP_CONCENTRATION_MULTIPLIER,
        GROUP_QTY_EXPONENT,
        MAX_COMPOSITE_SCORE,
        MAX_VALUE_SHARE_MULTIPLIER,
        MAX_VALUE_SHARE_THRESHOLD,
        PREF_VIOLATION_PENALTY,
        SAME_ITEM_MULTIPLIER,
        SIZE_FLOOR_MULTIPLIER,
        SIZE_FLOOR_TARGETS,
        VALUE_PENALTY_EXPONENT,
        VALUE_SWEET_FROM,
        VALUE_SWEET_TO,
    )

    return {
        "value_sweet_from": VALUE_SWEET_FROM,
        "value_sweet_to": VALUE_SWEET_TO,
        "value_penalty_exponent": VALUE_PENALTY_EXPONENT,
        "same_item_multiplier": SAME_ITEM_MULTIPLIER,
        "group_concentration_multiplier": GROUP_CONCENTRATION_MULTIPLIER,
        "group_qty_exponent": GROUP_QTY_EXPONENT,
        "group_allowances": {k: dict(v) for k, v in GROUP_ALLOWANCES.items()},
        "diversity_penalty_multiplier": DIVERSITY_PENALTY_MULTIPLIER,
        "w_subcat": DIVERSITY_WEIGHTS["sub_category"],
        "w_usage": DIVERSITY_WEIGHTS["usage"],
        "w_colour": DIVERSITY_WEIGHTS["colour"],
        "max_value_share_threshold": MAX_VALUE_SHARE_THRESHOLD,
        "max_value_share_multiplier": MAX_VALUE_SHARE_MULTIPLIER,
        "size_floor_targets": dict(SIZE_FLOOR_TARGETS),
        "size_floor_multiplier": SIZE_FLOOR_MULTIPLIER,
        "pref_violation_penalty": PREF_VIOLATION_PENALTY,
        "max_composite_score": MAX_COMPOSITE_SCORE,
    }


def _value_penalty(value_pct: float, params: dict) -> float:
    """Power-function value penalty (mirrors _scoring.value_penalty)."""
    sweet_from = params["value_sweet_from"]
    sweet_to = params["value_sweet_to"]
    exponent = params["value_penalty_exponent"]

    if sweet_from <= value_pct <= sweet_to:
        return 0.0
    if value_pct < sweet_from:
        x = sweet_from - value_pct
    else:
        x = value_pct - sweet_to
    return x ** exponent


def _same_item_penalty(item_quantities: list, params: dict) -> float:
    """Per-item excess penalty from precomputed item quantities.

    item_quantities: list of [qty, price, item_allowance] per item.
    penalty = sum(max(0, qty - allowance) * price * multiplier / 100).
    """
    multiplier = params["same_item_multiplier"]

    penalty = 0.0
    for qty, price, allowance in item_quantities:
        excess = max(0, qty - allowance)
        if excess > 0:
            penalty += excess * price * multiplier / 100.0
    return penalty


def _group_concentration_penalty(group_totals: list, tier: str, params: dict) -> float:
    """Group concentration penalty from precomputed group totals.

    group_totals: list of [group_load, degree, group_allowance] per fungible group.
    Only groups with explicit allowances are included (pre-filtered in features).
    """
    exponent = params["group_qty_exponent"]
    multiplier = params["group_concentration_multiplier"]

    penalty = 0.0
    for group_load, degree, group_allowance in group_totals:
        excess = max(0, group_load - group_allowance)
        if excess > 0:
            penalty += (excess ** exponent) * degree * multiplier
    return penalty


def _diversity_penalty(dim_ratios: dict, dim_available: dict, params: dict) -> float:
    """Diversity penalty from precomputed per-dimension ratios.

    dim_ratios: {dim: effective_species / n_available} for each dimension.
    dim_available: {dim: n_available} (0 means full marks for that dimension).
    """
    w_subcat = params["w_subcat"]
    w_usage = params["w_usage"]
    w_colour = params["w_colour"]
    w_shape = max(0.0, 1.0 - w_subcat - w_usage - w_colour)
    multiplier = params["diversity_penalty_multiplier"]

    weights = {
        "sub_category": w_subcat,
        "usage": w_usage,
        "colour": w_colour,
        "shape": w_shape,
    }

    score = 0.0
    for dim, weight in weights.items():
        n_avail = dim_available.get(dim, 0)
        if n_avail == 0:
            score += weight  # no variety available = full marks
        else:
            ratio = dim_ratios.get(dim, 0.0)
            score += weight * min(ratio, 1.0)

    return (1.0 - score) * multiplier


def _max_value_share_penalty(max_value_share: float, params: dict) -> float:
    """Penalty when any single item exceeds a threshold share of box value.

    penalty = max(0, share - threshold) * multiplier.
    """
    threshold = params["max_value_share_threshold"]
    multiplier = params["max_value_share_multiplier"]

    excess = max(0.0, max_value_share - threshold)
    return excess * multiplier


def _size_floor_penalty(total_size_points: int, tier: str, params: dict) -> float:
    """Weak penalty when total size points fall below tier target.

    penalty = max(0, target - total) * multiplier.
    """
    targets = params["size_floor_targets"]
    multiplier = params["size_floor_multiplier"]

    target = targets.get(tier, 0)
    if target <= 0:
        return 0.0

    deficit = max(0, target - total_size_points)
    return deficit * multiplier


def rescore_box(feature: dict, params: dict) -> dict:
    """Recompute per-box penalties from precomputed features and params.

    Returns dict with individual penalty components and total box penalty.
    """
    val_pen = _value_penalty(feature["value_pct"], params)
    si_pen = _same_item_penalty(feature["item_quantities"], params)
    gc_pen = _group_concentration_penalty(feature["group_totals"], feature["tier"], params)
    div_pen = _diversity_penalty(feature["dim_ratios"], feature["dim_available"], params)
    mvs_pen = _max_value_share_penalty(feature["max_value_share"], params)
    sf_pen = _size_floor_penalty(feature["total_size_points"], feature["tier"], params)

    pref_pen = feature["pref_violations"] * params["pref_violation_penalty"]

    total = val_pen + si_pen + gc_pen + div_pen + mvs_pen + sf_pen + pref_pen

    return {
        "value_pen": val_pen,
        "si_pen": si_pen,
        "gc_pen": gc_pen,
        "diversity_pen": div_pen,
        "mvs_pen": mvs_pen,
        "sf_pen": sf_pen,
        "pref_pen": pref_pen,
        "box_penalty": total,
    }


def rescore_offer(boxes: list[dict], params: dict) -> dict:
    """Recompute offer-level score from a list of box features.

    Returns composite score and penalty breakdown.
    No fairness term (dropped in v1 rework).
    """
    max_score = params.get("max_composite_score", 100.0)
    n = len(boxes)
    if n == 0:
        return {
            "score": max_score,
            "value_pen": 0.0,
            "si_pen": 0.0,
            "gc_pen": 0.0,
            "diversity_pen": 0.0,
            "mvs_pen": 0.0,
            "sf_pen": 0.0,
            "pref_pen": 0.0,
        }

    box_results = [rescore_box(f, params) for f in boxes]

    avg_val = sum(r["value_pen"] for r in box_results) / n
    avg_si = sum(r["si_pen"] for r in box_results) / n
    avg_gc = sum(r["gc_pen"] for r in box_results) / n
    avg_div = sum(r["diversity_pen"] for r in box_results) / n
    avg_mvs = sum(r["mvs_pen"] for r in box_results) / n
    avg_sf = sum(r["sf_pen"] for r in box_results) / n
    total_pref = sum(r["pref_pen"] for r in box_results)

    score = max_score - avg_val - avg_si - avg_gc - avg_div - avg_mvs - avg_sf - total_pref

    return {
        "score": score,
        "value_pen": avg_val,
        "si_pen": avg_si,
        "gc_pen": avg_gc,
        "diversity_pen": avg_div,
        "mvs_pen": avg_mvs,
        "sf_pen": avg_sf,
        "pref_pen": total_pref,
    }


def compute_marginal_deltas(
    base_features: dict,
    candidate_features: list[dict],
    params: dict | None = None,
) -> list[float]:
    """Compute per-candidate scoring deltas for adding each item to a box.

    base_features: precomputed feature dict for the box as-is.
    candidate_features: list of feature dicts, each representing the box
        with one candidate item added.  Must be in the same order as the
        candidate list the caller is tracking.
    params: scoring parameters (defaults to current config if None).

    Returns list of score deltas (positive = box improves, negative = worsens).
    Caller is responsible for recomputing features via extract_box_features()
    for each candidate addition — this function only does the rescoring.
    """
    if params is None:
        params = default_params()

    max_score = params.get("max_composite_score", 100.0)
    base_result = rescore_box(base_features, params)
    base_score = max_score - base_result["box_penalty"]

    deltas = []
    for cf in candidate_features:
        after_result = rescore_box(cf, params)
        after_score = max_score - after_result["box_penalty"]
        deltas.append(after_score - base_score)

    return deltas


def compute_objective(
    manual_features: list[dict],
    bad_features: list[dict],
    params: dict,
    *,
    prune_monoculture_threshold: float = 50.0,
) -> float | None:
    """Compute the Optuna objective value.

    manual_features: list of box feature dicts from historical manual packing.
    bad_features: list of synthetic bad box feature dicts.
    params: scoring parameters to evaluate.

    Returns objective value (higher is better), or None if trial should be pruned.
    """
    # Score manual boxes grouped by offer
    offers: dict[int, list[dict]] = {}
    for f in manual_features:
        offers.setdefault(f["offer_id"], []).append(f)

    offer_scores = []
    for offer_id, boxes in offers.items():
        result = rescore_offer(boxes, params)
        offer_scores.append(result["score"])

    if not offer_scores:
        return None

    mean_manual = sum(offer_scores) / len(offer_scores)

    # Score bad boxes individually (no cross-box terms — they're not real orders)
    max_score = params.get("max_composite_score", 100.0)
    bad_scores = []
    for f in bad_features:
        r = rescore_box(f, params)
        box_score = max_score - r["box_penalty"]
        bad_scores.append((box_score, f.get("source", "")))

    # Prune if any monoculture box scores above threshold
    for score, source in bad_scores:
        if "monoculture" in source and score > prune_monoculture_threshold:
            return None

    max_bad = max((s for s, _ in bad_scores), default=0.0)

    # Objective: maximise manual scores, penalise high bad scores
    obj = mean_manual - 100 * max(0, max_bad - 40) - 50 * max(0, 85 - mean_manual)

    return obj
