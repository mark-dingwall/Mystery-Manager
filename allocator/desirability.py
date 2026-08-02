"""
Item desirability scores derived from historical packing decisions.

Loads per-item residuals from diagnostics/desirability_items.csv,
applies Bayesian shrinkage, normalises to [0, 1], and provides
lookup functions for scoring.

Unknown items return 0.5 (neutral).
"""

import csv
from pathlib import Path

# Default shrinkage prior (was in scoring_config.json, now hardcoded since
# desirability is no longer a scoring dimension — only used by analyze_desirability.py)
DESIRABILITY_SHRINKAGE_PRIOR = 5

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "diagnostics" / "desirability_items.csv"

# Module-level cache
_cache: dict[str, float] | None = None
_cache_path: str | None = None


def _load(csv_path: str | Path | None = None, prior: int | None = None) -> dict[str, float]:
    """Load CSV, apply shrinkage, normalise to [0, 1]. Returns {name: score}."""
    path = Path(csv_path) if csv_path else _DEFAULT_CSV
    if prior is None:
        prior = DESIRABILITY_SHRINKAGE_PRIOR

    rows: list[tuple[str, int, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            n = int(row["n_appearances"])
            residual = float(row["residual"])
            rows.append((name, n, residual))

    if not rows:
        return {}

    # Global mean residual (weighted by appearances)
    total_n = sum(n for _, n, _ in rows)
    global_mean = sum(n * r for _, n, r in rows) / total_n if total_n > 0 else 0.0

    # Bayesian shrinkage: adjusted = (n * residual + prior * global_mean) / (n + prior)
    adjusted: dict[str, float] = {}
    for name, n, residual in rows:
        adjusted[name] = (n * residual + prior * global_mean) / (n + prior)

    # Normalise to [0, 1]
    vals = list(adjusted.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {name: 0.5 for name in adjusted}
    return {name: (v - lo) / (hi - lo) for name, v in adjusted.items()}


def _ensure_loaded(csv_path: str | Path | None = None, prior: int | None = None) -> dict[str, float]:
    """Return cached scores, loading on first call."""
    global _cache, _cache_path
    key = str(csv_path or _DEFAULT_CSV)
    if _cache is None or _cache_path != key:
        _cache = _load(csv_path, prior)
        _cache_path = key
    return _cache


def get_item_desirability(
    name: str,
    csv_path: str | Path | None = None,
    prior: int | None = None,
) -> float:
    """Return desirability score for an item name. Unknown items return 0.5."""
    scores = _ensure_loaded(csv_path, prior)
    return scores.get(name, 0.5)


def compute_box_desirability(
    allocations: dict[int, int],
    item_lookup: dict,
    csv_path: str | Path | None = None,
    prior: int | None = None,
) -> float:
    """
    Qty-weighted mean desirability for a box's allocations.

    item_lookup maps item_id to something with a "name" key (dict) or .name attr.
    Empty box returns 0.5 (neutral).
    """
    scores = _ensure_loaded(csv_path, prior)
    total_qty = 0
    weighted_sum = 0.0
    for item_id, qty in allocations.items():
        if qty <= 0:
            continue
        info = item_lookup.get(item_id)
        if info is None:
            continue
        name = info["name"] if isinstance(info, dict) else info.name
        desir = scores.get(name, 0.5)
        weighted_sum += desir * qty
        total_qty += qty
    if total_qty == 0:
        return 0.5
    return weighted_sum / total_qty


def _reset_cache() -> None:
    """Reset module-level cache (for test isolation)."""
    global _cache, _cache_path
    _cache = None
    _cache_path = None
