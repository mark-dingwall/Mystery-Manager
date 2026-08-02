"""
Allocation strategies: one canonical production strategy plus runnable baselines.

`ilp-optimal` is the canonical production strategy (see CLAUDE.md § Project
Direction). The remaining strategies are BASELINES — runnable regression
benchmarks the canonical model must beat; they are not production-selectable
and should not be extended. `local-search` is also the ILP fallback
(load-bearing — do not remove).

A strategy is a callable (AllocationResult) -> None that fills box.allocations
in place. Everything before (data loading, box building) and after (charity
allocation, stock) is shared infrastructure in allocate().
"""

from typing import Callable

from allocator.models import AllocationResult

Strategy = Callable[[AllocationResult], None]

_REGISTRY: dict[str, tuple[str, str]] = {
    # name -> (module_path, function_name) — lazy-loaded to avoid circular imports
    "deal-topup": ("allocator.strategies.deal_topup", "run"),
    "greedy-best-fit": ("allocator.strategies.greedy_best_fit", "run"),
    "round-robin": ("allocator.strategies.round_robin", "run"),
    "minmax-deficit": ("allocator.strategies.minmax_deficit", "run"),
    "local-search": ("allocator.strategies.local_search", "run"),
    "ilp-optimal": ("allocator.strategies.ilp_optimal", "run"),
    "discard-worst": ("allocator.strategies.discard_worst", "run"),
}

# --- Canonical direction (see CLAUDE.md § Project Direction) ---
CANONICAL_STRATEGY = "ilp-optimal"
FALLBACK_STRATEGY = "local-search"  # used when ilp-optimal is unavailable
BASELINE_STRATEGIES = frozenset(
    {
        "deal-topup",
        "greedy-best-fit",
        "round-robin",
        "minmax-deficit",
        "discard-worst",
        "local-search",
    }
)

DEFAULT_STRATEGY = CANONICAL_STRATEGY


def get_strategy(name: str) -> Strategy:
    """Look up a strategy by name, importing its module lazily."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name!r}. Available: {available}")
    module_path, func_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def list_strategies(include_baselines: bool = False) -> list[str]:
    """List selectable strategies.

    By default returns only canonical (production) strategies, so production
    pickers surface `ilp-optimal` alone. Benchmark and diagnostic surfaces pass
    include_baselines=True for the full set (canonical + baselines) used in
    leaderboards/comparisons. Baseline strategies remain runnable via
    get_strategy() regardless of this filter.
    """
    if include_baselines:
        return list(_REGISTRY.keys())
    return [name for name in _REGISTRY if name not in BASELINE_STRATEGIES]
