"""Strategy comparison service -- wraps compare.py for TUI use."""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class ComparisonService:
    """Provides blocking strategy comparison operations for the TUI.

    All methods are synchronous. Callers must run them in a thread
    worker (@work(thread=True)) to avoid blocking the Textual event loop.

    All compare.py imports are lazy (inside method bodies) to avoid
    triggering DB connections on import.
    """

    @staticmethod
    def offer_tier(offer_id: int) -> str:
        """Determine data tier from offer ID using CLAUDE.md boundaries.

        CLAUDE.md defines: A=64-106, B=55-63, C=45-54, D=22-44.
        Note: compare.py uses >=49 for C boundary, but CLAUDE.md is canonical.
        """
        if offer_id >= 64:
            return "A"
        if offer_id >= 55:
            return "B"
        if offer_id >= 45:
            return "C"
        return "D"

    @staticmethod
    def strategy_count() -> int:
        """Return the number of strategies (including 'manual')."""
        from allocator.strategies import list_strategies
        return len(list_strategies(include_baselines=True)) + 1  # +1 for "manual"

    def run_all_strategies(
        self,
        progress_callback: Callable[[str, int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, dict]:
        """Run all strategies sequentially against historical offers.

        Args:
            progress_callback: Called with (strategy_name, completed, total)
                before each strategy starts.
            cancel_check: Called before each strategy; if it returns True,
                execution stops and partial results are returned.

        Returns:
            {
                "strategy_name": {
                    "composite": {"score": float, "value_pen": float, ...},
                    "averages": {"count": int, "avg_value_pct": float, ...},
                    "per_offer": {offer_id: {"manual": [...], "algo": [...]}},
                },
                ...
            }

        Strategies are run sequentially (not via run_all_strategies_parallel)
        to support per-strategy progress reporting and cancellation. The
        discard-worst strategy is run before local-search to preserve the
        bootstrap optimisation path.
        """
        import compare
        from compare import (
            _build_offer_ids,
            compute_averages,
            compute_composite_score,
            load_summary,
            run_single_strategy_parallel,
        )
        from allocator.strategies import list_strategies

        summary = load_summary()

        # Set module-level OFFER_IDS to include A+B+C offers (45-106)
        compare.OFFER_IDS = _build_offer_ids(summary, only_offers="45-106")
        logger.info(
            "Comparison running across %d offers (45-106)",
            len(compare.OFFER_IDS),
        )

        # Build strategy list with discard-worst before local-search
        raw_strategies = list_strategies(include_baselines=True)
        ordered: list[str] = []
        deferred_local_search = False
        for s in raw_strategies:
            if s == "local-search":
                deferred_local_search = True
                continue
            ordered.append(s)
            # Insert local-search right after discard-worst
            if s == "discard-worst" and deferred_local_search:
                ordered.append("local-search")
                deferred_local_search = False
        # If local-search wasn't inserted (discard-worst came first or missing)
        if deferred_local_search:
            ordered.append("local-search")
        # Add manual pseudo-strategy at the end
        ordered.append("manual")

        total = len(ordered)
        results: dict[str, dict] = {}

        for completed, strategy in enumerate(ordered):
            # Check for cancellation
            if cancel_check is not None and cancel_check():
                logger.info("Comparison cancelled after %d/%d strategies", completed, total)
                break

            # Report progress
            if progress_callback is not None:
                progress_callback(strategy, completed, total)

            logger.debug("Running strategy: %s (%d/%d)", strategy, completed + 1, total)

            try:
                all_manual, all_algo, per_offer = run_single_strategy_parallel(
                    strategy, summary
                )

                composite = compute_composite_score(all_algo)
                averages = compute_averages(all_algo)

                # Build per-offer dict with manual and algo metrics
                per_offer_data: dict[int, dict] = {}
                for offer_id, (manual_metrics, algo_metrics) in per_offer.items():
                    per_offer_data[offer_id] = {
                        "manual": manual_metrics,
                        "algo": algo_metrics,
                    }

                results[strategy] = {
                    "composite": composite,
                    "averages": averages,
                    "per_offer": per_offer_data,
                }

                logger.info(
                    "Strategy %s: score=%.1f",
                    strategy,
                    composite["score"],
                )

            except Exception:
                logger.exception("Strategy %s failed", strategy)
                continue

        return results
