"""Strategy comparison screen -- runs all strategies against historical offers
and displays a ranked leaderboard with colour-coded scores and per-offer
drill-down.

Four visual states managed via .hidden CSS class:
  1. confirm  -- pre-run confirmation with estimated time
  2. running  -- per-strategy progress counter with elapsed timer
  3. leaderboard -- ranked table (canonical vs baselines) + detail drill-down
  4. error    -- shown on failure (hidden by default)

Implements STRAT-01 (highest-score highlight) and STRAT-02 (colour-coded scores).
"""
from __future__ import annotations

import re as _re
import time

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label
from textual.worker import Worker, WorkerState, get_current_worker

from allocator.screens.help_overlay import HelpMixin


def _sort_key(cell) -> tuple:
    """Universal DataTable sort key: numeric if cell looks numeric, else string."""
    plain = cell.plain if hasattr(cell, "plain") else str(cell)
    stripped = _re.sub(r"[^\d.-]", "", plain)
    try:
        return (0, float(stripped), "")
    except ValueError:
        return (1, 0.0, plain.lower())


def _score_colour(score: float) -> str:
    """Colour code for composite scores: green >= 90, yellow >= 80, red < 80."""
    if score >= 90:
        return "green"
    if score >= 80:
        return "yellow"
    return "red"


class StrategyComparisonScreen(HelpMixin, Screen):
    """Full strategy comparison screen with confirm/progress/leaderboard/detail states."""

    BINDINGS = [
        Binding("escape", "cancel_or_back", "Back/Cancel"),
        Binding("enter", "start_run", "Start", show=True),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Strategy Comparison"
    HELP_TEXT = (
        "This screen benchmarks the canonical strategy (ilp-optimal) against the "
        "baselines across ~60 historical offers, ranked by composite score.\n\n"
        "Flow:\n"
        "  1. Confirmation -- shows estimated time. Press Enter to start.\n"
        "  2. Running -- progress counter shows which strategy is running and elapsed time.\n"
        "  3. Leaderboard -- ranked table of all strategies with colour-coded scores.\n"
        "  4. Drill-down -- select a strategy row to see per-offer breakdown.\n\n"
        "Score colours:\n"
        "  Green  = 90+   (excellent)\n"
        "  Yellow = 80-89  (good)\n"
        "  Red    = <80    (needs improvement)\n\n"
        "The highest-scoring strategy is highlighted above the table. ilp-optimal "
        "is the canonical production strategy; the rest are baselines.\n\n"
        "Escape cancels during a run (partial results shown if any strategies "
        "completed) or returns to the main menu from the leaderboard."
    )

    def __init__(self) -> None:
        super().__init__()
        self._view_state: str = "confirm"  # confirm | running | leaderboard
        self._results: dict[str, dict] = {}
        self._ranked_names: list[str] = []
        self._expanded_strategy: str | None = None
        self._t0: float = 0.0
        self._timer = None
        self._total_strategies: int = 8
        self._sort_column: str | None = None
        self._sort_reverse: bool = False
        self._detail_sort_column: str | None = None
        self._detail_sort_reverse: bool = False

    def on_show(self) -> None:
        self.app.sub_title = "Strategy Comparison"

    def compose(self) -> ComposeResult:
        yield Header()

        # -- Confirmation state (visible initially) --
        yield Label(
            "This will run 7 strategies against ~60 historical offers.\n"
            "Estimated time: 3-5 minutes.",
            id="confirm-message",
        )
        yield Label(
            "Run all strategies and compare results.",
            id="confirm-prompt",
        )
        yield Button("Run", id="run-btn")

        # -- Progress state (hidden) --
        yield Label(
            "Preparing comparison...",
            id="progress-label",
            classes="hidden",
        )
        yield Label("0s elapsed", id="elapsed-label", classes="hidden")

        # -- Leaderboard state (hidden) --
        yield Label("", id="recommendation-label", classes="hidden")
        leaderboard = DataTable(id="leaderboard-table", classes="hidden")
        leaderboard.cursor_type = "row"
        yield leaderboard
        detail = DataTable(id="detail-table", classes="hidden")
        detail.cursor_type = "row"
        yield detail

        # -- Error state (hidden via CSS) --
        yield Label("", id="error-label")

        yield Footer()

    def on_mount(self) -> None:
        # Set up leaderboard table columns
        lb = self.query_one("#leaderboard-table", DataTable)
        lb.add_columns("Rank", "Strategy", "Score", "Value", "Dupes", "Diversity", "Fairness", "Pref")

        # Set up detail table columns
        dt = self.query_one("#detail-table", DataTable)
        dt.add_columns("Offer", "Tier", "Score", "Value%", "Dupes", "Diversity", "Fairness", "Pref")

    # -----------------------------------------------------------------------
    # State transitions
    # -----------------------------------------------------------------------

    def _show_state(self, new_state: str) -> None:
        """Transition between visual states by toggling .hidden class."""
        self._view_state = new_state

        # Confirm widgets
        for wid in ("#confirm-message", "#confirm-prompt", "#run-btn"):
            self.query_one(wid).set_class(new_state != "confirm", "hidden")

        # Progress widgets
        for wid in ("#progress-label", "#elapsed-label"):
            self.query_one(wid).set_class(new_state != "running", "hidden")

        # Leaderboard widgets
        for wid in ("#recommendation-label", "#leaderboard-table"):
            self.query_one(wid).set_class(new_state != "leaderboard", "hidden")

        # Detail table stays hidden unless explicitly shown
        if new_state != "leaderboard":
            self.query_one("#detail-table").add_class("hidden")

    # -----------------------------------------------------------------------
    # Confirmation -> Running
    # -----------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self.action_start_run()

    def action_start_run(self) -> None:
        """Start the comparison run (only active from confirmation state)."""
        if self._view_state != "confirm":
            return
        self._show_state("running")
        self._t0 = time.monotonic()
        self._timer = self.set_interval(1.0, self._tick)
        self._run_comparison()

    def _tick(self) -> None:
        """Update the elapsed timer label."""
        elapsed = int(time.monotonic() - self._t0)
        self.query_one("#elapsed-label", Label).update(f"{elapsed}s elapsed")

    # -----------------------------------------------------------------------
    # Background worker
    # -----------------------------------------------------------------------

    @work(thread=True, exit_on_error=False, name="comparison")
    def _run_comparison(self) -> dict:
        """Run all strategies in a background thread."""
        from allocator.services.comparison_service import ComparisonService

        worker = get_current_worker()

        def progress_cb(strategy: str, completed: int, total: int) -> None:
            self.call_from_thread(self._update_progress, strategy, completed, total)

        def cancel_check() -> bool:
            return worker.is_cancelled

        return ComparisonService().run_all_strategies(
            progress_callback=progress_cb,
            cancel_check=cancel_check,
        )

    def _update_progress(self, strategy: str, completed: int, total: int) -> None:
        """Update the progress label from the main thread."""
        self._total_strategies = total
        self.query_one("#progress-label", Label).update(
            f"Running strategy {completed + 1}/{total}: {strategy}..."
        )

    # -----------------------------------------------------------------------
    # Worker completion
    # -----------------------------------------------------------------------

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "comparison":
            return

        if event.state == WorkerState.SUCCESS:
            self._stop_timer()
            results = event.worker.result
            if results:
                self._results = results
            self._show_leaderboard(self._results, partial=False)

        elif event.state == WorkerState.ERROR:
            self._stop_timer()
            err = str(event.worker.error).strip().splitlines()[-1] if event.worker.error else "Unknown error"
            self._show_state("confirm")  # reset to something visible
            # Hide confirm widgets and show error
            for wid in ("#confirm-message", "#confirm-prompt", "#progress-label", "#elapsed-label"):
                self.query_one(wid).add_class("hidden")
            err_label = self.query_one("#error-label", Label)
            err_label.update(f"Comparison failed: {err}\n\nPress Escape to return to menu.")
            err_label.add_class("error-visible")

        elif event.state == WorkerState.CANCELLED:
            self._stop_timer()
            if self._results:
                self._show_leaderboard(self._results, partial=True)
            else:
                self._pop_to_menu()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    # -----------------------------------------------------------------------
    # Leaderboard
    # -----------------------------------------------------------------------

    def _show_leaderboard(self, results: dict[str, dict], partial: bool = False) -> None:
        """Populate and display the leaderboard table."""
        self._show_state("leaderboard")

        # Rank strategies by composite score descending
        ranked = sorted(
            results.items(),
            key=lambda kv: kv[1].get("composite", {}).get("score", 0.0),
            reverse=True,
        )
        self._ranked_names = [name for name, _ in ranked]

        # Count offers from first strategy's per_offer data
        n_offers = 0
        if ranked:
            first_data = ranked[0][1]
            n_offers = len(first_data.get("per_offer", {}))

        # Highest-score highlight (canonical vs baselines)
        rec_label = self.query_one("#recommendation-label", Label)
        if ranked:
            winner_name = ranked[0][0]
            winner_score = ranked[0][1].get("composite", {}).get("score", 0.0)
            if partial:
                rec_label.update(
                    f"Partial results ({len(results)}/{self._total_strategies} strategies). "
                    f"Best so far: {winner_name} ({winner_score:.1f}/100) across {n_offers} offers"
                )
            else:
                rec_label.update(
                    f"Highest score: {winner_name} ({winner_score:.1f}/100) across "
                    f"{n_offers} offers. Canonical production strategy: ilp-optimal."
                )
        else:
            rec_label.update("No results available.")

        # Populate leaderboard table
        table = self.query_one("#leaderboard-table", DataTable)
        table.clear()
        for rank, (name, data) in enumerate(ranked, 1):
            comp = data.get("composite", {})
            score = comp.get("score", 0.0)
            colour = _score_colour(score)
            table.add_row(
                str(rank),
                name,
                f"[{colour}]{score:.1f}[/{colour}]",
                f"-{comp.get('value_pen', 0.0):.1f}",
                f"-{comp.get('si_pen', 0.0):.1f}",
                f"-{comp.get('gc_pen', 0.0):.1f}",
                f"-{comp.get('diversity_pen', 0.0):.1f}",
                f"-{comp.get('pref_pen', 0.0):.1f}",
                key=name,
            )

        # Hide detail table (fresh leaderboard)
        self.query_one("#detail-table").add_class("hidden")
        self._expanded_strategy = None

    # -----------------------------------------------------------------------
    # Per-offer drill-down
    # -----------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection on the leaderboard table for drill-down."""
        if event.data_table.id != "leaderboard-table":
            return

        strategy_name = str(event.row_key.value)

        # Toggle: if same strategy is already expanded, hide detail
        if self._expanded_strategy == strategy_name:
            self.query_one("#detail-table").add_class("hidden")
            self._expanded_strategy = None
            return

        self._expanded_strategy = strategy_name
        self._populate_detail(strategy_name)

    def _populate_detail(self, strategy_name: str) -> None:
        """Populate the detail table with per-offer breakdown for a strategy."""
        from allocator.services.comparison_service import ComparisonService

        detail = self.query_one("#detail-table", DataTable)
        detail.clear()

        data = self._results.get(strategy_name, {})
        per_offer = data.get("per_offer", {})

        if not per_offer:
            detail.remove_class("hidden")
            return

        # Lazy import scoring utilities
        from compare import compute_composite_score

        # Build rows sorted by offer ID
        for offer_id in sorted(per_offer.keys()):
            offer_data = per_offer[offer_id]
            algo_metrics = offer_data.get("algo", [])

            if not algo_metrics:
                continue

            tier = ComparisonService.offer_tier(offer_id)
            comp = compute_composite_score(algo_metrics)

            score = comp.get("score", 0.0)
            score_colour = _score_colour(score)

            # Value% -- average across boxes in this offer
            n_boxes = len(algo_metrics)
            avg_value_pct = sum(m.get("value_pct", 0.0) for m in algo_metrics) / n_boxes if n_boxes else 0.0

            # Colour code for Value% column
            from allocator.config import VALUE_SWEET_FROM
            if avg_value_pct >= VALUE_SWEET_FROM:
                val_colour = "green"
            elif avg_value_pct >= 100:
                val_colour = "yellow"
            else:
                val_colour = "red"

            detail.add_row(
                str(offer_id),
                tier,
                f"[{score_colour}]{score:.1f}[/{score_colour}]",
                f"[{val_colour}]{avg_value_pct:.1f}%[/{val_colour}]",
                f"-{comp.get('si_pen', 0.0):.1f}",
                f"-{comp.get('gc_pen', 0.0):.1f}",
                f"-{comp.get('diversity_pen', 0.0):.1f}",
                f"-{comp.get('pref_pen', 0.0):.1f}",
            )

        detail.remove_class("hidden")

    # -----------------------------------------------------------------------
    # Column sorting
    # -----------------------------------------------------------------------

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort leaderboard or detail table by clicked column header."""
        table_id = event.data_table.id
        col = str(event.column_key)

        if table_id == "leaderboard-table":
            if self._sort_column == col:
                self._sort_reverse = not self._sort_reverse
            else:
                self._sort_column = col
                self._sort_reverse = False
            event.data_table.sort(event.column_key, key=_sort_key, reverse=self._sort_reverse)
        elif table_id == "detail-table":
            if self._detail_sort_column == col:
                self._detail_sort_reverse = not self._detail_sort_reverse
            else:
                self._detail_sort_column = col
                self._detail_sort_reverse = False
            event.data_table.sort(event.column_key, key=_sort_key, reverse=self._detail_sort_reverse)

    # -----------------------------------------------------------------------
    # Escape handling (context-sensitive)
    # -----------------------------------------------------------------------

    def action_cancel_or_back(self) -> None:
        """Context-sensitive Escape: cancel running, or return to menu."""
        if self._view_state == "confirm":
            self.app.pop_screen()
        elif self._view_state == "running":
            for w in self.workers:
                w.cancel()
            # on_worker_state_changed(CANCELLED) handles showing partial results
        elif self._view_state == "leaderboard":
            self._pop_to_menu()
        else:
            # Error or unknown state -- go back to menu
            self._pop_to_menu()

    def _pop_to_menu(self) -> None:
        """Return to the main menu screen."""
        from allocator.app import MainMenuScreen
        while not isinstance(self.app.screen, MainMenuScreen):
            self.app.pop_screen()
