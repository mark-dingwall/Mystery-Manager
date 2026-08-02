"""Clean History screen -- configure and run historical data cleaning interactively.

Three visual states managed via .hidden CSS class:
  1. configure -- form to choose mode (standard / LLM), options, and offer filter
  2. running   -- progress counter with elapsed timer
  3. results   -- summary line with per-offer results table

Supports both standard XLSX→CSV cleaning and LLM extraction for Tier C/D offers.
"""
from __future__ import annotations

import time

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Switch,
)
from textual.worker import Worker, WorkerState, get_current_worker

from allocator.screens.help_overlay import HelpMixin


class CleanHistoryScreen(HelpMixin, Screen):
    """Interactive screen for configuring and running historical data cleaning."""

    BINDINGS = [
        Binding("escape", "cancel_or_back", "Back/Cancel"),
        Binding("enter", "start_run", "Run", show=True),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Clean History"
    HELP_TEXT = (
        "This screen lets you clean historical XLSX files into CSVs for "
        "algorithm comparison.\n\n"
        "Modes:\n"
        "  Standard Processing -- converts XLSX files from historical/ and "
        "historical/older/ into cleaned CSVs in cleaned/.\n"
        "  LLM Extraction -- uses Claude CLI to extract allocation data from "
        "non-standard Tier C/D workbooks. Writes to cleaned_llm/{method}/.\n\n"
        "Options:\n"
        "  Skip older/ -- only scan historical/ (skip historical/older/).\n"
        "  Only offers -- filter to specific offer IDs (e.g. 55-63 or 45,50).\n"
        "  LLM method -- extraction strategy (default: haiku-whole).\n"
        "  Force re-extraction -- ignore cached LLM results.\n\n"
        "Press Enter to start, Escape to cancel during a run or go back.\n"
        "LLM mode calls Claude CLI externally and may take several minutes."
    )

    def __init__(self) -> None:
        super().__init__()
        self._view_state: str = "configure"  # configure | running | results
        self._mode: str = "standard"  # standard | llm
        self._t0: float = 0.0
        self._timer = None
        self._results: dict = {}
        self._total: int = 0

    def compose(self) -> ComposeResult:
        yield Header()

        # -- Configure state --
        with Vertical(id="clean-config"):
            yield Label("Clean Historical Data", id="clean-title")

            yield RadioSet(
                RadioButton("Standard Processing", value=True, id="mode-standard"),
                RadioButton("LLM Extraction", id="mode-llm"),
                id="mode-radio",
            )

            with Horizontal(id="standard-options"):
                yield Switch(id="skip-older-switch", value=False)
                yield Label("Skip older/ directory", id="skip-older-label")

            with Horizontal(id="llm-options", classes="hidden"):
                yield Label("Method: ")
                yield Select(
                    [(m, m) for m in self._get_llm_methods()],
                    value="haiku-whole",
                    id="llm-method-select",
                    allow_blank=False,
                )

            with Horizontal(id="llm-force-option", classes="hidden"):
                yield Switch(id="force-switch", value=False)
                yield Label("Force re-extraction (ignore cache)")

            with Horizontal(id="offers-row"):
                yield Label("Only offers: ")
                yield Input(
                    placeholder="e.g. 55-63 or 45,50,55-63",
                    id="offers-input",
                )

            yield Label("", id="offers-error", classes="hidden")
            yield Label("Standard mode: all directories", id="config-summary")
            yield Label(
                "LLM mode calls Claude CLI externally and may take several minutes.",
                id="llm-warning",
                classes="hidden",
            )
            yield Button("Run", id="run-btn")

        # -- Running state (hidden) --
        yield Label("Preparing...", id="clean-progress", classes="hidden")
        yield Label("0s elapsed", id="clean-elapsed", classes="hidden")

        # -- Results state (hidden) --
        yield Label("", id="clean-summary", classes="hidden")
        results_table = DataTable(id="clean-results-table", classes="hidden")
        results_table.cursor_type = "row"
        yield results_table

        yield Footer()

    def on_mount(self) -> None:
        dt = self.query_one("#clean-results-table", DataTable)
        dt.add_columns("Offer", "Tier", "Boxes", "Items", "Status")
        self._update_config_summary()

    def on_show(self) -> None:
        self.app.sub_title = "Clean History"

    @staticmethod
    def _get_llm_methods() -> list[str]:
        """Get available LLM methods, with fallback if import fails."""
        try:
            from allocator.services.clean_history_service import CleanHistoryService
            return CleanHistoryService.available_llm_methods()
        except Exception:
            return ["haiku-whole", "sonnet-low"]

    # -------------------------------------------------------------------
    # State transitions
    # -------------------------------------------------------------------

    def _show_state(self, new_state: str) -> None:
        self._view_state = new_state

        # Configure widgets
        self.query_one("#clean-config").set_class(new_state != "configure", "hidden")

        # Running widgets
        for wid in ("#clean-progress", "#clean-elapsed"):
            self.query_one(wid).set_class(new_state != "running", "hidden")

        # Results widgets
        for wid in ("#clean-summary", "#clean-results-table"):
            self.query_one(wid).set_class(new_state != "results", "hidden")

    # -------------------------------------------------------------------
    # Form interaction
    # -------------------------------------------------------------------

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id != "mode-radio":
            return
        is_llm = event.pressed.id == "mode-llm"
        self._mode = "llm" if is_llm else "standard"

        self.query_one("#standard-options").set_class(is_llm, "hidden")
        self.query_one("#llm-options").set_class(not is_llm, "hidden")
        self.query_one("#llm-force-option").set_class(not is_llm, "hidden")
        self.query_one("#llm-warning").set_class(not is_llm, "hidden")

        self._update_config_summary()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "offers-input":
            return
        err_label = self.query_one("#offers-error", Label)
        val = event.value.strip()
        if not val:
            err_label.add_class("hidden")
            self._update_config_summary()
            return
        try:
            from allocator.services.clean_history_service import CleanHistoryService
            parsed = CleanHistoryService.parse_offer_range(val)
            err_label.add_class("hidden")
            self._update_config_summary()
        except (ValueError, TypeError):
            err_label.update("Invalid offer range format")
            err_label.remove_class("hidden")

    def _update_config_summary(self) -> None:
        """Update the live preview summary label."""
        parts = []
        if self._mode == "standard":
            skip_older = self.query_one("#skip-older-switch", Switch).value
            parts.append("Standard mode")
            if skip_older:
                parts.append("historical/ only")
            else:
                parts.append("all directories")
        else:
            try:
                method = self.query_one("#llm-method-select", Select).value
            except Exception:
                method = "haiku-whole"
            force = self.query_one("#force-switch", Switch).value
            parts.append(f"LLM mode ({method})")
            if force:
                parts.append("force re-extraction")

        offer_val = self.query_one("#offers-input", Input).value.strip()
        if offer_val:
            try:
                from allocator.services.clean_history_service import CleanHistoryService
                parsed = CleanHistoryService.parse_offer_range(offer_val)
                parts.append(f"{len(parsed)} offers")
            except (ValueError, TypeError):
                parts.append("invalid offer filter")
        else:
            parts.append("all offers")

        self.query_one("#config-summary", Label).update(" | ".join(parts))

    # -------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self.action_start_run()

    def action_start_run(self) -> None:
        if self._view_state != "configure":
            return

        # Validate offer input
        offer_val = self.query_one("#offers-input", Input).value.strip()
        only_offers = None
        if offer_val:
            try:
                from allocator.services.clean_history_service import CleanHistoryService
                only_offers = CleanHistoryService.parse_offer_range(offer_val)
            except (ValueError, TypeError):
                self.query_one("#offers-error", Label).update("Invalid offer range")
                self.query_one("#offers-error").remove_class("hidden")
                return

        self._show_state("running")
        self._t0 = time.monotonic()
        self._timer = self.set_interval(1.0, self._tick)
        self._run_clean(only_offers)

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._t0)
        self.query_one("#clean-elapsed", Label).update(f"{elapsed}s elapsed")

    @work(thread=True, exit_on_error=False, name="clean-history")
    def _run_clean(self, only_offers: set[int] | None) -> dict:
        from allocator.services.clean_history_service import CleanHistoryService

        worker = get_current_worker()

        def progress_cb(offer_id, completed, total):
            self.app.call_from_thread(
                self._update_progress, offer_id, completed, total
            )

        def cancel_ck():
            return worker.is_cancelled

        svc = CleanHistoryService()
        if self._mode == "standard":
            skip_older = self.app.call_from_thread(
                lambda: self.query_one("#skip-older-switch", Switch).value
            )
            return svc.run_standard_clean(
                include_older=not skip_older,
                only_offers=only_offers,
                progress_callback=progress_cb,
                cancel_check=cancel_ck,
            )
        else:
            method = self.app.call_from_thread(
                lambda: self.query_one("#llm-method-select", Select).value
            )
            force = self.app.call_from_thread(
                lambda: self.query_one("#force-switch", Switch).value
            )
            return svc.run_llm_clean(
                methods=[method] if method else None,
                force=force,
                only_offers=only_offers,
                progress_callback=progress_cb,
                cancel_check=cancel_ck,
            )

    def _update_progress(self, offer_id: int, completed: int, total: int) -> None:
        self._total = total
        self.query_one("#clean-progress", Label).update(
            f"Processing offer {offer_id} ({completed + 1}/{total})..."
        )

    # -------------------------------------------------------------------
    # Worker completion
    # -------------------------------------------------------------------

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "clean-history":
            return

        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            if self._timer is not None:
                self._timer.stop()
                self._timer = None

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result or {}
            self._results = result
            self._show_results(result)

        elif event.state == WorkerState.ERROR:
            err = (
                str(event.worker.error).strip().splitlines()[-1]
                if event.worker.error
                else "Unknown error"
            )
            self._show_state("configure")
            self.app.notify(f"Clean failed: {err}", severity="error")

        elif event.state == WorkerState.CANCELLED:
            # Back to configure on cancel
            self._show_state("configure")
            self.app.notify("Cancelled", severity="warning")

    # -------------------------------------------------------------------
    # Results display
    # -------------------------------------------------------------------

    def _show_results(self, result: dict) -> None:
        self._show_state("results")

        elapsed = int(time.monotonic() - self._t0)

        if self._mode == "standard":
            offers = result.get("offers", {})
            n = len(offers)
            self.query_one("#clean-summary", Label).update(
                f"Cleaned {n} offers in {elapsed}s -> cleaned/"
            )
            self._populate_results_table(offers)
        else:
            # LLM mode: result is keyed by method name
            total_offers = sum(len(v) for v in result.values())
            methods = ", ".join(result.keys()) if result else "none"
            self.query_one("#clean-summary", Label).update(
                f"LLM extracted {total_offers} offers in {elapsed}s "
                f"({methods}) -> cleaned_llm/"
            )
            # Flatten all method results into the table
            combined: dict[str, dict] = {}
            for method_data in result.values():
                if isinstance(method_data, dict):
                    combined.update(method_data)
            self._populate_results_table(combined)

    def _populate_results_table(self, offers: dict[str, dict]) -> None:
        dt = self.query_one("#clean-results-table", DataTable)
        dt.clear()

        for offer_id_str in sorted(offers.keys(), key=lambda k: int(k)):
            meta = offers[offer_id_str]
            offer_id = int(offer_id_str)
            tier = meta.get("tier", "?")
            boxes = str(meta.get("box_count", "--"))
            items = str(meta.get("total_items", "--"))
            status = "OK"
            if meta.get("extraction_method"):
                status = f"LLM ({meta['extraction_method']})"
            dt.add_row(
                str(offer_id), tier, boxes, items, status,
                key=offer_id_str,
            )

    # -------------------------------------------------------------------
    # Escape handling
    # -------------------------------------------------------------------

    def action_cancel_or_back(self) -> None:
        if self._view_state == "running":
            for w in self.workers:
                w.cancel()
        else:
            self.app.pop_screen()
