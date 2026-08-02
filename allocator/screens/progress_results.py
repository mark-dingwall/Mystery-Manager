"""Progress and Results screens for the weekly allocation wizard.

ProgressScreen -- shows a spinner and elapsed-time counter while the
allocation runs in a background thread.  Once the worker completes,
it pushes ResultsScreen automatically.  No Escape binding -- the user
cannot go back once allocation starts.

ResultsScreen -- shows per-box allocation metrics (email, tier, value%,
item count), aggregate composite score, charity/stock summary, and
provides clipboard and file-save export actions.
"""
from __future__ import annotations

import re as _re
import subprocess
import time
from pathlib import Path


def _sort_key(cell) -> tuple:
    """Universal DataTable sort key: numeric if cell looks numeric, else string."""
    plain = cell.plain if hasattr(cell, "plain") else str(cell)
    stripped = _re.sub(r"[^\d.-]", "", plain)
    try:
        return (0, float(stripped), "")
    except ValueError:
        return (1, 0.0, plain.lower())

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, LoadingIndicator, Static  # noqa: F401
from textual.worker import Worker, WorkerState

from allocator.screens.help_overlay import HelpMixin
from allocator.screens.wizard_state import WizardState


# ---------------------------------------------------------------------------
# ProgressScreen
# ---------------------------------------------------------------------------


class ProgressScreen(HelpMixin, Screen):
    """Shows spinner + elapsed time while allocation runs in a worker thread."""

    # No Escape binding — user cannot go back once allocation starts.
    # The only exit is forward (to results) or error (start over).
    BINDINGS = [
        Binding("s", "start_over", "Start over"),
        Binding("q", "request_quit", "Quit"),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Allocation Running"
    HELP_TEXT = (
        "The allocation is running in the background. Please wait.\n\n"
        "The screen shows the strategy, offer ID, and box count. "
        "The elapsed timer counts how long the allocation has been running.\n\n"
        "ILP-optimal typically takes 10-30 seconds. Faster strategies finish in under 1 second.\n\n"
        "There is no cancel — the process runs to completion. "
        "If the allocation fails, press S to start over or Q to quit."
    )

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Running"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Running allocation...", id="progress-label")
        yield Label(
            f"Strategy: {self._state.strategy}  |  Offer: {self._state.offer_id}  |  "
            f"Boxes: {len(self._state.boxes)}",
            id="progress-details",
        )
        yield LoadingIndicator(id="spinner")
        yield Label("0s elapsed", id="elapsed")
        yield Label("", id="error-label")
        yield Footer()

    def on_mount(self) -> None:
        self._t0 = time.monotonic()
        self.set_interval(1.0, self._tick)
        self._run_allocation()

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._t0)
        self.query_one("#elapsed", Label).update(f"{elapsed}s elapsed")

    @work(thread=True, exit_on_error=False)
    def _run_allocation(self) -> object:
        from allocator.services.allocation_service import AllocationService
        return AllocationService().run_allocation(
            offer_id=self._state.offer_id,
            xlsx_path=self._state.xlsx_path,
            boxes=self._state.boxes,
            strategy=self._state.strategy,
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            self._state.result = result
            # Replace this screen with results (push forward — no going back to progress)
            self.app.push_screen(ResultsScreen(self._state))
        elif event.state == WorkerState.ERROR:
            err = str(event.worker.error)
            self.query_one("#spinner").display = False
            self.query_one("#progress-label", Label).update("Allocation failed")
            # Strip traceback — show only the last line of the exception message
            short_err = err.strip().splitlines()[-1] if err.strip() else "Unknown error"
            err_label = self.query_one("#error-label", Label)
            err_label.update(
                f"Error: {short_err}\n\nPress S to start over or Q to quit."
            )
            err_label.add_class("error-visible")

    def action_start_over(self) -> None:
        from allocator.app import MainMenuScreen
        while not isinstance(self.app.screen, MainMenuScreen):
            self.app.pop_screen()

    def action_request_quit(self) -> None:
        from allocator.app import QuitScreen
        self.app.push_screen(QuitScreen())


# ---------------------------------------------------------------------------
# ResultsScreen
# ---------------------------------------------------------------------------


class ResultsScreen(HelpMixin, Screen):
    """Shows per-box allocation metrics and provides export actions."""

    BINDINGS = [
        Binding("c", "copy_clipboard", "Copy to clipboard"),
        Binding("f", "save_file", "Save to file"),
        Binding("escape", "return_to_menu", "Main menu"),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Allocation Results"
    HELP_TEXT = (
        "The results table shows per-box allocation metrics.\n\n"
        "Columns:\n"
        "  Box — customer email or standalone box name.\n"
        "  Tier — box size tier.\n"
        "  Value — total allocated value in dollars.\n"
        "  Val% — box value as a percentage of target price.\n"
        "    Green = 114%+ (sweet spot), Yellow = 100-113%, Red = below target.\n"
        "  Score — 100 minus total penalties (higher is better).\n"
        "  Penalty — total composite penalty.\n"
        "  -Val, -Dup, -Div — breakdown: value, dupe, and diversity penalties.\n"
        "  Items — number of items allocated to this box.\n\n"
        "Actions:\n"
        "  C — copy the tab-delimited allocation output to clipboard.\n"
        "  F — save the output to a file in the output/ directory.\n"
        "  Escape — return to the main menu.\n"
        "  Click a column header to sort."
    )

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state
        self._output_text: str | None = None
        self._sort_column: str | None = None
        self._sort_reverse: bool = False

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Results"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Computing scores...", id="aggregate-label")
        yield LoadingIndicator(id="scores-spinner")
        table = DataTable(id="results-table")
        table.display = False
        yield table
        yield Label("", id="charity-label")
        yield Label("", id="stock-label")
        yield Label("", id="export-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Box", "Tier", "Value", "Val%", "Score", "Penalty", "-Val", "-Dup", "-Div", "Items")
        self._compute_scores()

    @work(thread=True, exit_on_error=False, name="scores")
    def _compute_scores(self) -> dict:
        """Compute box metrics + format output text."""
        from compare import (
            build_item_lookup,
            compute_available_tags,
            compute_box_metrics,
            compute_composite_score,
        )
        from allocator.excel_io import format_output

        result = self._state.result
        item_lookup = build_item_lookup(result.offer_id)
        avail_tags = compute_available_tags(item_lookup)

        from allocator.strategies._scoring import value_penalty
        from allocator.config import (
            DIVERSITY_PENALTY_MULTIPLIER,
            PREF_VIOLATION_PENALTY,
        )

        metrics = []
        rows = []
        for box in result.boxes:
            m = compute_box_metrics(
                box.name,
                box.allocations,
                item_lookup,
                box.tier,
                preference=box.preference,
                available_tags=avail_tags,
            )
            if m:
                val_pen = value_penalty(m["value_pct"])
                si_pen = m.get("same_item_penalty", 0.0)
                gc_pen = m.get("group_concentration_penalty", 0.0)
                div_pen = (1.0 - m["diversity_score"]) * DIVERSITY_PENALTY_MULTIPLIER
                mvs_pen = m.get("max_value_share_penalty", 0.0)
                sf_pen = m.get("size_floor_penalty", 0.0)
                pref_pen = m["pref_violations"] * PREF_VIOLATION_PENALTY
                total_pen = val_pen + si_pen + gc_pen + div_pen + mvs_pen + sf_pen + pref_pen
                m["_val_pen"] = val_pen
                m["_si_pen"] = si_pen
                m["_gc_pen"] = gc_pen
                m["_div_pen"] = div_pen
                m["_mvs_pen"] = mvs_pen
                m["_sf_pen"] = sf_pen
                m["_total_pen"] = total_pen
                m["_box_score"] = 100.0 - total_pen
                metrics.append(m)
                rows.append((box, m))

        composite = compute_composite_score(metrics) if metrics else {"score": 0.0}
        output_text = format_output(result)

        return {
            "rows": rows,
            "composite": composite,
            "output_text": output_text,
            "result": result,
        }

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        name = event.worker.name
        if name == "scores":
            if event.state == WorkerState.SUCCESS:
                data = event.worker.result
                self._output_text = data["output_text"]
                self._populate_results(data["rows"], data["composite"], data["result"])
                self.query_one("#scores-spinner").display = False
            elif event.state == WorkerState.ERROR:
                err = str(event.worker.error).strip().splitlines()[-1]
                self.query_one("#scores-spinner").display = False
                self.query_one("#aggregate-label", Label).update(
                    f"Score computation failed: {err} — output still available for export"
                )
                # Still try to format output for export even if scoring failed
                try:
                    from allocator.excel_io import format_output
                    self._output_text = format_output(self._state.result)
                except Exception:
                    pass
        elif name == "clipboard":
            if event.state == WorkerState.SUCCESS:
                success = event.worker.result
                status = "Copied to clipboard!" if success else "Clipboard copy failed"
                self.query_one("#export-status", Label).update(status)
                self.app.notify(status, severity="information" if success else "error")

    def _populate_results(self, rows: list, composite: dict, result) -> None:
        from allocator.config import VALUE_SWEET_FROM, VALUE_SWEET_TO

        score = composite.get("score", 0.0)
        self.query_one("#aggregate-label", Label).update(
            f"Composite score: {score:.1f} / 100   "
            f"(value -{composite.get('value_pen', 0):.1f}  "
            f"same-item -{composite.get('si_pen', 0):.1f}  "
            f"grp-conc -{composite.get('gc_pen', 0):.1f}  "
            f"diversity -{composite.get('diversity_pen', 0):.1f}  "
            f"mvs -{composite.get('mvs_pen', 0):.1f}  "
            f"sz-floor -{composite.get('sf_pen', 0):.1f})"
        )

        table = self.query_one("#results-table", DataTable)
        table.display = True
        for box, m in rows:
            pct = m.get("value_pct", 0.0)
            if pct >= VALUE_SWEET_FROM:
                colour = "green"
            elif pct >= 100:
                colour = "yellow"
            else:
                colour = "red"
            item_count = sum(box.allocations.values()) if box.allocations else 0
            total_val = m.get("total_value", 0)
            box_score = m.get("_box_score", 0.0)
            total_pen = m.get("_total_pen", 0.0)
            val_pen = m.get("_val_pen", 0.0)
            dup_pen = m.get("_gq_pen", 0.0)
            div_pen = m.get("_div_pen", 0.0)
            table.add_row(
                box.name[:30],
                box.tier,
                f"${total_val / 100:.2f}",
                f"[{colour}]{pct:.1f}%[/{colour}]",
                f"{box_score:.1f}",
                f"{total_pen:.1f}",
                f"{val_pen:.1f}",
                f"{dup_pen:.1f}",
                f"{div_pen:.1f}",
                str(item_count),
            )

        # Charity summary: allocated vs target vs committed giving
        total_charity_val = sum(result.charity_value(c) for c in result.charity)
        total_charity_items = sum(len(c.allocations) for c in result.charity)
        charity_names = " + ".join(c.name for c in result.charity) if result.charity else "Charity"
        target = getattr(result, "charity_target", 0)
        giving = getattr(result, "charity_giving", 0)
        diff = total_charity_val - target
        diff_str = (f"+${diff/100:.2f}" if diff >= 0 else f"-${abs(diff)/100:.2f}")
        self.query_one("#charity-label", Label).update(
            f"{charity_names}: {total_charity_items} items  ${total_charity_val/100:.2f}"
            f"  |  Target: ${target/100:.2f}  Diff: {diff_str}"
            f"  |  Committed: ${giving/100:.2f}"
        )

        # Stock (should normally be empty — all overage goes to charity)
        stock_count = sum(result.stock.values()) if result.stock else 0
        if stock_count > 0:
            stock_val = sum(
                result.items[iid].price * qty
                for iid, qty in result.stock.items()
                if iid in result.items
            )
            self.query_one("#stock-label", Label).update(
                f"Stock: {stock_count} units  ${stock_val/100:.2f}"
            )
        else:
            self.query_one("#stock-label").display = False

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        col = str(event.column_key)
        if self._sort_column == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col
            self._sort_reverse = False
        event.data_table.sort(event.column_key, key=_sort_key, reverse=self._sort_reverse)

    def action_copy_clipboard(self) -> None:
        if not self._output_text:
            self.app.notify("No output ready yet — wait for scores to load", severity="warning")
            return
        self._do_clipboard_copy()

    @work(thread=True, exit_on_error=False, name="clipboard")
    def _do_clipboard_copy(self) -> bool:
        proc = subprocess.run(
            ["clip.exe"],
            input=self._output_text,
            text=True,
            timeout=5,
            capture_output=True,
        )
        return proc.returncode == 0

    def action_save_file(self) -> None:
        if not self._output_text:
            self.app.notify("No output ready yet", severity="warning")
            return
        try:
            out_path = Path(f"output/offer_{self._state.offer_id}_allocation.txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(self._output_text, encoding="utf-8")
            self.query_one("#export-status", Label).update(f"Saved to {out_path}")
            self.app.notify(f"Saved to {out_path}", severity="information")
        except Exception as e:
            self.app.notify(f"Save failed: {e}", severity="error")

    def action_return_to_menu(self) -> None:
        from allocator.app import MainMenuScreen
        while not isinstance(self.app.screen, MainMenuScreen):
            self.app.pop_screen()
