"""Early wizard screens: file selection, offer ID entry, strategy selection, and confirm.

FileStep -- shows a list of XLSX files in the project root, sorted
newest-first, and lets the user select one.

OfferIdStep -- shows the offer ID pre-filled from the filename and lets
the user confirm or override it before advancing to box review.

StrategyStep -- lists all allocation strategies after box review; the user
selects one before the mandatory confirm screen.

ConfirmScreen -- summary of file, offer ID, box count, and strategy;
user must press Y or Enter before allocation starts.

Heavy allocator imports (wizard_state, box_review, strategies) are all lazy:
they are done inside method bodies so that importing this module never
triggers module-level DB or SSH side effects.
"""
from __future__ import annotations

from pathlib import Path

from textual.binding import Binding
from textual.screen import Screen
from textual.validation import Integer
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView

from allocator.screens.help_overlay import HelpMixin
from allocator.screens.wizard_state import WizardState


_STRATEGY_DESCRIPTIONS = {
    "ilp-optimal":     "Best quality (ILP solver, ~30s) — canonical",
    "local-search":    "Near-optimal via iterative improvement (~10-20s)",
    "discard-worst":   "Fast subtractive greedy (<1s)",
    "round-robin":     "Fast round-robin draft (<1s)",
    "deal-topup":      "Three-phase deal + top-up (<1s) — baseline",
    "minmax-deficit":  "Minimise worst-off box (<1s)",
    "greedy-best-fit": "Greedy item-at-a-time (<1s)",
}
_RECOMMENDED_STRATEGY = "ilp-optimal"


class FileStep(HelpMixin, Screen):
    """Wizard step 1: select the shopping-list XLSX file."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("enter", "select_file", "Select", show=True),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Select File"
    HELP_TEXT = (
        "This is the first step each week. Select the XLSX shopping-list file "
        "for this week's offer.\n\n"
        "Files are shown newest-first from the project directory. "
        "Use the arrow keys to highlight a file and press Enter to select it, "
        "or press Escape to go back.\n\n"
        "The offer ID is auto-detected from the filename (e.g. offer_106_shopping_list.xlsx "
        "gives offer ID 106). You can override this on the next step."
    )

    def action_select_file(self) -> None:
        """Select the currently highlighted file in the list."""
        lv = self.query_one("#file-list", ListView)
        if lv.highlighted_child is not None:
            lv.action_select_cursor()

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state
        self._files: list[Path] = []

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Select File"

    def compose(self):
        yield Header()
        yield Label("Select an XLSX file:")
        yield ListView(id="file-list")
        yield Label("No XLSX files found in project directory.", id="empty-label")
        yield Footer()

    def on_mount(self) -> None:
        self._populate_files()

    def _populate_files(self) -> None:
        from allocator.screens.wizard_state import discover_xlsx_files

        self._files = discover_xlsx_files(Path.cwd())
        empty_label = self.query_one("#empty-label", Label)
        file_list = self.query_one("#file-list", ListView)

        if not self._files:
            empty_label.display = True
        else:
            empty_label.display = False
            for i, p in enumerate(self._files):
                file_list.append(ListItem(Label(p.name), id=f"file-{i}"))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.disabled:
            return

        # Extract index from item id "file-{i}"
        item_id = event.item.id or ""
        if not item_id.startswith("file-"):
            return
        try:
            idx = int(item_id[len("file-"):])
        except ValueError:
            return

        self._state.xlsx_path = self._files[idx]

        # Auto-detect offer ID from filename
        from allocator.screens.wizard_state import detect_offer_id
        self._state.offer_id = detect_offer_id(self._files[idx])

        self.app.push_screen(OfferIdStep(self._state))


class OfferIdStep(HelpMixin, Screen):
    """Wizard step 2: confirm or override the offer ID."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("enter", "submit", "Submit", show=True),
        Binding("question_mark", "help", "Help", key_display="?", priority=True),
    ]

    HELP_TITLE = "Offer ID"
    HELP_TEXT = (
        "Enter the offer ID (a positive whole number) for this week's allocation.\n\n"
        "The offer ID is automatically detected from the filename if possible — "
        "for example, offer_106_shopping_list.xlsx gives ID 106. "
        "Type to override if the detection is wrong.\n\n"
        "Press Enter or click Continue to proceed to box review. "
        "Press Escape to go back and select a different file.\n\n"
        "When to use: verify the correct offer number before loading boxes from the DB."
    )

    def action_submit(self) -> None:
        """Submit the offer ID (same as clicking Continue)."""
        self._submit()

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Offer ID"

    def compose(self):
        file_name = self._state.xlsx_path.name if self._state.xlsx_path else "(no file selected)"
        yield Header()
        yield Label(f"File: {file_name}")
        yield Input(
            value=str(self._state.offer_id or ""),
            type="integer",
            placeholder="Enter offer ID",
            validators=[Integer(minimum=1)],
            validate_on=["changed", "submitted"],
            id="offer-id-input",
        )
        yield Label("", id="offer-id-error", classes="hidden")
        yield Button("Continue \u2192", id="continue-btn")
        yield Footer()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "offer-id-input":
            return
        err = self.query_one("#offer-id-error", Label)
        if event.validation_result and not event.validation_result.is_valid:
            err.update("Offer ID must be a positive whole number")
            err.remove_class("hidden")
        else:
            err.add_class("hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue-btn":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:  # noqa: ARG002
        self._submit()

    def _submit(self) -> None:
        val = self.query_one("#offer-id-input", Input).value
        r = Integer(minimum=1).validate(val)
        if not r.is_valid:
            err = self.query_one("#offer-id-error", Label)
            err.update("Offer ID must be a positive whole number")
            err.remove_class("hidden")
            return

        self._state.offer_id = int(val)

        from allocator.screens.box_review import BoxReviewScreen
        self.app.push_screen(BoxReviewScreen(self._state))


class StrategyStep(HelpMixin, Screen):
    """Wizard step 4: select an allocation strategy after box review.

    Reached from BoxReviewScreen.action_confirm_boxes (not from OfferIdStep).
    Selecting a strategy advances to the mandatory ConfirmScreen.
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Select Strategy"
    HELP_TEXT = (
        "The weekly allocation uses ilp-optimal, the canonical production "
        "strategy (see CLAUDE.md > Project Direction).\n\n"
        "  ilp-optimal — Best results via an ILP solver (~30s). Falls back to "
        "local-search automatically if the solver is unavailable.\n\n"
        "Baseline strategies (deal-topup, greedy-best-fit, round-robin, "
        "minmax-deficit, discard-worst, local-search) are regression benchmarks "
        "only — run them via `compare.py --all-strategies` or "
        "`run.py --algorithm <name>`, not from this wizard.\n\n"
        "Press Enter to select and Escape to go back to box review."
    )

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Select Strategy"

    def compose(self):
        from allocator.strategies import list_strategies
        strategies = list_strategies()
        yield Header()
        yield Label("Select allocation strategy:")
        items = [
            ListItem(Label(f"{name}: {_STRATEGY_DESCRIPTIONS.get(name, '')}"), id=f"strat-{name}")
            for name in strategies
        ]
        yield ListView(*items, id="strategy-list")
        yield Footer()

    def on_mount(self) -> None:
        # Pre-select the recommended strategy
        lv = self.query_one("#strategy-list", ListView)
        for i, item in enumerate(lv.query(ListItem)):
            if item.id == f"strat-{_RECOMMENDED_STRATEGY}":
                lv.index = i
                break

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.disabled:
            return
        # Extract strategy name from item id "strat-{name}"
        item_id = event.item.id or ""
        if item_id.startswith("strat-"):
            self._state.strategy = item_id[len("strat-"):]
        self.app.push_screen(ConfirmScreen(self._state))


class ConfirmScreen(HelpMixin, Screen):
    """Summary + confirm screen before allocation runs.

    Locked decision from CONTEXT.md: shows file, offer ID, box count, strategy.
    User must press Y or Enter to proceed. Escape goes back to re-select strategy.
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("y", "confirm_run", "Confirm & run"),
        Binding("enter", "confirm_run", "Confirm & run"),
        Binding("question_mark", "help", "Help", key_display="?"),
    ]

    HELP_TITLE = "Confirm Allocation"
    HELP_TEXT = (
        "This is the final check before allocation runs.\n\n"
        "The summary shows the file, offer ID, number of boxes, and chosen strategy. "
        "Review these carefully — allocation cannot be cancelled once it starts.\n\n"
        "Press Y or Enter to start allocation. "
        "Press Escape to go back and change the strategy or box configuration."
    )

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self._state = state

    def on_show(self) -> None:
        self.app.sub_title = "Weekly Allocation — Confirm"

    def compose(self):
        yield Header()
        yield Label("Ready to run allocation", id="confirm-title")
        xlsx_name = self._state.xlsx_path.name if self._state.xlsx_path else "—"
        yield Label(f"File:       {xlsx_name}", id="confirm-file")
        yield Label(f"Offer ID:   {self._state.offer_id}", id="confirm-offer")
        yield Label(f"Boxes:      {len(self._state.boxes)}", id="confirm-boxes")
        yield Label(f"Strategy:   {self._state.strategy}", id="confirm-strategy")
        yield Label(
            "Press Y or Enter to run, Escape to go back.",
            id="confirm-prompt",
        )
        yield Footer()

    def action_confirm_run(self) -> None:
        from allocator.screens.progress_results import ProgressScreen
        self.app.push_screen(ProgressScreen(self._state))
