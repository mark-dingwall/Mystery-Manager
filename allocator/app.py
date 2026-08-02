"""Textual TUI application for Mystery Manager.

Provides MysteryManagerApp as the main entry point when run.py is
called with no arguments. Implements a 5-section main menu with DB
status badge and section screens.
"""

from __future__ import annotations

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Label, ListItem, ListView
from textual.worker import Worker, WorkerState

from allocator.screens.help_overlay import HelpMixin


# ---------------------------------------------------------------------------
# Quit confirmation
# ---------------------------------------------------------------------------


class QuitScreen(ModalScreen):
    """Modal confirmation dialog for quitting the application."""

    BINDINGS = [
        Binding("q,Q", "confirm_quit", "Quit"),
        Binding("escape", "app.pop_screen", "Cancel"),
    ]

    def compose(self):
        yield Label("Quit Mystery Manager? Press Q to confirm, Escape to cancel.")

    def action_confirm_quit(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

# Map menu item IDs to their screen classes.
# All named sections are now handled with explicit lazy-import branches in
# on_list_view_selected. This dict is kept as an extension point for future use.
_SCREEN_MAP: dict[str, type[Screen]] = {}


class MainMenuScreen(HelpMixin, Screen):
    """The primary navigation screen with 5 named sections."""

    BINDINGS = [
        Binding("q,Q", "request_quit", "Quit"),
        Binding("escape", "request_quit", "Quit", show=False),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("1", "select_1", show=False),
        Binding("2", "select_2", show=False),
        Binding("3", "select_3", show=False),
        Binding("4", "select_4", show=False),
        Binding("5", "select_5", show=False),
    ]

    HELP_TITLE = "Main Menu"
    HELP_TEXT = (
        "This is the main menu for Mystery Manager.\n\n"
        "Sections:\n"
        "  1. Weekly Allocation — run the weekly box-packing wizard to allocate "
        "overage items into mystery boxes and export the result. Requires DB connection.\n"
        "  2. Strategy Comparison — benchmark the canonical strategy against "
        "the baselines across historical offers, ranked by composite score.\n"
        "  3. Historical Data — browse past offer allocations and cleaned CSVs (Phase 5).\n"
        "  4. Clean History — run historical XLSX cleaning interactively "
        "(standard or LLM extraction).\n"
        "  5. Help — open the glossary to look up terms.\n\n"
        "The DB status badge (top-left) shows whether the local DB is reachable. "
        "Weekly Allocation requires a DB connection.\n\n"
        "Press Q to quit."
    )

    def compose(self):
        yield Header()
        yield Label("\u25cf Checking DB...", id="db-status")
        yield ListView(
            ListItem(Label("1. Weekly Allocation"), id="menu-allocation"),
            ListItem(Label("2. Strategy Comparison"), id="menu-strategy"),
            ListItem(Label("3. Historical Data"), id="menu-history"),
            ListItem(Label("4. Clean History"), id="menu-clean"),
            ListItem(Label("5. Help"), id="menu-help"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self._check_db_connectivity()

    def on_show(self) -> None:
        self.app.sub_title = "Main Menu"

    @work(thread=True, exit_on_error=False)
    def _check_db_connectivity(self) -> bool:
        from allocator.services.db_service import DBService

        return DBService().check_connectivity()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion — update badge from worker result."""
        if event.state == WorkerState.SUCCESS:
            connected = event.worker.result
            self.app.db_connected = connected

            badge = self.query_one("#db-status", Label)
            badge.remove_class("db-connected", "db-disconnected")
            if connected:
                badge.update("\u25cf Connected")
                badge.add_class("db-connected")
            else:
                badge.update("\u25cf Disconnected")
                badge.add_class("db-disconnected")

            for item_id in ("menu-allocation",):
                item = self.query_one(f"#{item_id}", ListItem)
                item.disabled = not connected

            if not connected:
                list_view = self.query_one(ListView)
                # Move focus to the first enabled item.
                for i, child in enumerate(list_view.children):
                    if isinstance(child, ListItem) and not child.disabled:
                        list_view.index = i
                        break

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Guard: Textual 6.5.0 mouse clicks can fire Selected on
        # disabled items, so always check the disabled flag.
        if event.item.disabled:
            self.app.notify(
                "DB required \u2014 check connection", severity="warning"
            )
            return
        self._navigate(event.item.id)

    def _navigate(self, item_id: str) -> None:
        """Navigate to the screen identified by *item_id*."""
        # Check disabled state for items that require DB.
        if item_id == "menu-allocation":
            item = self.query_one(f"#{item_id}", ListItem)
            if item.disabled:
                self.app.notify(
                    "DB required \u2014 check connection", severity="warning"
                )
                return
            from allocator.screens.wizard_state import WizardState
            from allocator.screens.early_steps import FileStep
            self.app.push_screen(FileStep(WizardState()))
            return

        if item_id == "menu-strategy":
            try:
                from allocator.screens.strategy_comparison import StrategyComparisonScreen
                self.app.push_screen(StrategyComparisonScreen())
            except ImportError:
                self.app.notify(
                    "Strategy Comparison not yet available",
                    severity="warning",
                )
            return

        if item_id == "menu-history":
            try:
                from allocator.screens.historical_data import HistoricalDataScreen
                self.app.push_screen(HistoricalDataScreen())
            except ImportError:
                self.app.notify(
                    "Historical Data not yet available",
                    severity="warning",
                )
            return

        if item_id == "menu-clean":
            try:
                from allocator.screens.clean_history_screen import CleanHistoryScreen
                self.app.push_screen(CleanHistoryScreen())
            except ImportError:
                self.app.notify(
                    "Clean History not yet available",
                    severity="warning",
                )
            return

        if item_id == "menu-help":
            from allocator.screens.glossary import GlossaryScreen
            self.app.push_screen(GlossaryScreen())
            return

        screen_cls = _SCREEN_MAP.get(item_id)
        if screen_cls is not None:
            self.app.push_screen(screen_cls())

    def action_select_1(self) -> None:
        self._navigate("menu-allocation")

    def action_select_2(self) -> None:
        self._navigate("menu-strategy")

    def action_select_3(self) -> None:
        self._navigate("menu-history")

    def action_select_4(self) -> None:
        self._navigate("menu-clean")

    def action_select_5(self) -> None:
        self._navigate("menu-help")

    def action_request_quit(self) -> None:
        self.app.push_screen(QuitScreen())


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class MysteryManagerApp(App):
    """Mystery Manager Textual application."""

    TITLE = "Mystery Manager"
    CSS_PATH = "app.tcss"

    db_connected: bool = False

    def compose(self):
        yield Header()
        yield Footer()

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())

    def action_quit(self) -> None:
        """Override Textual's default ctrl+q to show confirmation."""
        self.push_screen(QuitScreen())
