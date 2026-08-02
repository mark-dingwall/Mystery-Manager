"""Tests for WizardState helper functions (migrated to pytest).

Covers: detect_offer_id, discover_xlsx_files, WizardState defaults.
"""

import os
import tempfile
import time
from pathlib import Path

from allocator.screens.wizard_state import (
    WizardState,
    detect_offer_id,
    discover_xlsx_files,
)


class TestDetectOfferId:
    def test_standard(self):
        assert detect_offer_id(Path("offer_108_shopping_list.xlsx")) == 108

    def test_single_digit(self):
        assert detect_offer_id(Path("offer_5_shopping_list.xlsx")) == 5

    def test_non_matching_name(self):
        assert detect_offer_id(Path("my_file.xlsx")) is None

    def test_non_numeric(self):
        assert detect_offer_id(Path("offer_abc_shopping_list.xlsx")) is None


class TestDiscoverXlsxFiles:
    def test_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "older.xlsx"
            newer = root / "newer.xlsx"
            older.touch()
            newer.touch()
            old_time = time.time() - 10
            os.utime(older, (old_time, old_time))
            result = discover_xlsx_files(root)
            assert len(result) == 2
            assert result[0].name == "newer.xlsx"
            assert result[1].name == "older.xlsx"

    def test_only_xlsx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data.xlsx").touch()
            (root / "notes.txt").touch()
            (root / "other.csv").touch()
            result = discover_xlsx_files(root)
            assert len(result) == 1
            assert result[0].name == "data.xlsx"

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_xlsx_files(Path(tmpdir))
            assert result == []


class TestWizardStateDefaults:
    def test_defaults(self):
        state = WizardState()
        assert state.strategy == "ilp-optimal"
        assert state.xlsx_path is None
        assert state.offer_id is None
        assert state.boxes == []
        assert state.result is None


def test_strategy_step_sources_from_canonical_list():
    import inspect

    from allocator.screens.early_steps import StrategyStep

    src = inspect.getsource(StrategyStep.compose)
    assert "list_strategies()" in src           # reads the filtered (canonical-only) list
    assert "include_baselines" not in src        # wizard must never opt baselines back in
