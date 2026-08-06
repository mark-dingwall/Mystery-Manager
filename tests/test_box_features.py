"""Tests for allocator.box_features and the diagnostics test-marker infrastructure."""

import json
from pathlib import Path

import pytest

from tests.conftest import require_dep


def test_require_dep_returns_module_when_present():
    # `json` is always importable, so this exercises the success path in both
    # strict and non-strict modes without needing the diagnostics stack.
    mod = require_dep("json")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_dep_skips_when_absent_and_not_strict():
    # Outside `-m diagnostics`, a missing dependency must skip, not error.
    from tests import conftest

    assert conftest._STRICT is False
    with pytest.raises(BaseException) as exc:
        require_dep("a_module_that_does_not_exist_xyz")
    assert exc.typename == "Skipped"


def test_declared_dependency_floors_and_distribution_names_are_exact():
    from tests.conftest import _DIAGNOSTIC_DEPENDENCIES

    assert _DIAGNOSTIC_DEPENDENCIES == {
        "interpret": ("interpret", "0.6.0"),
        "statsmodels": ("statsmodels", "0.14.0"),
        "sklearn": ("scikit-learn", "1.3.0"),
        "numpy": ("numpy", "1.24.0"),
        "pandas": ("pandas", "2.0.0"),
        "numexpr": ("numexpr", "2.8.4"),
        "bottleneck": ("bottleneck", "1.3.6"),
    }


def test_require_dep_accepts_version_at_floor_without_diagnostic_stack(monkeypatch):
    from tests import conftest

    sentinel = object()
    seen = []
    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: sentinel)
    monkeypatch.setattr(
        conftest.importlib_metadata,
        "version",
        lambda distribution: seen.append(distribution) or "1.3.0",
    )
    assert require_dep("sklearn") is sentinel
    assert seen == ["scikit-learn"]


def test_require_dep_skips_when_below_floor_and_not_strict(monkeypatch):
    from tests import conftest

    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(conftest.importlib_metadata, "version", lambda name: "1.2.9")
    with pytest.raises(BaseException) as exc:
        require_dep("sklearn")
    assert exc.typename == "Skipped"


def test_require_dep_raises_when_below_floor_and_strict(monkeypatch):
    from tests import conftest

    monkeypatch.setattr(conftest, "_STRICT", True)
    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(conftest.importlib_metadata, "version", lambda name: "1.2.9")
    with pytest.raises(ImportError, match="scikit-learn>=1.3.0"):
        require_dep("sklearn")


def test_require_dep_raises_when_absent_and_strict(monkeypatch):
    from tests import conftest

    def absent(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(conftest, "_STRICT", True)
    monkeypatch.setattr(conftest.importlib, "import_module", absent)
    with pytest.raises(ImportError, match="required diagnostic dependency"):
        require_dep("statsmodels")


@pytest.mark.parametrize("expr,expected", [
    ("", False),
    ("diagnostics", True),
    ("diagnostics and not slow", True),
    ("diagnostics and slow", True),       # satisfiable when slow is true
    ("diagnostics or slow", True),
    ("not diagnostics", False),          # exclusion must NOT arm strict mode
    ("not (diagnostics or slow)", False),
    ("not diagnostics or slow", False),  # diagnostics has only negative influence
    ("slow and not diagnostics", False),
    ("diagnostics and not diagnostics", False),
    ("diagnostics and slow and not slow", False),
    ("diagnostics or not diagnostics", False),
    ("not slow", False),                 # never named diagnostics: opt-in only
])
def test_strict_mode_follows_positive_selection_not_substring(expr, expected):
    """`-m "not diagnostics"` excludes these tests, but require_dep() runs at
    import time — during collection, before deselection. A substring check would
    hard-fail that command on the dependencies it is excluding."""
    from tests.conftest import _selects_diagnostics

    assert _selects_diagnostics(expr) is expected


# ---------------------------------------------------------------------------
# Shared fixture data for the box-feature tests
# ---------------------------------------------------------------------------
#
# Values come from tests/fixtures/scoring_config.json (synthetic) via conftest's
# bootstrap: fungible groups apple/banana/tomato, quantity classes snack_piece
# (small=2) and cooking_piece (small=1), box price small = BOX_PRICE_SMALL.


@pytest.fixture(autouse=True)
def _use_synthetic_scoring_config(monkeypatch):
    """Keep box-feature expectations independent of ignored local config links."""
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    config = json.loads(
        (Path(__file__).resolve().parent / "fixtures" / "scoring_config.json").read_text()
    )
    monkeypatch.setattr(box_features, "CATEGORY_FRUIT", config["category_fruit"])
    monkeypatch.setattr(box_features, "CATEGORY_VEGETABLES", config["category_vegetables"])
    monkeypatch.setattr(box_features, "GROUP_ALLOWANCES", config["group_allowances"])
    monkeypatch.setattr(scoring, "FUNGIBLE_GROUPS", config["fungible_groups"])
    monkeypatch.setattr(scoring, "QUANTITY_CLASSES", config["quantity_classes"])


def _item_lookup():
    """Three items spanning both categories, all three fungible groups."""
    from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES

    return {
        1: {"name": "Apples - Fuji", "price": 100, "size": 1,
            "category_id": CATEGORY_FRUIT, "fungible_group": "apple",
            "fungible_degree": 0.7, "sub_category": "pome_fruit",
            "usage": "snacking", "colour": "red", "shape": "round"},
        2: {"name": "Bananas - Cavendish", "price": 150, "size": 2,
            "category_id": CATEGORY_FRUIT, "fungible_group": "banana",
            "fungible_degree": 1.0, "sub_category": "tropical",
            "usage": "snacking", "colour": "yellow", "shape": "long"},
        3: {"name": "Tomatoes - Roma", "price": 200, "size": 1,
            "category_id": CATEGORY_VEGETABLES, "fungible_group": "tomato",
            "fungible_degree": 1.0, "sub_category": "fruiting_veg",
            "usage": "cooking", "colour": "red", "shape": "round"},
    }


def _available_tags():
    return {
        "sub_category": {"pome_fruit", "tropical", "fruiting_veg", "root_veg"},
        "usage": {"snacking", "cooking"},
        "colour": {"red", "yellow", "orange"},
        "shape": {"round", "long"},
    }


def _record():
    """The canonical record used across these tests.

    allocations {1: 3, 2: 1, 3: 2} on a small box. Hand-computed expectations:
      total_value  = 100*3 + 150*1 + 200*2 = 850
      value_pct    = 850 / BOX_TIERS["small"]["price"] * 100
      allowances   = apple/banana -> snack_piece small = 2; tomato -> cooking_piece small = 1
    """
    from allocator.box_features import extract_box_features

    return extract_box_features(
        box_name="test@example.com",
        allocations={1: 3, 2: 1, 3: 2},
        item_lookup=_item_lookup(),
        tier="small",
        available_tags=_available_tags(),
        offer_id=999,
        source="manual",
    )


def test_box_features_module_imports_without_queries_json(monkeypatch):
    """The module must not reach allocator.db, directly or transitively."""
    import subprocess
    import sys

    # A subprocess with an import hook that hard-fails on allocator.db proves the
    # absence of the transitive path, which a plain import in this process cannot
    # (conftest may already have imported db via another test module).
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name in ('allocator.db', 'compare', 'scripts.extract_features'):\n"
        "            raise AssertionError('box_features imported ' + name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import allocator.box_features\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_extract_box_features_scalar_fields():
    from allocator.config import BOX_TIERS

    rec = _record()
    assert rec["offer_id"] == 999
    assert rec["box_name"] == "test@example.com"
    assert rec["tier"] == "small"
    assert rec["source"] == "manual"
    assert rec["value_pct"] == round(850 / BOX_TIERS["small"]["price"] * 100, 4)
    assert rec["total_size_points"] == 7          # 1*3 + 2*1 + 1*2
    assert rec["max_value_share"] == 0.470588     # 400/850
    assert rec["pref_violations"] == 0


def test_extract_box_features_scored_lists_unchanged():
    """group_totals and item_quantities are rescore_box()'s inputs — capped,
    positional, and must not drift."""
    rec = _record()
    assert rec["item_quantities"] == [[3, 100, 2], [1, 150, 2], [2, 200, 1]]
    # [capped_load, degree, group_allowance] per group, insertion-ordered
    assert rec["group_totals"] == [[2, 0.7, 2], [1, 1.0, 2], [1, 1.0, 1]]


def test_extract_box_features_dim_ratios():
    rec = _record()
    assert rec["dim_available"] == {"sub_category": 4, "usage": 2, "colour": 3, "shape": 2}
    assert rec["dim_ratios"] == {
        "sub_category": 0.642857,   # eff_species({3,1,2}) = 2.571429 / 4
        "usage": 0.9,               # eff_species({4,2})   = 1.8      / 2
        "colour": 0.461538,         # eff_species({5,1})   = 1.384615 / 3
        "shape": 0.692308,          # eff_species({5,1})   = 1.384615 / 2
    }


def test_extract_box_features_returns_none_when_nothing_resolves():
    from allocator.box_features import extract_box_features

    assert extract_box_features(
        box_name="x", allocations={42: 3}, item_lookup=_item_lookup(),
        tier="small", available_tags=_available_tags(), offer_id=1,
    ) is None


def test_extract_features_script_reexports_the_same_object():
    """Existing CLI callers must keep working, against one implementation."""
    import allocator.box_features as bf

    src = (Path(__file__).resolve().parent.parent / "scripts" / "extract_features.py").read_text()
    assert "from allocator.box_features import extract_box_features" in src
    assert "def extract_box_features(" not in src
    assert "def _effective_species(" not in src
    assert callable(bf.extract_box_features)
