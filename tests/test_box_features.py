"""Tests for allocator.box_features and the diagnostics test-marker infrastructure."""

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
