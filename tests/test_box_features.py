"""Tests for allocator.box_features and the diagnostics test-marker infrastructure."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.conftest import require_dep


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCORING_FIXTURE_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "scoring_config.json"
_SCORING_FIXTURE = json.loads(_SCORING_FIXTURE_PATH.read_text())
_TEST_ONLY_CONFIG = _SCORING_FIXTURE["test_only"]
_TEST_BOX_TIERS = {
    tier: {
        "price": price,
        "target_value": round(price * _TEST_ONLY_CONFIG["box_target_pct"] / 100),
    }
    for tier, price in _TEST_ONLY_CONFIG["box_prices"].items()
}


def _collect_module_scope_dependency(
    tmp_path,
    *pytest_args,
    dependency="a_module_that_does_not_exist_xyz",
    version_overrides=None,
    absent_distributions=(),
    running_pytest_version=None,
):
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "import json, os\n"
        "from importlib import metadata\n"
        "_real_version = metadata.version\n"
        "_overrides = json.loads(os.environ.get('DIAGNOSTIC_VERSION_OVERRIDES', '{}'))\n"
        "_absent = set(json.loads(os.environ.get('DIAGNOSTIC_ABSENT_DISTRIBUTIONS', '[]')))\n"
        "_running_pytest = os.environ.get('DIAGNOSTIC_RUNNING_PYTEST_VERSION')\n"
        "if _running_pytest:\n"
        "    import pytest\n"
        "    pytest.__version__ = _running_pytest\n"
        "def _version(name):\n"
        "    if name in _absent:\n"
        "        raise metadata.PackageNotFoundError(name)\n"
        "    return _overrides.get(name, _real_version(name))\n"
        "metadata.version = _version\n"
    )
    test_module = tmp_path / "test_diagnostic_dependency.py"
    test_module.write_text(
        "import pytest\n"
        "from tests.conftest import require_dep\n"
        "pytestmark = pytest.mark.diagnostics(reason='synthetic')\n"
        f"require_dep({dependency!r})\n"
        "def test_unreachable():\n"
        "    raise AssertionError('dependency gate did not stop collection')\n"
    )
    (tmp_path / "test_sentinel.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.diagnostics(reason='synthetic')\n"
        "def test_pytest_completed_collection():\n"
        "    pass\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(_PROJECT_ROOT)))
    env["DIAGNOSTIC_VERSION_OVERRIDES"] = json.dumps(version_overrides or {})
    env["DIAGNOSTIC_ABSENT_DISTRIBUTIONS"] = json.dumps(absent_distributions)
    if running_pytest_version is not None:
        env["DIAGNOSTIC_RUNNING_PYTEST_VERSION"] = running_pytest_version
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_PROJECT_ROOT / "pyproject.toml"),
            "-p",
            "tests.conftest",
            *pytest_args,
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def test_require_dep_returns_module_when_present():
    # `json` is always importable, so this exercises the success path in both
    # strict and non-strict modes without needing the diagnostics stack.
    mod = require_dep("json")
    assert mod.dumps({"a": 1}) == '{"a": 1}'


def test_require_dep_skips_when_absent_and_not_strict():
    # Outside explicit strict mode, a missing dependency must skip, not error.
    from tests import conftest

    assert conftest._STRICT is False
    with pytest.raises(BaseException) as exc:
        require_dep("a_module_that_does_not_exist_xyz")
    assert exc.typename == "Skipped"


def test_module_scope_require_dep_skips_in_plain_pytest(tmp_path):
    proc = _collect_module_scope_dependency(tmp_path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 skipped" in proc.stdout


def test_module_scope_require_dep_fails_in_explicit_strict_mode(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m",
        "diagnostics",
        "--strict-diagnostics-deps",
    )

    assert proc.returncode == pytest.ExitCode.INTERRUPTED
    output = proc.stdout + proc.stderr
    assert "ImportError: required diagnostic dependency" in output
    assert "a_module_that_does_not_exist_xyz" in output


def test_marker_keyword_selection_does_not_arm_dependency_strictness(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m",
        "diagnostics(reason='synthetic')",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed, 1 skipped" in proc.stdout


def test_dependency_floors_are_loaded_from_the_manifest(tmp_path):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text(
        "# diagnostic stack\n"
        "example-lib>=1.2.3\n"
        "scikit-learn>=9.8.7\n"
    )

    assert _load_diagnostic_dependencies(manifest) == {
        "example_lib": ("example-lib", "1.2.3"),
        "sklearn": ("scikit-learn", "9.8.7"),
    }


def test_strict_manifest_rejects_duplicate_canonical_distributions(tmp_path):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text("numpy>=999\nNumPy>=1\n")

    with pytest.raises(ValueError, match="duplicate distribution.*NumPy"):
        _load_diagnostic_dependencies(manifest, strict=True)


def test_plain_manifest_keeps_strongest_canonical_duplicate_floor(tmp_path):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text("Example_Lib>=999.0\nexample-lib>=1\n")

    assert _load_diagnostic_dependencies(manifest) == {
        "example_lib": ("Example_Lib", "999.0")
    }


def test_strict_manifest_rejects_import_module_alias_collisions(tmp_path):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text("scikit-learn>=999\nsklearn>=1\n")

    with pytest.raises(ValueError, match="duplicate import module.*sklearn"):
        _load_diagnostic_dependencies(manifest, strict=True)


def test_production_manifest_declares_every_diagnostic_module():
    from tests.conftest import _DIAGNOSTIC_DEPENDENCIES

    assert set(_DIAGNOSTIC_DEPENDENCIES) == {
        "pytest",
        "packaging",
        "interpret",
        "statsmodels",
        "sklearn",
        "numpy",
        "pandas",
        "numexpr",
        "bottleneck",
    }
    assert _DIAGNOSTIC_DEPENDENCIES["sklearn"][0] == "scikit-learn"


def test_manifest_parser_is_tolerant_for_plain_collection_but_strict_on_demand(
    tmp_path,
):
    from tests.conftest import _load_diagnostic_dependencies

    manifest = tmp_path / "requirements-diagnostics.txt"
    manifest.write_text(
        "valid>=1.0\n"
        "conditional>=2; python_version >= '3.10'\n"
        "pinned==3\n"
    )

    assert _load_diagnostic_dependencies(manifest) == {
        "valid": ("valid", "1.0")
    }
    with pytest.raises(ValueError, match="expected distribution>=minimum"):
        _load_diagnostic_dependencies(manifest, strict=True)


def test_missing_manifest_is_empty_in_plain_mode_and_errors_in_strict_mode(
    tmp_path,
):
    from tests.conftest import _load_diagnostic_dependencies

    missing = tmp_path / "missing-requirements.txt"
    assert _load_diagnostic_dependencies(missing) == {}
    with pytest.raises(ValueError, match="missing-requirements.txt.*does not exist"):
        _load_diagnostic_dependencies(missing, strict=True)


def test_bootstrap_dependency_must_be_present_in_manifest():
    from tests.conftest import _validate_bootstrap_dependencies

    with pytest.raises(pytest.UsageError, match="packaging.*manifest"):
        _validate_bootstrap_dependencies({
            "pytest": ("pytest", "9.0.0"),
        })


def test_conftest_does_not_import_packaging_version_at_module_scope():
    code = r'''\
import importlib.abc
import runpy
import sys
import packaging
import pytest

sys.modules.pop("packaging.version", None)
if hasattr(packaging, "version"):
    delattr(packaging, "version")

class BlockPackagingVersion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "packaging.version":
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockPackagingVersion())
runpy.run_path("tests/conftest.py", run_name="conftest_import_probe")
'''
    provisioned_paths = [
        _PROJECT_ROOT / "scoring_config.json",
        _PROJECT_ROOT / "identifiers.json",
    ]
    absent_before = {path for path in provisioned_paths if not path.exists()}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
    finally:
        for path in absent_before:
            if path.exists():
                path.unlink()
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_plain_pytest_skips_when_packaging_distribution_is_absent(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path, dependency="packaging", absent_distributions=["packaging"]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 skipped" in proc.stdout


def test_strict_mode_rejects_absent_packaging_distribution(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m",
        "diagnostics",
        "--strict-diagnostics-deps",
        dependency="json",
        absent_distributions=["packaging"],
    )
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    assert "packaging" in proc.stdout + proc.stderr
    assert "absent" in proc.stdout + proc.stderr


def test_strict_mode_rejects_below_floor_packaging(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m",
        "diagnostics",
        "--strict-diagnostics-deps",
        dependency="json",
        version_overrides={"packaging": "21.9"},
    )
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    output = proc.stdout + proc.stderr
    assert "packaging>=22" in output
    assert "found 21.9" in output


def test_strict_mode_checks_the_running_pytest_version(tmp_path):
    proc = _collect_module_scope_dependency(
        tmp_path,
        "-m",
        "diagnostics",
        "--strict-diagnostics-deps",
        dependency="json",
        version_overrides={"pytest": "99.0.0"},
        running_pytest_version="8.4.2",
    )
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    output = proc.stdout + proc.stderr
    assert "pytest>=9.0.0" in output
    assert "found 8.4.2" in output


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


def test_require_dep_rejects_below_floor_running_module_despite_newer_metadata(
    monkeypatch,
):
    from tests import conftest

    class StaleSklearn:
        __version__ = "0.1.0"

    monkeypatch.setattr(conftest, "_STRICT", True)
    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: StaleSklearn())
    monkeypatch.setattr(conftest.importlib_metadata, "version", lambda name: "99.0.0")

    with pytest.raises(ImportError, match="running module.*found 0.1.0"):
        require_dep("sklearn")


def test_require_dep_accepts_running_version_object_at_floor(monkeypatch):
    from packaging.version import Version
    from tests import conftest

    class SklearnAtFloor:
        __version__ = Version("1.3.0")

    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: SklearnAtFloor())
    monkeypatch.setattr(conftest.importlib_metadata, "version", lambda name: "1.3.0")

    assert isinstance(require_dep("sklearn"), SklearnAtFloor)


def test_require_dep_skips_invalid_running_version_in_plain_mode(monkeypatch):
    from tests import conftest

    class InvalidSklearn:
        __version__ = "not-a-version"

    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: InvalidSklearn())
    monkeypatch.setattr(conftest.importlib_metadata, "version", lambda name: "99.0.0")

    with pytest.raises(BaseException) as exc:
        require_dep("sklearn")
    assert exc.typename == "Skipped"
    assert "running module version" in str(exc.value)


def test_require_dep_rejects_explicit_none_running_version_in_strict_mode(
    monkeypatch,
):
    from tests import conftest

    class InvalidSklearn:
        __version__ = None

    monkeypatch.setattr(conftest, "_STRICT", True)
    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: InvalidSklearn())
    monkeypatch.setattr(conftest.importlib_metadata, "version", lambda name: "99.0.0")

    with pytest.raises(ImportError, match="invalid running module version None"):
        require_dep("sklearn")


def test_require_dep_rejects_invalid_installed_version_in_strict_mode(monkeypatch):
    from tests import conftest

    class CurrentSklearn:
        __version__ = "99.0.0"

    monkeypatch.setattr(conftest, "_STRICT", True)
    monkeypatch.setattr(conftest.importlib, "import_module", lambda name: CurrentSklearn())
    monkeypatch.setattr(
        conftest.importlib_metadata, "version", lambda name: "not-a-version"
    )

    with pytest.raises(ImportError, match="installed distribution version"):
        require_dep("sklearn")


def test_require_dep_raises_when_absent_and_strict(monkeypatch):
    from tests import conftest

    def absent(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(conftest, "_STRICT", True)
    monkeypatch.setattr(conftest.importlib, "import_module", absent)
    with pytest.raises(ImportError, match="required diagnostic dependency"):
        require_dep("statsmodels")


# ---------------------------------------------------------------------------
# Shared fixture data for the box-feature tests
# ---------------------------------------------------------------------------
#
# Values come from tests/fixtures/scoring_config.json (synthetic): fungible
# groups apple/banana/tomato, quantity classes snack_piece and cooking_piece,
# and deliberately non-production test-only box/value scalars.


@pytest.fixture(autouse=True)
def _use_synthetic_scoring_config(monkeypatch):
    """Keep box-feature expectations independent of ignored local config links."""
    import allocator.box_features as box_features
    import allocator.config as allocator_config
    import allocator.strategies._scoring as scoring

    config = _SCORING_FIXTURE
    value_band = _TEST_ONLY_CONFIG["value_band"]
    monkeypatch.setattr(box_features, "CATEGORY_FRUIT", config["category_fruit"])
    monkeypatch.setattr(box_features, "CATEGORY_VEGETABLES", config["category_vegetables"])
    monkeypatch.setattr(box_features, "BOX_TIERS", _TEST_BOX_TIERS)
    monkeypatch.setattr(
        box_features,
        "BOX_TARGET_PCT",
        _TEST_ONLY_CONFIG["box_target_pct"],
    )
    monkeypatch.setattr(
        allocator_config,
        "BOX_TARGET_PCT",
        _TEST_ONLY_CONFIG["box_target_pct"],
    )
    monkeypatch.setattr(allocator_config, "BOX_TIERS", _TEST_BOX_TIERS)
    monkeypatch.setattr(box_features, "GROUP_ALLOWANCES", config["group_allowances"])
    monkeypatch.setattr(allocator_config, "GROUP_ALLOWANCES", config["group_allowances"])
    classifications = {
        key: (value[0], value[1], value[2], value[3], value[4])
        for key, value in config["item_classifications"].items()
    }
    fallback = {
        config["category_fruit"]: tuple(config["classification_fallback"]["fruit"]),
        config["category_vegetables"]: tuple(config["classification_fallback"]["veg"]),
    }
    monkeypatch.setattr(
        box_features, "ITEM_CLASSIFICATIONS", classifications, raising=False
    )
    monkeypatch.setattr(
        box_features, "CLASSIFICATION_FALLBACK", fallback, raising=False
    )
    monkeypatch.setattr(allocator_config, "ITEM_CLASSIFICATIONS", classifications)
    monkeypatch.setattr(allocator_config, "CLASSIFICATION_FALLBACK", fallback)
    monkeypatch.setattr(allocator_config, "FUNGIBLE_GROUPS", {
        key: (value[0], value[1], value[2] if len(value) > 2 else "portioned")
        for key, value in config["fungible_groups"].items()
    })
    monkeypatch.setattr(
        allocator_config, "QUANTITY_CLASSES", config["quantity_classes"]
    )
    monkeypatch.setattr(
        allocator_config,
        "QTY_CLASS_PRICE_THRESHOLDS",
        config["qty_class_price_thresholds"],
    )
    monkeypatch.setattr(
        allocator_config,
        "VALUE_SWEET_FROM",
        value_band["sweet_from"],
    )
    monkeypatch.setattr(
        allocator_config,
        "VALUE_SWEET_TO",
        value_band["sweet_to"],
    )
    monkeypatch.setattr(
        allocator_config,
        "VALUE_PENALTY_EXPONENT",
        value_band["penalty_exponent"],
    )
    monkeypatch.setattr(
        box_features,
        "VALUE_SWEET_FROM",
        value_band["sweet_from"],
    )
    monkeypatch.setattr(
        box_features,
        "VALUE_SWEET_TO",
        value_band["sweet_to"],
    )
    monkeypatch.setattr(
        box_features,
        "VALUE_PENALTY_EXPONENT",
        value_band["penalty_exponent"],
    )
    monkeypatch.setattr(scoring, "FUNGIBLE_GROUPS", config["fungible_groups"])
    monkeypatch.setattr(scoring, "QUANTITY_CLASSES", config["quantity_classes"])
    monkeypatch.setattr(
        scoring,
        "QTY_CLASS_PRICE_THRESHOLDS",
        config["qty_class_price_thresholds"],
    )


def _item_lookup():
    """Three items spanning both categories, all three fungible groups."""
    config = _SCORING_FIXTURE
    category_fruit = config["category_fruit"]
    category_vegetables = config["category_vegetables"]

    return {
        1: {"name": "Apples - Fuji", "price": 100, "size": 1,
            "category_id": category_fruit, "fungible_group": "apple",
            "fungible_degree": 0.7, "sub_category": "pome_fruit",
            "usage": "snacking", "colour": "red", "shape": "round"},
        2: {"name": "Bananas - Cavendish", "price": 150, "size": 2,
            "category_id": category_fruit, "fungible_group": "banana",
            "fungible_degree": 1.0, "sub_category": "tropical",
            "usage": "snacking", "colour": "yellow", "shape": "long"},
        3: {"name": "Tomatoes - Roma", "price": 200, "size": 1,
            "category_id": category_vegetables, "fungible_group": "tomato",
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
      value_pct    = total_value / synthetic small-box price * 100
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


def test_box_features_module_imports_from_isolated_root_without_db(tmp_path):
    """The module must not reach allocator.db, directly or transitively."""
    isolated_root = tmp_path / "isolated-project"
    shutil.copytree(
        _PROJECT_ROOT / "allocator",
        isolated_root / "allocator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(_SCORING_FIXTURE_PATH, isolated_root / "scoring_config.json")
    shutil.copy2(
        _PROJECT_ROOT / "tests" / "fixtures" / "identifiers.json",
        isolated_root / "identifiers.json",
    )

    env = os.environ.copy()
    for tier, price in _TEST_ONLY_CONFIG["box_prices"].items():
        env[f"BOX_PRICE_{tier.upper()}"] = str(price)
    env["BOX_TARGET_PCT"] = str(_TEST_ONLY_CONFIG["box_target_pct"])
    value_band = _TEST_ONLY_CONFIG["value_band"]
    env["VALUE_SWEET_FROM"] = str(value_band["sweet_from"])
    env["VALUE_SWEET_TO"] = str(value_band["sweet_to"])
    env["VALUE_PENALTY_EXPONENT"] = str(value_band["penalty_exponent"])
    env["PYTHONPATH"] = str(isolated_root)

    # A subprocess with an import hook that hard-fails on allocator.db proves the
    # absence of the transitive path, which a plain import in this process cannot
    # (conftest may already have imported db via another test module).
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name in ('allocator.db', 'compare', 'scripts.extract_features'):\n"
        "            raise AssertionError('box_features imported ' + name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import allocator.box_features as box_features\n"
        "assert Path(box_features.__file__).resolve().is_relative_to("
        "Path(sys.argv[1]).resolve())\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(isolated_root)],
        cwd=isolated_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_extract_features_batch_helpers_import_without_database():
    code = (
        "import sys\n"
        "import scripts.extract_features\n"
        "assert 'compare' not in sys.modules\n"
        "assert 'allocator.db' not in sys.modules\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_extract_box_features_scalar_fields():
    rec = _record()
    assert rec["offer_id"] == 999
    assert rec["box_name"] == "test@example.com"
    assert rec["tier"] == "small"
    assert rec["source"] == "manual"
    expected_value_pct = round(
        850 / _TEST_BOX_TIERS["small"]["price"] * 100,
        4,
    )
    assert rec["value_pct"] == expected_value_pct
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


def test_extract_box_features_uses_synthetic_price_threshold_for_allowance():
    from allocator.box_features import extract_box_features

    config = _SCORING_FIXTURE
    price = config["qty_class_price_thresholds"]["snacking_max"]
    lookup = _item_lookup()
    lookup[4] = {
        **lookup[1],
        "name": "Unclassified Snack",
        "price": price,
        "fungible_group": None,
        "fungible_degree": 0.0,
    }

    rec = extract_box_features(
        box_name="x",
        allocations={4: 2},
        item_lookup=lookup,
        tier="small",
        available_tags=_available_tags(),
        offer_id=1,
    )

    assert rec["item_quantities"] == [
        [2, price, config["quantity_classes"]["cooking_piece"]["small"]]
    ]


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


def test_raw_group_totals_are_uncapped():
    """Uncapped loads, over the same key space as group_totals."""
    rec = _record()
    # allocations 3 apple / 1 banana / 2 tomato; allowances 2 / 2 / 1
    assert rec["raw_group_totals"] == {"apple": 3, "banana": 1, "tomato": 2}


def test_capped_group_totals_match_group_totals_loads():
    """capped_group_totals supplies the key that group_totals' positional
    triples cannot — the aggregate each raw_group_totals column pairs with."""
    rec = _record()
    assert rec["capped_group_totals"] == {"apple": 2, "banana": 1, "tomato": 1}
    assert sorted(rec["capped_group_totals"].values()) == sorted(
        load for load, _degree, _allow in rec["group_totals"]
    )


def test_ungrouped_items_produce_no_group_columns():
    """__item_N synthetic keys must not leak into either group dict."""
    from allocator.box_features import extract_box_features

    lookup = _item_lookup()
    lookup[4] = {"name": "Kiwi - Green", "price": 90, "size": 1,
                 "category_id": lookup[1]["category_id"], "fungible_group": None,
                 "fungible_degree": 0.0, "sub_category": "tropical",
                 "usage": "snacking", "colour": "green", "shape": "round"}
    rec = extract_box_features(
        box_name="x", allocations={1: 3, 4: 5}, item_lookup=lookup,
        tier="small", available_tags=_available_tags(), offer_id=1,
    )
    assert set(rec["raw_group_totals"]) == {"apple"}
    assert set(rec["capped_group_totals"]) == {"apple"}
    assert not any(k.startswith("__item_") for k in rec["raw_group_totals"])


def test_raw_tag_counts_are_quantity_weighted_and_sparse():
    rec = _record()
    assert rec["raw_tag_counts"] == {
        "sub_category": {"pome_fruit": 3, "tropical": 1, "fruiting_veg": 2},
        "usage": {"snacking": 4, "cooking": 2},
        "colour": {"red": 5, "yellow": 1},
        "shape": {"round": 5, "long": 1},
    }
    # Absent tags are omitted, not zero-filled.
    assert "root_veg" not in rec["raw_tag_counts"]["sub_category"]


def test_raw_tag_counts_each_dimension_sums_to_resolved_qty():
    """assign_classification always returns all four tags and the loop adds qty
    to exactly one tag per dimension per item, so every dimension block sums to
    the box's resolved allocated quantity. Three exact linear dependencies —
    stated here so a future change that breaks them fails loudly."""
    rec = _record()
    total_qty = 3 + 1 + 2
    for dim, counts in rec["raw_tag_counts"].items():
        assert sum(counts.values()) == total_qty, dim


def test_category_value_share_has_exactly_two_keys():
    rec = _record()
    assert set(rec["category_value_share"]) == {"fruit", "veg"}
    assert rec["category_value_share"]["fruit"] == 0.529412   # (300+150)/850
    assert rec["category_value_share"]["veg"] == 0.470588     # 400/850
    assert sum(rec["category_value_share"].values()) == 1.0


def test_category_value_share_sums_exactly_for_rounding_edge_case():
    """Complementary rounded shares must retain the two-key sum invariant."""
    from allocator.box_features import extract_box_features

    lookup = _item_lookup()
    lookup[1] = {**lookup[1], "price": 7}
    lookup[3] = {**lookup[3], "price": 633}
    rec = extract_box_features(
        box_name="x", allocations={1: 1, 3: 1}, item_lookup=lookup,
        tier="small", available_tags=_available_tags(), offer_id=1,
    )
    assert sum(rec["category_value_share"].values()) == 1.0


def test_category_value_share_is_zero_for_a_valueless_box():
    from allocator.box_features import extract_box_features

    lookup = _item_lookup()
    lookup[1] = {**lookup[1], "price": 0}
    rec = extract_box_features(
        box_name="x", allocations={1: 2}, item_lookup=lookup, tier="small",
        available_tags=_available_tags(), offer_id=1,
    )
    assert rec["category_value_share"] == {"fruit": 0.0, "veg": 0.0}


def test_a_third_category_raises_compatible_specific_error():
    """The pair sums to 1 only while every resolved item is fruit or veg —
    which is what lets flatten() drop the veg column. A third category is a
    schema event, not a data point: it needs a third column and a re-derived
    maxT family. Fail where it is introduced, not silently in the matrix."""
    from allocator.box_features import UnsupportedCategoryError, extract_box_features

    lookup = _item_lookup()
    lookup[4] = {**lookup[1], "name": "Mystery Herb", "price": 500,
                 "category_id": 987654, "fungible_group": None,
                 "fungible_degree": 0.0}
    with pytest.raises(UnsupportedCategoryError, match="neither CATEGORY_FRUIT") as exc:
        extract_box_features(
            box_name="x", allocations={1: 3, 3: 2, 4: 1}, item_lookup=lookup,
            tier="small", available_tags=_available_tags(), offer_id=1,
        )
    assert isinstance(exc.value, ValueError)


def test_batch_boundary_skips_unsupported_record_and_allows_next(capsys):
    from scripts.extract_features import _extract_or_skip

    lookup = _item_lookup()
    lookup[4] = {**lookup[1], "category_id": 987654,
                 "fungible_group": None, "fungible_degree": 0.0}
    skipped = _extract_or_skip(
        "bad", {4: 1}, lookup, "small", _available_tags(), 10
    )
    accepted = _extract_or_skip(
        "good", {1: 1}, lookup, "small", _available_tags(), 10
    )

    assert skipped is None
    assert accepted["box_name"] == "good"
    assert "[SKIP]" in capsys.readouterr().out


def test_synthetic_generation_continues_after_unsupported_candidate(monkeypatch):
    import scripts.extract_features as script
    from allocator.box_features import UnsupportedCategoryError

    def selective_extract(box_name, *args, **kwargs):
        if box_name == "synth_mono_small":
            raise UnsupportedCategoryError("unsupported synthetic")
        if box_name == "synth_random_small":
            return {"box_name": box_name, "source": "synth_random"}
        return None

    monkeypatch.setattr(script, "extract_box_features", selective_extract)
    features = script.generate_synthetic_boxes(1, _item_lookup(), _available_tags())
    assert features == [{"box_name": "synth_random_small", "source": "synth_random"}]


def test_batch_boundary_does_not_swallow_unrelated_value_error(monkeypatch):
    import scripts.extract_features as script

    def broken(*args, **kwargs):
        raise ValueError("unrelated")

    monkeypatch.setattr(script, "extract_box_features", broken)
    with pytest.raises(ValueError, match="unrelated"):
        script._extract_or_skip("x", {}, {}, "small", {}, 1)


def test_a_zero_price_third_category_is_still_refused():
    """The guard is about the resolved category schema, not value truthiness."""
    from allocator.box_features import extract_box_features

    lookup = _item_lookup()
    lookup[4] = {**lookup[1], "name": "Free Mystery Herb", "price": 0,
                 "category_id": 987654, "fungible_group": None,
                 "fungible_degree": 0.0}
    with pytest.raises(ValueError, match="item 4.*category 987654"):
        extract_box_features(
            box_name="x", allocations={1: 3, 4: 1}, item_lookup=lookup,
            tier="small", available_tags=_available_tags(), offer_id=1,
        )


def test_an_unresolved_item_does_not_trigger_the_third_category_guard():
    """Items absent from item_lookup are skipped before the category check —
    they contribute to neither total_value nor either numerator, so the sum
    holds. Only *resolved* items in a third category are an error."""
    from allocator.box_features import extract_box_features

    rec = extract_box_features(
        box_name="x", allocations={1: 3, 2: 1, 3: 2, 99: 5},
        item_lookup=_item_lookup(), tier="small",
        available_tags=_available_tags(), offer_id=1,
    )
    assert sum(rec["category_value_share"].values()) == 1.0


def test_tag_vocabulary_is_dimension_qualified_and_sorted():
    from allocator.box_features import tag_vocabulary

    vocab = tag_vocabulary()
    assert vocab == sorted(vocab)
    assert all("." in entry for entry in vocab)
    assert len(vocab) == len(set(vocab))


def test_tag_vocabulary_includes_all_three_fallback_sub_categories():
    """Fallbacks reserve columns before an unclassified live item arrives."""
    from allocator.box_features import tag_vocabulary

    vocab = set(tag_vocabulary())
    assert "sub_category.other_fruit" in vocab
    assert "sub_category.other_veg" in vocab
    assert "sub_category.other" in vocab
    assert "colour.green" in vocab


def test_tag_vocabulary_size_matches_config():
    from allocator.box_features import tag_vocabulary
    from allocator.categorizer import DEFAULT_CLASSIFICATION
    from allocator.config import CLASSIFICATION_FALLBACK, ITEM_CLASSIFICATIONS

    dims = ["sub_category", "usage", "colour", "shape"]
    expected = {dimension: set() for dimension in dims}
    for _prefixes, sub_cat, usage, colour, shape in ITEM_CLASSIFICATIONS.values():
        for dimension, tag in zip(dims, (sub_cat, usage, colour, shape)):
            expected[dimension].add(tag)
    for fallback in list(CLASSIFICATION_FALLBACK.values()) + [DEFAULT_CLASSIFICATION]:
        for dimension, tag in zip(dims, fallback):
            expected[dimension].add(tag)

    assert len(tag_vocabulary()) == sum(len(tags) for tags in expected.values()) == 15


def test_flatten_column_count_derived_from_config():
    from allocator.box_features import flatten, tag_vocabulary
    from allocator.config import GROUP_ALLOWANCES

    cols = flatten(_record())
    expected = 3 + 8 + 3 + 5 + 1 + 2 * len(GROUP_ALLOWANCES) + len(tag_vocabulary())
    assert len(cols) == expected == 41


def test_flatten_is_globally_name_sorted():
    from allocator.box_features import flatten

    names = list(flatten(_record()))
    assert names == sorted(names)


def test_flatten_column_set_is_identical_across_disjoint_boxes():
    from allocator.box_features import extract_box_features, flatten

    only_apple = extract_box_features(
        box_name="a", allocations={1: 2}, item_lookup=_item_lookup(), tier="small",
        available_tags=_available_tags(), offer_id=1,
    )
    only_tomato = extract_box_features(
        box_name="b", allocations={3: 1}, item_lookup=_item_lookup(), tier="small",
        available_tags=_available_tags(), offer_id=1,
    )
    apple, tomato = flatten(only_apple), flatten(only_tomato)
    assert list(apple) == list(tomato)
    assert apple["raw_group_totals.tomato"] == 0.0
    assert tomato["raw_group_totals.apple"] == 0.0
    assert apple["raw_tag_counts.sub_category.fruiting_veg"] == 0.0
    assert tomato["raw_tag_counts.sub_category.pome_fruit"] == 0.0


def test_flatten_tier_slices_value_pct():
    from allocator.box_features import flatten

    cols = flatten(_record())
    assert cols["value_pct_small"] == _record()["value_pct"]
    assert cols["value_pct_medium"] == 0.0
    assert cols["value_pct_large"] == 0.0
    assert "value_pct" not in cols


def test_flatten_rejects_tier_outside_fixed_matrix_schema():
    from allocator.box_features import flatten

    record = _record()
    record["tier"] = "extra_large"

    with pytest.raises(ValueError, match="unsupported tier.*extra_large"):
        flatten(record)


def test_flatten_price_stats_are_item_weighted():
    import statistics

    from allocator.box_features import flatten

    cols = flatten(_record())
    assert cols["n_unique_items"] == 3.0
    assert cols["total_qty"] == 6.0
    assert cols["price_mean"] == 150.0
    assert cols["price_max"] == 200.0
    assert cols["price_sd"] == statistics.pstdev([100, 150, 200])


def test_flatten_price_sd_is_zero_for_a_single_item_box():
    from allocator.box_features import extract_box_features, flatten

    record = extract_box_features(
        box_name="a", allocations={1: 4}, item_lookup=_item_lookup(), tier="small",
        available_tags=_available_tags(), offer_id=1,
    )
    assert flatten(record)["price_sd"] == 0.0


def test_flatten_emits_one_category_column():
    from allocator.box_features import flatten

    cols = flatten(_record())
    assert cols["fruit_value_share"] == 0.529412
    assert "veg_value_share" not in cols
    assert "category_value_share.veg" not in cols


def test_flatten_excludes_config_parameter_columns():
    from allocator.box_features import flatten

    names = list(flatten(_record()))
    assert not any(name.startswith("item_quantities") for name in names)
    assert not any(name.startswith("group_totals") for name in names)
    assert not any("allowance" in name for name in names)
    assert not any("degree" in name for name in names)


def test_config_hash_is_sixteen_hex_chars_and_stable():
    from allocator.box_features import config_hash

    h = config_hash()
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)
    assert h == config_hash()


def test_feature_schema_version_and_public_stable_hash():
    import allocator.box_features as features

    assert features.FEATURE_SCHEMA_VERSION == 2
    assert features.stable_hash({"b": [2], "a": {1}}) == (
        features.stable_hash({"a": {1}, "b": [2]})
    )
    assert features.config_hash() == features.stable_hash(features._hash_inputs())


@pytest.mark.parametrize(
    ("owner", "rules"),
    [
        (
            "ITEM_CLASSIFICATIONS",
            [
                ("broad", (["Apples -"], "broad", "cooking", "green", "round")),
                (
                    "specific",
                    (["Apples - Royal Gala"], "specific", "snacking", "red", "round"),
                ),
            ],
        ),
        (
            "FUNGIBLE_GROUPS",
            [
                ("broad", (0.5, ["Apples -"], "cooking_piece")),
                ("specific", (1.0, ["Apples - Royal Gala"], "snack_piece")),
            ],
        ),
    ],
)
def test_first_match_rule_order_changes_hash_and_snapshot(monkeypatch, owner, rules):
    """Reordering overlapping rules changes classification/group behaviour."""
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    target = scoring if owner == "FUNGIBLE_GROUPS" else box_features
    monkeypatch.setattr(target, owner, dict(rules))
    before = (box_features.config_hash(), box_features.config_snapshot())

    monkeypatch.setattr(target, owner, dict(reversed(rules)))

    assert (box_features.config_hash(), box_features.config_snapshot()) != before


@pytest.mark.parametrize("name", [
    "BOX_TIERS",
    "GROUP_ALLOWANCES",
    "ITEM_CLASSIFICATIONS",
    "CLASSIFICATION_FALLBACK",
    "VALUE_SWEET_FROM",
    "VALUE_SWEET_TO",
    "VALUE_PENALTY_EXPONENT",
])
def test_local_config_owner_changes_hash(monkeypatch, name):
    """Changing an effective box_features binding must restamp feature data."""
    import allocator.box_features as box_features

    before = box_features.config_hash()
    current = getattr(box_features, name)
    if isinstance(current, (int, float)):
        mutated = current + 1
    else:
        sample = next(iter(current))
        probe = max(current) + 1 if isinstance(sample, int) else "zzz_probe"
        mutated = {**current, probe: 1}
    monkeypatch.setattr(box_features, name, mutated)
    assert box_features.config_hash() != before


def test_default_classification_owner_changes_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    monkeypatch.setattr(
        box_features,
        "DEFAULT_CLASSIFICATION",
        ("reserved_default", "cooking", "green", "round"),
    )
    assert box_features.config_hash() != before_hash
    assert (
        box_features.config_snapshot()["default_classification_hash"]
        != before_snapshot["default_classification_hash"]
    )


def test_box_tier_owner_carries_target_percentage_into_hash(monkeypatch):
    import allocator.box_features as box_features

    before = box_features.config_hash()
    tiers = {tier: dict(values) for tier, values in box_features.BOX_TIERS.items()}
    tiers["small"]["target_value"] += 1
    monkeypatch.setattr(box_features, "BOX_TIERS", tiers)
    assert box_features.config_hash() != before


def test_config_snapshot_has_exactly_the_pinned_keys():
    from allocator.box_features import config_snapshot

    snap = config_snapshot()
    assert set(snap) == {
        "box_tiers", "box_target_pct", "category_fruit", "category_vegetables",
        "value_sweet_from", "value_sweet_to", "value_penalty_exponent",
        "group_allowances", "quantity_classes", "qty_class_price_thresholds",
        "item_classifications_hash", "fungible_groups_hash",
        "classification_fallback_hash", "default_classification_hash",
    }


def test_config_snapshot_box_tiers_carry_price_and_target_value():
    from allocator.box_features import config_snapshot
    from allocator.config import BOX_TIERS

    snap = config_snapshot()
    assert set(snap["box_tiers"]) == {"small", "medium", "large"}
    for tier, entry in snap["box_tiers"].items():
        assert set(entry) == {"price", "target_value"}
        assert entry["price"] == BOX_TIERS[tier]["price"]
        assert entry["target_value"] == BOX_TIERS[tier]["target_value"]


def test_config_snapshot_classification_structures_are_digests():
    from allocator.box_features import config_snapshot

    snap = config_snapshot()
    for key in (
        "item_classifications_hash",
        "fungible_groups_hash",
        "classification_fallback_hash",
        "default_classification_hash",
    ):
        assert isinstance(snap[key], str)
        assert len(snap[key]) == 16


def test_classification_fallback_owner_changes_snapshot(monkeypatch):
    import allocator.box_features as box_features

    before = box_features.config_snapshot()
    fallback = dict(box_features.CLASSIFICATION_FALLBACK)
    category_id = next(iter(fallback))
    fallback[category_id] = ("reserved_fallback", *fallback[category_id][1:])
    monkeypatch.setattr(box_features, "CLASSIFICATION_FALLBACK", fallback)
    assert (
        box_features.config_snapshot()["classification_fallback_hash"]
        != before["classification_fallback_hash"]
    )


@pytest.mark.parametrize("name", ["CATEGORY_FRUIT", "CATEGORY_VEGETABLES"])
def test_category_id_owner_changes_invalidate_both_identity_contracts(monkeypatch, name):
    import allocator.box_features as box_features

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    monkeypatch.setattr(box_features, name, getattr(box_features, name) + 1000)
    after_snapshot = box_features.config_snapshot()

    assert box_features.config_hash() != before_hash
    assert after_snapshot != before_snapshot
    assert after_snapshot[
        "category_fruit" if name == "CATEGORY_FRUIT" else "category_vegetables"
    ] == getattr(box_features, name)


def test_frozen_owner_ignores_live_config_and_categorizer_reassignment(
    monkeypatch,
):
    import allocator.box_features as box_features
    import allocator.categorizer as categorizer
    import allocator.config as config

    before = (box_features.config_hash(), box_features.config_snapshot())
    monkeypatch.setattr(config, "CATEGORY_FRUIT", config.CATEGORY_FRUIT + 1000)
    monkeypatch.setattr(
        config,
        "BOX_TIERS",
        {"changed": {"price": 1, "target_value": 1}},
    )
    monkeypatch.setattr(categorizer, "DEFAULT_CLASSIFICATION", ("changed",) * 4)
    assert (box_features.config_hash(), box_features.config_snapshot()) == before


def test_quantity_class_owner_changes_allowance_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    classes = {name: dict(values) for name, values in scoring.QUANTITY_CLASSES.items()}
    classes["snack_piece"]["small"] += 7
    monkeypatch.setattr(scoring, "QUANTITY_CLASSES", classes)

    record = box_features.extract_box_features(
        "x", {1: 1}, _item_lookup(), "small", _available_tags(), 1
    )
    assert record["item_quantities"][0][2] == classes["snack_piece"]["small"]
    assert box_features.config_hash() != before_hash
    assert box_features.config_snapshot() != before_snapshot


def test_fungible_group_owner_changes_allowance_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    groups = dict(scoring.FUNGIBLE_GROUPS)
    degree, prefixes, _quantity_class = groups["apple"]
    groups["apple"] = (degree, prefixes, "cooking_piece")
    monkeypatch.setattr(scoring, "FUNGIBLE_GROUPS", groups)

    record = box_features.extract_box_features(
        "x", {1: 1}, _item_lookup(), "small", _available_tags(), 1
    )
    assert record["item_quantities"][0][2] == scoring.QUANTITY_CLASSES[
        "cooking_piece"
    ]["small"]
    assert box_features.config_hash() != before_hash
    assert box_features.config_snapshot() != before_snapshot


def test_price_threshold_owner_changes_allowance_hash_and_snapshot(monkeypatch):
    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    lookup = _item_lookup()
    original_threshold = scoring.QTY_CLASS_PRICE_THRESHOLDS["snacking_max"]
    lookup[4] = {
        **lookup[1],
        "price": original_threshold - 1,
        "fungible_group": None,
        "fungible_degree": 0.0,
    }
    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    before_record = box_features.extract_box_features(
        "x", {4: 1}, lookup, "small", _available_tags(), 1
    )

    thresholds = dict(scoring.QTY_CLASS_PRICE_THRESHOLDS)
    thresholds["snacking_max"] -= 1
    monkeypatch.setattr(scoring, "QTY_CLASS_PRICE_THRESHOLDS", thresholds)
    after_record = box_features.extract_box_features(
        "x", {4: 1}, lookup, "small", _available_tags(), 1
    )

    assert before_record["item_quantities"][0][2] == scoring.QUANTITY_CLASSES[
        "snack_piece"
    ]["small"]
    assert after_record["item_quantities"][0][2] == scoring.QUANTITY_CLASSES[
        "cooking_piece"
    ]["small"]
    assert box_features.config_hash() != before_hash
    assert box_features.config_snapshot() != before_snapshot


def test_config_snapshot_is_json_serialisable_and_round_trips():
    """The generator writes it to a file and the analyser compares the parsed
    object for equality, so a tuple that survives in-process but becomes a list
    on reload would make the guard fire on every run."""
    import json

    from allocator.box_features import config_snapshot

    snap = config_snapshot()
    assert json.loads(json.dumps(snap, sort_keys=True)) == snap


def test_config_snapshot_detaches_mutable_scoring_inputs(monkeypatch):
    import copy

    import allocator.box_features as box_features
    import allocator.strategies._scoring as scoring

    group_allowances = copy.deepcopy(box_features.GROUP_ALLOWANCES)
    quantity_classes = copy.deepcopy(scoring.QUANTITY_CLASSES)
    thresholds = copy.deepcopy(scoring.QTY_CLASS_PRICE_THRESHOLDS)
    monkeypatch.setattr(box_features, "GROUP_ALLOWANCES", group_allowances)
    monkeypatch.setattr(scoring, "QUANTITY_CLASSES", quantity_classes)
    monkeypatch.setattr(scoring, "QTY_CLASS_PRICE_THRESHOLDS", thresholds)

    before_hash = box_features.config_hash()
    before_snapshot = box_features.config_snapshot()
    returned = box_features.config_snapshot()

    group = next(iter(returned["group_allowances"]))
    group_tier = next(iter(returned["group_allowances"][group]))
    returned["group_allowances"][group][group_tier] += 100

    quantity_class = next(iter(returned["quantity_classes"]))
    quantity_tier = next(iter(returned["quantity_classes"][quantity_class]))
    returned["quantity_classes"][quantity_class][quantity_tier] += 100

    threshold = next(iter(returned["qty_class_price_thresholds"]))
    returned["qty_class_price_thresholds"][threshold] += 100

    assert box_features.config_hash() == before_hash
    assert box_features.config_snapshot() == before_snapshot
    assert returned["group_allowances"] is not box_features.GROUP_ALLOWANCES
    assert returned["quantity_classes"] is not scoring.QUANTITY_CLASSES
    assert returned["qty_class_price_thresholds"] is not scoring.QTY_CLASS_PRICE_THRESHOLDS

    box_features.GROUP_ALLOWANCES[group][group_tier] += 200
    scoring.QUANTITY_CLASSES[quantity_class][quantity_tier] += 200
    scoring.QTY_CLASS_PRICE_THRESHOLDS[threshold] += 200
    assert before_snapshot != box_features.config_snapshot()
    assert before_snapshot["group_allowances"][group][group_tier] + 200 == (
        box_features.GROUP_ALLOWANCES[group][group_tier]
    )


def test_box_target_pct_constant_is_the_only_input_to_box_tiers(monkeypatch):
    """The loader and snapshot share the same import-time-frozen value."""
    import allocator.config as cfg

    monkeypatch.setattr(cfg, "BOX_TARGET_PCT", 123)
    monkeypatch.setenv("BOX_TARGET_PCT", "999")
    monkeypatch.setenv("BOX_PRICE_SMALL", "1000")
    tiers = cfg._load_box_tiers()
    assert tiers["small"]["target_value"] == round(1000 * 123 / 100)


def test_config_snapshot_ignores_environment_changes_after_import(monkeypatch):
    """Never mix a live percentage with import-time-frozen BOX_TIERS."""
    import allocator.config as cfg
    from allocator.box_features import config_snapshot

    before = config_snapshot()
    monkeypatch.setenv("BOX_TARGET_PCT", str(cfg.BOX_TARGET_PCT + 7))
    assert config_snapshot() == before
    assert before["box_target_pct"] == cfg.BOX_TARGET_PCT
