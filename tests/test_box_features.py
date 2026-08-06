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


def _collect_module_scope_dependency(tmp_path, *pytest_args):
    test_module = tmp_path / "test_diagnostic_dependency.py"
    test_module.write_text(
        "import pytest\n"
        "from tests.conftest import require_dep\n"
        "pytestmark = pytest.mark.diagnostics(reason='synthetic')\n"
        "require_dep('a_module_that_does_not_exist_xyz')\n"
        "def test_unreachable():\n"
        "    raise AssertionError('missing dependency did not stop collection')\n"
    )
    (tmp_path / "test_sentinel.py").write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.diagnostics(reason='synthetic')\n"
        "def test_pytest_completed_collection():\n"
        "    pass\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
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


def test_a_third_category_is_refused_rather_than_normalised_away():
    """The pair sums to 1 only while every resolved item is fruit or veg —
    which is what lets flatten() drop the veg column. A third category is a
    schema event, not a data point: it needs a third column and a re-derived
    maxT family. Fail where it is introduced, not silently in the matrix."""
    from allocator.box_features import extract_box_features

    lookup = _item_lookup()
    lookup[4] = {**lookup[1], "name": "Mystery Herb", "price": 500,
                 "category_id": 987654, "fungible_group": None,
                 "fungible_degree": 0.0}
    with pytest.raises(ValueError, match="neither CATEGORY_FRUIT"):
        extract_box_features(
            box_name="x", allocations={1: 3, 3: 2, 4: 1}, item_lookup=lookup,
            tier="small", available_tags=_available_tags(), offer_id=1,
        )


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


@pytest.mark.parametrize("name", [
    "BOX_TIERS", "GROUP_ALLOWANCES", "ITEM_CLASSIFICATIONS", "FUNGIBLE_GROUPS",
    "CLASSIFICATION_FALLBACK", "QUANTITY_CLASSES", "QTY_CLASS_PRICE_THRESHOLDS",
    "VALUE_SWEET_FROM", "VALUE_SWEET_TO", "VALUE_PENALTY_EXPONENT",
])
def test_config_hash_changes_when_any_pinned_input_changes(monkeypatch, name):
    """All ten allocator.config inputs are load-bearing. QUANTITY_CLASSES and
    QTY_CLASS_PRICE_THRESHOLDS were the original omission — they determine the
    item_allowance persisted in every feature record."""
    import allocator.config as cfg
    from allocator.box_features import config_hash

    before = config_hash()
    current = getattr(cfg, name)
    if isinstance(current, (int, float)):
        mutated = current + 1
    else:
        # The probe key MUST match the existing key type. CLASSIFICATION_FALLBACK
        # is keyed by integer category IDs (allocator/config.py:207-209); adding a
        # str key makes json.dumps(..., sort_keys=True) raise
        # "TypeError: '<' not supported between instances of 'str' and 'int'"
        # inside _digest, so the test would error rather than assert.
        sample = next(iter(current))
        probe = max(current) + 1 if isinstance(sample, int) else "zzz_probe"
        assert probe not in current
        mutated = {**current, probe: 1}
    monkeypatch.setattr(cfg, name, mutated)
    assert config_hash() != before


def test_config_hash_changes_with_canonical_default_classification(monkeypatch):
    import allocator.categorizer as categorizer
    from allocator.box_features import config_hash

    before = config_hash()
    monkeypatch.setattr(
        categorizer,
        "DEFAULT_CLASSIFICATION",
        ("reserved_other", "cooking", "green", "round"),
    )
    assert config_hash() != before


def test_config_hash_includes_target_percentage_through_box_tiers(monkeypatch):
    import allocator.config as cfg
    from allocator.box_features import config_hash

    before = config_hash()
    tiers = {tier: dict(values) for tier, values in cfg.BOX_TIERS.items()}
    tiers["small"]["target_value"] += 1
    monkeypatch.setattr(cfg, "BOX_TIERS", tiers)
    assert config_hash() != before


def test_config_snapshot_has_exactly_the_pinned_keys():
    from allocator.box_features import config_snapshot

    snap = config_snapshot()
    assert set(snap) == {
        "box_tiers", "box_target_pct", "value_sweet_from", "value_sweet_to",
        "value_penalty_exponent", "group_allowances", "quantity_classes",
        "qty_class_price_thresholds", "item_classifications_hash",
        "fungible_groups_hash", "classification_fallback_hash",
        "default_classification_hash",
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


def test_config_snapshot_changes_with_classification_fallback(monkeypatch):
    import allocator.config as cfg
    from allocator.box_features import config_snapshot

    before = config_snapshot()
    fallback = dict(cfg.CLASSIFICATION_FALLBACK)
    category_id = next(iter(fallback))
    fallback[category_id] = ("reserved_fallback", *fallback[category_id][1:])
    monkeypatch.setattr(cfg, "CLASSIFICATION_FALLBACK", fallback)

    after = config_snapshot()
    assert after != before
    assert (
        after["classification_fallback_hash"]
        != before["classification_fallback_hash"]
    )


def test_config_snapshot_changes_with_default_classification(monkeypatch):
    import allocator.categorizer as categorizer
    from allocator.box_features import config_snapshot

    before = config_snapshot()
    monkeypatch.setattr(
        categorizer,
        "DEFAULT_CLASSIFICATION",
        ("reserved_default", *categorizer.DEFAULT_CLASSIFICATION[1:]),
    )

    after = config_snapshot()
    assert after != before
    assert (
        after["default_classification_hash"]
        != before["default_classification_hash"]
    )


def test_config_snapshot_is_json_serialisable_and_round_trips():
    """The generator writes it to a file and the analyser compares the parsed
    object for equality, so a tuple that survives in-process but becomes a list
    on reload would make the guard fire on every run."""
    import json

    from allocator.box_features import config_snapshot

    snap = config_snapshot()
    assert json.loads(json.dumps(snap, sort_keys=True)) == snap


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
