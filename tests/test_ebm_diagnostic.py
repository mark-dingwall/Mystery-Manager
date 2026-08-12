"""Contract tests for the DB-free EBM diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.conftest import require_dep


pytestmark = pytest.mark.diagnostics
np = require_dep("numpy")
require_dep("interpret")
require_dep("sklearn")

from allocator.box_features import (
    FEATURE_SCHEMA_VERSION,
    config_hash,
    extract_box_features,
    flatten,
)
from allocator.config import CATEGORY_FRUIT
from allocator.hard_negative_roster import roster_config_hash


def _record(
    source: str,
    *,
    offer_id: int = 1,
    box_name: str = "packer@example.test",
    tier: str = "small",
    quantity: int = 1,
) -> dict:
    """Build one complete, synthetic schema-v2 feature record."""
    record = extract_box_features(
        box_name,
        {1: quantity},
        {
            1: {
                "price": 100,
                "size": 1,
                "fungible_group": "diagnostic_group",
                "fungible_degree": 1.0,
                "sub_category": "diagnostic_orbit",
                "usage": "diagnostic_use",
                "colour": "cyan",
                "shape": "star",
                "category_id": CATEGORY_FRUIT,
            }
        },
        tier,
        {
            "sub_category": {"diagnostic_orbit"},
            "usage": {"diagnostic_use"},
            "colour": {"cyan"},
            "shape": {"star"},
        },
        offer_id,
        source=source,
    )
    assert record is not None
    return record


def _artifact(records: list[dict]) -> dict:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": config_hash(),
        "roster_config_hash": roster_config_hash(),
        "records": records,
        "source_counts": {"manual": 1, "ilp_optimal": 1},
        "roster_check": {},
        "attrition": {},
        "exclusions": [],
        "run_metadata": {"generator_version": 1},
    }


def test_load_artifact_accepts_current_provenance_and_rejects_failure_report(tmp_path):
    """A stale or failed generator output must never become EBM training data."""
    from scripts.ebm_diagnostic import load_artifact

    valid_path = tmp_path / "hard_negatives.json"
    payload = _artifact([_record("manual"), _record("ilp_optimal", quantity=2)])
    valid_path.write_text(json.dumps(payload))
    assert load_artifact(valid_path) == payload

    failed_path = tmp_path / "hard_negatives_report.json"
    failed_path.write_text(json.dumps({"status": "validation_failed"}))
    with pytest.raises(ValueError, match="failure report"):
        load_artifact(failed_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("feature_schema_version", FEATURE_SCHEMA_VERSION + 1, "schema"),
        ("config_hash", "0" * 16, "config_hash"),
        ("roster_config_hash", "0" * 16, "roster_config_hash"),
    ],
)
def test_load_artifact_rejects_each_stale_provenance_stamp(
    tmp_path, field, value, match
):
    """Each stale provenance input can change the matrix or roster semantics."""
    from scripts.ebm_diagnostic import load_artifact

    payload = _artifact([_record("manual"), _record("ilp_optimal", quantity=2)])
    payload[field] = value
    path = tmp_path / "hard_negatives.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        load_artifact(path)


@pytest.mark.parametrize(
    "field",
    ["feature_schema_version", "config_hash", "roster_config_hash"],
)
def test_load_artifact_rejects_each_missing_provenance_stamp(tmp_path, field):
    """Omitting a stamp must not bypass stale-artifact protection."""
    from scripts.ebm_diagnostic import load_artifact

    payload = _artifact([_record("manual"), _record("ilp_optimal", quantity=2)])
    del payload[field]
    path = tmp_path / "hard_negatives.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="missing required fields"):
        load_artifact(path)


def test_load_artifact_rejects_rows_that_cannot_build_the_feature_matrix(tmp_path):
    """A malformed record cannot become an underpowered non-result by accident."""
    from scripts.ebm_diagnostic import load_artifact

    payload = _artifact([_record("manual"), _record("ilp_optimal", quantity=2)])
    del payload["records"][0]["value_pct"]
    path = tmp_path / "hard_negatives.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="record 0"):
        load_artifact(path)


def test_basis_for_columns_covers_the_live_flatten_contract_and_rejects_unknowns():
    """A new feature cannot silently enter the maxT family without a basis."""
    from scripts.ebm_diagnostic import basis_for_columns

    columns = list(flatten(_record("manual")))
    basis = basis_for_columns(columns)

    assert set(basis) == set(columns)
    assert basis["value_pct_small"] == "scored"
    assert basis["capped_group_totals.apple"] == "scored"
    assert basis["raw_group_totals.apple"] == "parent"
    assert basis["n_unique_items"] == "agnostic"
    with pytest.raises(ValueError, match="unclassified"):
        basis_for_columns([*columns, "unclassified.feature"])


def test_prepare_rung_keeps_only_complete_casefolded_box_clusters():
    """Incomplete boxes must not influence either AUC or the permutation null."""
    from scripts.ebm_diagnostic import prepare_rung

    records = [
        _record("manual", box_name="Packer@Example.Test"),
        _record("ilp_optimal", box_name="packer@example.test", quantity=2),
        _record("manual", box_name="manual-only@example.test"),
        _record("baseline_deal_topup", box_name="baseline-only@example.test"),
    ]

    prepared = prepare_rung(records, "manual_vs_ilp")

    assert [record["source"] for record in prepared.records] == [
        "manual",
        "ilp_optimal",
    ]
    assert prepared.clusters == [(1, "small", "packer@example.test")] * 2
    assert prepared.attrition == {
        "input_clusters": 3,
        "missing_manual_clusters": 1,
        "missing_negative_clusters": 1,
        "retained_clusters": 1,
    }


def test_prepare_rung_rejects_boolean_offer_ids():
    """Booleans compare equal to integer offer IDs but are never valid identities."""
    from scripts.ebm_diagnostic import prepare_rung

    records = [_record("manual"), _record("ilp_optimal", quantity=2)]
    records[0]["offer_id"] = True

    with pytest.raises(ValueError, match="invalid offer_id"):
        prepare_rung(records, "manual_vs_ilp")


@pytest.mark.parametrize(
    ("rung", "expected_sources"),
    [
        ("manual_vs_synth", ["manual", "synth_random"]),
        ("manual_vs_baseline", ["manual", "baseline_deal_topup"]),
        ("manual_vs_ilp", ["manual", "ilp_optimal"]),
    ],
)
def test_prepare_rung_selects_only_its_declared_negative_sources(
    rung, expected_sources
):
    """Cross-rung negatives would leak different failure modes into one model."""
    from scripts.ebm_diagnostic import prepare_rung

    records = [
        _record("manual"),
        _record("synth_random", quantity=2),
        _record("baseline_deal_topup", quantity=3),
        _record("ilp_optimal", quantity=4),
    ]

    prepared = prepare_rung(records, rung)

    assert [record["source"] for record in prepared.records] == expected_sources


def test_build_design_matrix_keeps_only_features_and_prepared_cluster_identity():
    """Source labels and variable-length scoring internals must not leak into X."""
    from scripts.ebm_diagnostic import build_design_matrix, prepare_rung

    prepared = prepare_rung(
        [
            _record("manual", offer_id=9, quantity=1),
            _record("ilp_optimal", offer_id=9, quantity=2),
        ],
        "manual_vs_ilp",
    )
    X, columns, labels, offers, clusters = build_design_matrix(prepared.records)

    assert X.shape == (2, len(columns))
    assert columns == sorted(columns)
    assert labels.tolist() == [1, 0]
    assert offers.tolist() == [9, 9]
    assert clusters == [(9, "small", "packer@example.test")] * 2
    assert "source" not in columns
    assert "item_quantities" not in columns
    assert "group_totals" not in columns


def test_ebm_module_import_never_reaches_db_or_allocation_modules(tmp_path):
    """Adding an eager DB import would break diagnostics portability and tests."""
    project_root = Path(__file__).resolve().parent.parent
    target_lib = project_root / ".venv-diagnostics" / "lib"
    code = """
import importlib.abc
import sys

class BlockDiagnosticsLeaks(importlib.abc.MetaPathFinder):
    blocked = {"allocator.db", "allocator.allocator", "compare"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.blocked:
            raise AssertionError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, BlockDiagnosticsLeaks())
import scripts.ebm_diagnostic
"""
    env = os.environ | {
        "PYTHONPATH": os.pathsep.join((str(project_root), str(target_lib))),
        "BOX_PRICE_SMALL": "2000",
        "BOX_PRICE_MEDIUM": "3500",
        "BOX_PRICE_LARGE": "5000",
    }
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ebm_script_runs_the_documented_isolated_command():
    """The documented command supplies only diagnostic dependencies, not the root."""
    project_root = Path(__file__).resolve().parent.parent
    target_lib = project_root / ".venv-diagnostics" / "lib"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(target_lib),
            "BOX_PRICE_SMALL": "2000",
            "BOX_PRICE_MEDIUM": "3500",
            "BOX_PRICE_LARGE": "5000",
        }
    )

    proc = subprocess.run(
        [sys.executable, "-S", "scripts/ebm_diagnostic.py", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EBM hard-negative diagnostic" in proc.stdout


def test_diagnostic_versions_reports_a_missing_distribution_actionably(monkeypatch):
    """A missing diagnostic package must not leak metadata's raw exception."""
    from importlib.metadata import PackageNotFoundError
    from scripts import ebm_diagnostic

    def missing_version(distribution):
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(ebm_diagnostic, "version", missing_version)

    with pytest.raises(RuntimeError, match="requirements-diagnostics.txt"):
        ebm_diagnostic._diagnostic_versions()


def _fit_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str, str]]]:
    """Return five offer groups with separable but non-constant class signal."""
    rows = []
    labels = []
    offers = []
    clusters = []
    for offer_id in range(1, 6):
        for index, label in enumerate((1, 0, 1, 0)):
            rows.append(
                [float(label), float((offer_id + index) % 2), float(offer_id)]
            )
            labels.append(label)
            offers.append(offer_id)
            clusters.append((offer_id, "small", f"box-{index // 2}.example.test"))
    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(offers, dtype=int),
        clusters,
    )


def test_balanced_sample_weights_equalise_total_class_mass():
    """Unbalanced source counts must not let the larger class dominate an EBM."""
    from scripts.ebm_diagnostic import balanced_sample_weights

    weights = balanced_sample_weights(np.asarray([1, 1, 1, 0], dtype=int))

    assert weights.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3] == pytest.approx(0.5)


def test_fit_full_ebm_uses_named_main_effects_and_returns_importances():
    """A model with unnamed or interaction terms cannot support the finding contract."""
    from scripts.ebm_diagnostic import fit_full_ebm

    X, labels, _offers, _clusters = _fit_data()
    fitted = fit_full_ebm(X, labels, ["signal", "noise", "offer_marker"], seed=7)

    assert fitted.model.interactions == 0
    assert fitted.model.term_names_ == ["signal", "noise", "offer_marker"]
    assert set(fitted.importances) == {"signal", "noise", "offer_marker"}
    assert fitted.auc_insample > 0.9


def test_fit_without_value_pct_drops_all_value_columns_and_aligns_matrix(monkeypatch):
    """The descriptive ablation must refit exactly the non-value design matrix."""
    from scripts import ebm_diagnostic

    X = np.asarray(
        [[1.0, 10.0, 2.0, 30.0], [3.0, 20.0, 4.0, 40.0]], dtype=float
    )
    labels = np.asarray([1, 0], dtype=int)
    columns = ["value_pct_small", "kept_one", "value_pct_large", "kept_two"]
    captured = {}

    def fake_fit(matrix, target, retained_columns, seed):
        captured["matrix"] = matrix
        captured["target"] = target
        captured["columns"] = retained_columns
        captured["seed"] = seed
        return ebm_diagnostic.FittedRung(model=None, importances={}, auc_insample=0.5)

    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fake_fit)

    ebm_diagnostic.fit_without_value_pct(X, labels, columns, seed=23)

    assert np.array_equal(captured["matrix"], X[:, [1, 3]])
    assert np.array_equal(captured["target"], labels)
    assert captured["columns"] == ["kept_one", "kept_two"]
    assert captured["seed"] == 23


def test_auc_group_kfold_holds_out_entire_offers(monkeypatch):
    """Rows from an offer must not appear in both train and validation folds."""
    from scripts import ebm_diagnostic

    X, labels, offers, _clusters = _fit_data()
    fit_offer_sets = []

    class PerfectModel:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, validation_X):
            positives = validation_X[:, 0]
            return np.column_stack((1.0 - positives, positives))

    def fake_fit(train_X, _train_labels, _columns, seed):
        del _train_labels, _columns, seed
        fit_offer_sets.append(set(train_X[:, 2].tolist()))
        return ebm_diagnostic.FittedRung(
            model=PerfectModel(), importances={}, auc_insample=1.0
        )

    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fake_fit)

    result = ebm_diagnostic.auc_group_kfold(
        X, labels, offers, ["signal", "noise", "offer_marker"], seed=7
    )

    assert result == pytest.approx(1.0)
    assert len(fit_offer_sets) == 5
    assert {frozenset(offer_set) for offer_set in fit_offer_sets} == {
        frozenset({1, 2, 3, 4, 5} - {offer_id}) for offer_id in range(1, 6)
    }


def test_permutation_preserves_each_matched_cluster_label_count():
    """Cluster-local shuffling protects the shared manual/negative roster structure."""
    from scripts.ebm_diagnostic import permute_labels_within_clusters

    labels = np.asarray([1, 0, 0, 1, 0], dtype=int)
    clusters = [
        (1, "small", "a"),
        (1, "small", "a"),
        (1, "small", "a"),
        (2, "medium", "b"),
        (2, "medium", "b"),
    ]

    permuted = permute_labels_within_clusters(
        labels, clusters, np.random.default_rng(12)
    )

    assert sorted(permuted[:3].tolist()) == [0, 0, 1]
    assert sorted(permuted[3:].tolist()) == [0, 1]


def test_maxt_rejects_too_few_permutations_and_ignores_scored_terms(monkeypatch):
    """Reducing the null below 200 or including scored terms invalidates promotion."""
    from scripts import ebm_diagnostic

    X, labels, _offers, clusters = _fit_data()
    columns = ["agnostic", "raw_tag_counts.deferred_tag", "scored"]
    basis = {
        "agnostic": "agnostic",
        "raw_tag_counts.deferred_tag": "parent",
        "scored": "scored",
    }
    fit_calls = []

    def fake_fit(_X, fitted_labels, _columns, seed):
        del _X, _columns
        fit_calls.append((np.asarray(fitted_labels).copy(), seed))
        return ebm_diagnostic.FittedRung(
            model=None,
            importances={
                "agnostic": 1.0,
                "raw_tag_counts.deferred_tag": 50.0,
                "scored": 100.0,
            },
            auc_insample=1.0,
        )

    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fake_fit)

    with pytest.raises(ValueError, match="at least 200"):
        ebm_diagnostic.run_maxt(
            X, labels, clusters, columns, basis, seed=3, n_permutations=199
        )

    result = ebm_diagnostic.run_maxt(
        X, labels, clusters, columns, basis, seed=3, n_permutations=200
    )

    assert len(result.null_maxima) == 200
    assert result.family_columns == ["agnostic"]
    assert result.observed_importances == {"agnostic": pytest.approx(1.0)}
    assert result.threshold == pytest.approx(1.0)
    assert len(fit_calls) == 201
    assert np.array_equal(fit_calls[0][0], labels)
    assert {seed for _fitted_labels, seed in fit_calls} == {3}
    assert any(
        not np.array_equal(fitted_labels, labels)
        for fitted_labels, _seed in fit_calls[1:]
    )
    for fitted_labels, _seed in fit_calls[1:]:
        for cluster in set(clusters):
            indices = [index for index, value in enumerate(clusters) if value == cluster]
            assert sorted(fitted_labels[indices].tolist()) == sorted(labels[indices].tolist())


def test_maxt_reuses_a_supplied_full_data_fit(monkeypatch):
    """The report fit is also the observed fit for maxT, avoiding a duplicate EBM."""
    from scripts import ebm_diagnostic

    X, labels, _offers, clusters = _fit_data()
    columns = ["agnostic", "raw_tag_counts.deferred_tag", "scored"]
    basis = {
        "agnostic": "agnostic",
        "raw_tag_counts.deferred_tag": "parent",
        "scored": "scored",
    }
    observed_fit = ebm_diagnostic.FittedRung(
        model=None,
        importances={
            "agnostic": 1.0,
            "raw_tag_counts.deferred_tag": 50.0,
            "scored": 100.0,
        },
        auc_insample=1.0,
    )
    fit_calls = []

    def fake_fit(_X, _labels, _columns, seed):
        del _X, _labels, _columns, seed
        fit_calls.append(None)
        return observed_fit

    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fake_fit)

    result = ebm_diagnostic.run_maxt(
        X,
        labels,
        clusters,
        columns,
        basis,
        seed=3,
        n_permutations=200,
        observed_fit=observed_fit,
    )

    assert len(fit_calls) == 200
    assert result.observed_importances == {"agnostic": pytest.approx(1.0)}


def test_maxt_rejects_an_incomplete_matched_cluster(monkeypatch):
    """A cluster without both classes cannot support a paired label permutation."""
    from scripts import ebm_diagnostic

    def fail_if_fit(*_args, **_kwargs):
        raise AssertionError("incomplete clusters must fail before fitting")

    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fail_if_fit)

    with pytest.raises(ValueError, match="complete manual/negative cluster"):
        ebm_diagnostic.run_maxt(
            np.asarray([[1.0], [0.0], [1.0], [0.0]]),
            np.asarray([1, 0, 1, 0]),
            [(1, "small", "a"), (1, "small", "a"), (2, "small", "b"), (3, "small", "c")],
            ["agnostic"],
            {"agnostic": "agnostic"},
            seed=3,
            n_permutations=200,
        )


def test_maxt_findings_require_strict_exceedance_of_the_null_threshold():
    """A tie with the maxT threshold is not enough evidence to promote a term."""
    from scripts.ebm_diagnostic import _clears_maxt

    assert _clears_maxt(1.01, 1.0) is True
    assert _clears_maxt(1.0, 1.0) is False


def test_findings_report_tierwise_value_and_group_parent_metadata():
    """Promoted group parents retain the descriptive checks needed for review."""
    from scripts.ebm_diagnostic import FittedRung, MaxTResult, _build_findings

    columns = [
        "raw_group_totals.apple",
        "capped_group_totals.apple",
        "value_pct_small",
        "value_pct_medium",
        "value_pct_large",
    ]
    records = [
        {"tier": tier}
        for tier in ("small", "small", "medium", "medium", "large", "large")
    ]
    X = np.asarray(
        [
            [1.0, 10.0, 0.1, 0.0, 0.0],
            [2.0, 20.0, 0.2, 0.0, 0.0],
            [2.0, 20.0, 0.0, 0.2, 0.0],
            [4.0, 40.0, 0.0, 0.5, 0.0],
            [4.0, 40.0, 0.0, 0.0, 0.4],
            [7.0, 70.0, 0.0, 0.0, 0.7],
        ]
    )
    findings = _build_findings(
        FittedRung(model=None, importances={"raw_group_totals.apple": 0.6}, auc_insample=1.0),
        MaxTResult(
            observed_importances={"raw_group_totals.apple": 0.6},
            null_maxima=[0.5],
            threshold=0.5,
            family_columns=["raw_group_totals.apple"],
        ),
        X,
        records,
        columns,
        {column: "parent" for column in columns},
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding["value_confound_r"] == {
        "small": pytest.approx(1.0),
        "medium": pytest.approx(1.0),
        "large": pytest.approx(1.0),
    }
    assert finding["aggregate_term"] == "capped_group_totals.apple"
    assert finding["aggregate_r"] == {
        "small": pytest.approx(1.0),
        "medium": pytest.approx(1.0),
        "large": pytest.approx(1.0),
    }
    assert finding["aggregate_explained"] is True
    assert finding["aggregate_check"] == "per-tier Pearson correlation"


def test_findings_mark_degenerate_value_slices_and_tag_parent_checks():
    """Degenerate statistics and deferred tag checks stay explicit in output."""
    from scripts.ebm_diagnostic import FittedRung, MaxTResult, _build_findings

    term = "raw_tag_counts.usage.snacking"
    findings = _build_findings(
        FittedRung(model=None, importances={term: 0.6}, auc_insample=1.0),
        MaxTResult(
            observed_importances={term: 0.6},
            null_maxima=[0.5],
            threshold=0.5,
            family_columns=[term],
        ),
        np.asarray([[1.0, 0.1], [1.0, 0.2]]),
        [{"tier": "small"}, {"tier": "small"}],
        [term, "value_pct_small"],
        {term: "parent", "value_pct_small": "scored"},
    )

    assert findings[0]["value_confound_r"] == {
        "small": None,
        "medium": None,
        "large": None,
    }
    assert findings[0]["aggregate_term"] is None
    assert findings[0]["aggregate_r"] is None
    assert findings[0]["aggregate_explained"] is None
    assert (
        findings[0]["aggregate_check"]
        == "deferred tag-parent aggregate-preserving refit"
    )


def _underpowered_artifact() -> dict:
    """Return five matched offers, intentionally below the diagnostic floor."""
    records = []
    for offer_id in range(1, 6):
        records.extend(
            [
                _record("manual", offer_id=offer_id, quantity=1),
                _record("ilp_optimal", offer_id=offer_id, quantity=2),
            ]
        )
    return _artifact(records)


def test_cli_defaults_reject_invalid_rungs_and_keeps_plot_flag_compatible():
    """The documented operator interface stays small and validates rungs early."""
    from scripts.ebm_diagnostic import build_parser

    parser = build_parser()
    args = parser.parse_args([])

    assert args.features == Path("diagnostics/hard_negatives.json")
    assert args.out == Path("diagnostics/ebm_findings.json")
    assert args.seed == 0
    assert args.permutations == 200
    assert args.rungs == ["manual_vs_synth", "manual_vs_baseline", "manual_vs_ilp"]
    assert args.no_plots is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--rungs", "not-a-rung"])


def test_main_wires_parsed_arguments_to_run(monkeypatch, tmp_path):
    """The CLI must pass every documented option through to the runner."""
    from scripts import ebm_diagnostic

    captured = {}

    def fake_run(features, out, *, seed, permutations, rungs):
        captured.update(
            {
                "features": features,
                "out": out,
                "seed": seed,
                "permutations": permutations,
                "rungs": rungs,
            }
        )
        return {}

    monkeypatch.setattr(ebm_diagnostic, "run", fake_run)

    assert (
        ebm_diagnostic.main(
            [
                "--features",
                str(tmp_path / "input.json"),
                "--out",
                str(tmp_path / "output.json"),
                "--seed",
                "19",
                "--permutations",
                "200",
                "--rungs",
                "manual_vs_ilp",
            ]
        )
        == 0
    )
    assert captured == {
        "features": tmp_path / "input.json",
        "out": tmp_path / "output.json",
        "seed": 19,
        "permutations": 200,
        "rungs": ["manual_vs_ilp"],
    }


def test_run_serializes_provenance_and_empty_underpowered_rung_deterministically(
    tmp_path, monkeypatch
):
    """A rung below its registered floor must not fit or promote any term."""
    from scripts import ebm_diagnostic

    features = tmp_path / "hard_negatives.json"
    features.write_text(json.dumps(_underpowered_artifact()))
    first_out = tmp_path / "first.json"
    second_out = tmp_path / "second.json"

    def fail_if_fit(*_args, **_kwargs):
        raise AssertionError("an underpowered rung must not fit an EBM")

    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fail_if_fit)

    first = ebm_diagnostic.run(
        features,
        first_out,
        seed=11,
        permutations=200,
        rungs=["manual_vs_ilp"],
    )
    second = ebm_diagnostic.run(
        features,
        second_out,
        seed=11,
        permutations=200,
        rungs=["manual_vs_ilp"],
    )

    rung = first["rungs"]["manual_vs_ilp"]
    assert first["schema_version"] == 1
    assert first["artifact"] == {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": config_hash(),
        "roster_config_hash": roster_config_hash(),
        "input_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
        "run_metadata": {"generator_version": 1},
        "source_counts": {"manual": 1, "ilp_optimal": 1},
        "attrition": {},
    }
    assert first["methodology"]["seed"] == 11
    assert first["methodology"]["permutations"] == 200
    assert first["methodology"]["interactions_enabled"] is False
    assert set(first["methodology"]["versions"]) == {
        "interpret",
        "scikit-learn",
        "numpy",
    }
    assert "value confounding" in first["methodology"]["caveats"]
    assert "tag-parent" in first["methodology"]["caveats"]
    for deferred_capability in (
        "plots",
        "interaction models",
        "multi-seed stability",
        "parallel permutations",
        "tag-parent refit permutations",
        "leave-one-negative-source-out ablations",
    ):
        assert deferred_capability in first["methodology"]["caveats"]
    assert rung == {
        "underpowered": True,
        "n_pos": 5,
        "n_neg": 5,
        "n_offers": 5,
        "model_seed": 13,
        "attrition": {
            "input_clusters": 5,
            "missing_manual_clusters": 0,
            "missing_negative_clusters": 0,
            "retained_clusters": 5,
        },
        "column_count": None,
        "auc_oof": None,
        "auc_insample": None,
        "maxt_threshold": None,
        "maxT_family_count": None,
        "maxT_family_columns": None,
        "maxt_null_quantiles": None,
        "findings": [],
    }
    assert first == second
    assert first_out.read_text() == second_out.read_text()


def test_run_fingerprints_the_validated_artifact_byte_snapshot(monkeypatch, tmp_path):
    """The provenance hash is for the exact bytes consumed as diagnostic input."""
    from scripts import ebm_diagnostic

    features = tmp_path / "hard_negatives.json"
    artifact_bytes = json.dumps(_underpowered_artifact()).encode("utf-8")
    features.write_bytes(artifact_bytes)
    real_load_artifact = ebm_diagnostic.load_artifact

    def validate_then_replace(path, *, raw_bytes):
        artifact = real_load_artifact(path, raw_bytes=raw_bytes)
        path.write_text('{"status": "changed_after_validation"}')
        return artifact

    monkeypatch.setattr(ebm_diagnostic, "load_artifact", validate_then_replace)

    result = ebm_diagnostic.run(
        features,
        tmp_path / "findings.json",
        seed=11,
        permutations=200,
        rungs=["manual_vs_ilp"],
    )

    assert result["artifact"]["input_sha256"] == hashlib.sha256(artifact_bytes).hexdigest()


def test_run_reuses_one_prepared_cohort_for_all_inference(monkeypatch, tmp_path):
    """AUC, ablation, and maxT must not silently receive different populations."""
    from scripts import ebm_diagnostic

    features = tmp_path / "hard_negatives.json"
    features.write_text(json.dumps(_underpowered_artifact()))
    captured = {}

    def fake_full_fit(X, labels, columns, seed):
        captured["full"] = (np.asarray(X).copy(), np.asarray(labels).copy(), list(columns), seed)
        return ebm_diagnostic.FittedRung(
            model=None,
            importances={column: 1.0 for column in columns},
            auc_insample=0.75,
        )

    def fake_ablation(X, labels, columns, seed):
        captured["ablation"] = (
            np.asarray(X).copy(),
            np.asarray(labels).copy(),
            list(columns),
            seed,
        )
        return ebm_diagnostic.FittedRung(
            model=None,
            importances={column: 1.0 for column in columns},
            auc_insample=0.7,
        )

    def fake_oof(X, labels, offers, columns, seed):
        captured["oof"] = (
            np.asarray(X).copy(),
            np.asarray(labels).copy(),
            np.asarray(offers).copy(),
            list(columns),
            seed,
        )
        return 0.6

    def fake_maxt(
        X, labels, clusters, columns, basis, seed, n_permutations, observed_fit
    ):
        captured["maxt"] = (
            np.asarray(X).copy(),
            np.asarray(labels).copy(),
            list(clusters),
            list(columns),
            dict(basis),
            seed,
            n_permutations,
            observed_fit,
        )
        return ebm_diagnostic.MaxTResult(
            observed_importances={"n_unique_items": 2.0},
            null_maxima=[0.0] * 190 + [1.0] * 10,
            threshold=1.0,
            family_columns=["n_unique_items"],
        )

    monkeypatch.setattr(ebm_diagnostic, "_is_underpowered", lambda *_args: False)
    monkeypatch.setattr(ebm_diagnostic, "fit_full_ebm", fake_full_fit)
    monkeypatch.setattr(ebm_diagnostic, "fit_without_value_pct", fake_ablation)
    monkeypatch.setattr(ebm_diagnostic, "auc_group_kfold", fake_oof)
    monkeypatch.setattr(ebm_diagnostic, "run_maxt", fake_maxt)

    result = ebm_diagnostic.run(
        features,
        tmp_path / "findings.json",
        seed=4,
        permutations=200,
        rungs=["manual_vs_ilp"],
    )

    full_X, full_labels, full_columns, full_seed = captured["full"]
    for name in ("ablation", "oof", "maxt"):
        assert np.array_equal(captured[name][0], full_X)
        assert np.array_equal(captured[name][1], full_labels)
    assert captured["ablation"][2] == full_columns
    assert captured["oof"][3] == full_columns
    assert captured["maxt"][3] == full_columns
    assert full_seed == 6
    assert captured["ablation"][3] == 6
    assert captured["oof"][4] == 6
    assert captured["maxt"][5] == 6
    assert captured["maxt"][6] == 200
    assert captured["maxt"][7] is not None
    assert result["rungs"]["manual_vs_ilp"]["underpowered"] is False
    assert result["rungs"]["manual_vs_ilp"]["auc_oof"] == pytest.approx(0.6)
    assert result["rungs"]["manual_vs_ilp"]["maxT_family_count"] == 1
    assert result["rungs"]["manual_vs_ilp"]["maxT_family_columns"] == [
        "n_unique_items"
    ]
    assert result["rungs"]["manual_vs_ilp"]["maxt_null_quantiles"]["p95"] == 1.0
    assert result["rungs"]["manual_vs_ilp"]["findings"][0]["term"] == "n_unique_items"
    assert result["rungs"]["manual_vs_ilp"]["findings"][0][
        "ablation_drop_value_pct"
    ] == {
        "importance": 1.0,
        "interpretation": "descriptive full-fit importance only",
    }


def test_rung_seed_is_stable_when_requested_rungs_are_reordered():
    """The report must not turn a harmless CLI order change into a new model."""
    from scripts.ebm_diagnostic import _rung_seed

    assert _rung_seed(10, "manual_vs_synth") == 10
    assert _rung_seed(10, "manual_vs_baseline") == 11
    assert _rung_seed(10, "manual_vs_ilp") == 12


def test_findings_json_is_published_by_atomic_replace(monkeypatch, tmp_path):
    """A stale result remains intact until the complete replacement is ready."""
    from scripts import ebm_diagnostic

    output = tmp_path / "nested" / "findings.json"
    output.parent.mkdir()
    output.write_text('{"previous": true}\n')
    replace_calls = []
    real_replace = ebm_diagnostic.os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(ebm_diagnostic.os, "replace", recording_replace)
    ebm_diagnostic._write_json_atomically(output, {"complete": True})

    assert json.loads(output.read_text()) == {"complete": True}
    assert replace_calls[0][1] == output
    assert replace_calls[0][0].parent == output.parent
    assert not list(output.parent.glob(".findings.json.*.tmp"))


def test_findings_json_removes_temporary_file_when_replace_fails(monkeypatch, tmp_path):
    """A failed publish preserves the prior report and leaves no temporary file."""
    from scripts import ebm_diagnostic

    output = tmp_path / "findings.json"
    output.write_text('{"previous": true}\n')

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(ebm_diagnostic.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        ebm_diagnostic._write_json_atomically(output, {"complete": True})

    assert json.loads(output.read_text()) == {"previous": True}
    assert not list(output.parent.glob(".findings.json.*.tmp"))
