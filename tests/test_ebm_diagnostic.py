"""Contract tests for the DB-free EBM diagnostic."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import pytest

from allocator.box_features import (
    FEATURE_SCHEMA_VERSION,
    config_hash,
    extract_box_features,
    flatten,
)
from allocator.config import CATEGORY_FRUIT
from allocator.hard_negative_roster import roster_config_hash
from tests.conftest import require_dep


pytestmark = pytest.mark.diagnostics
require_dep("numpy")


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
                "fungible_group": "apple",
                "fungible_degree": 1.0,
                "sub_category": "pome_fruit",
                "usage": "snacking",
                "colour": "red",
                "shape": "round",
                "category_id": CATEGORY_FRUIT,
            }
        },
        tier,
        {
            "sub_category": {"pome_fruit"},
            "usage": {"snacking"},
            "colour": {"red"},
            "shape": {"round"},
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
    X = X[:, :2]
    columns = ["agnostic", "scored"]
    basis = {"agnostic": "agnostic", "scored": "scored"}

    def fake_fit(_X, _labels, _columns, seed):
        del _X, _labels, _columns, seed
        return ebm_diagnostic.FittedRung(
            model=None,
            importances={"agnostic": 1.0, "scored": 100.0},
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
    assert result == ebm_diagnostic.run_maxt(
        X, labels, clusters, columns, basis, seed=3, n_permutations=200
    )
