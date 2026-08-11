"""Contract tests for the DB-free EBM diagnostic."""

from __future__ import annotations

import json

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
