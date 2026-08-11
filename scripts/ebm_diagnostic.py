#!/usr/bin/env python3
"""Prepare DB-free inputs for the EBM hard-negative diagnostic.

The diagnostic intentionally consumes only the validated artifact written by
``generate_hard_negatives.py``. It never imports allocation, comparison, or DB
code: those paths belong exclusively to artifact generation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from allocator.box_features import FEATURE_SCHEMA_VERSION, config_hash, flatten
from allocator.hard_negative_roster import roster_config_hash


RUNGS = ("manual_vs_synth", "manual_vs_baseline", "manual_vs_ilp")
_REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "feature_schema_version",
        "config_hash",
        "roster_config_hash",
        "records",
        "source_counts",
        "roster_check",
        "attrition",
        "exclusions",
        "run_metadata",
    }
)
_AGNOSTIC_COLUMNS = frozenset(
    {
        "n_unique_items",
        "total_qty",
        "price_mean",
        "price_sd",
        "price_max",
        "fruit_value_share",
    }
)


@dataclass
class PreparedRung:
    """The one complete matched-cluster cohort used by a diagnostic rung."""

    records: list[dict]
    clusters: list[tuple[int, str, str]]
    attrition: dict[str, int]


def load_artifact(path: Path) -> dict:
    """Load a successful, current hard-negative artifact or fail early.

    A failure report and an artifact generated under stale feature or roster
    assumptions are not training data. Detect them at the boundary so later
    matrix errors cannot be mistaken for an inference result.
    """
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid hard-negative JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("hard-negative artifact must be a JSON object")
    if "status" in payload:
        raise ValueError(
            "hard-negative failure report is not usable training data: "
            f"{payload['status']!r}"
        )

    missing = sorted(_REQUIRED_ARTIFACT_FIELDS - payload.keys())
    if missing:
        raise ValueError(f"hard-negative artifact missing required fields: {missing}")
    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "hard-negative feature schema mismatch: "
            f"expected {FEATURE_SCHEMA_VERSION}, "
            f"got {payload['feature_schema_version']!r}"
        )
    if payload["config_hash"] != config_hash():
        raise ValueError("hard-negative config_hash does not match live feature config")
    if payload["roster_config_hash"] != roster_config_hash():
        raise ValueError(
            "hard-negative roster_config_hash does not match live roster config"
        )
    if not isinstance(payload["records"], list):
        raise ValueError("hard-negative artifact records must be a list")
    if not all(isinstance(record, dict) for record in payload["records"]):
        raise ValueError("hard-negative artifact records must contain only objects")
    return payload


def basis_for_columns(columns: Sequence[str]) -> dict[str, str]:
    """Classify every flattened feature for promotion and maxT eligibility."""
    basis: dict[str, str] = {}
    for column in columns:
        if (
            column.startswith("value_pct_")
            or column.startswith("dim_ratios.")
            or column.startswith("dim_available.")
            or column.startswith("capped_group_totals.")
            or column in {"max_value_share", "total_size_points", "pref_violations"}
        ):
            basis[column] = "scored"
        elif column.startswith("raw_group_totals.") or column.startswith(
            "raw_tag_counts."
        ):
            basis[column] = "parent"
        elif column in _AGNOSTIC_COLUMNS:
            basis[column] = "agnostic"
        else:
            raise ValueError(f"unclassified flattened feature column: {column}")
    return basis


def _is_rung_negative(source: object, rung: str) -> bool:
    if rung == "manual_vs_synth":
        return isinstance(source, str) and source.startswith("synth_")
    if rung == "manual_vs_baseline":
        return isinstance(source, str) and source.startswith("baseline_")
    if rung == "manual_vs_ilp":
        return source == "ilp_optimal"
    raise ValueError(f"unknown EBM rung: {rung!r}")


def _cluster_key(record: dict) -> tuple[int, str, str]:
    """Return the matched-roster cluster identity carried by artifact rows."""
    try:
        offer_id = record["offer_id"]
        tier = record["tier"]
        box_name = record["box_name"]
    except KeyError as exc:
        raise ValueError(f"hard-negative record missing cluster field: {exc.args[0]}") from exc
    if not isinstance(offer_id, int) or not isinstance(tier, str) or not isinstance(
        box_name, str
    ):
        raise ValueError("hard-negative record has invalid offer_id, tier, or box_name")
    return offer_id, tier, box_name.casefold()


def prepare_rung(records: Sequence[dict], rung: str) -> PreparedRung:
    """Select a rung's complete matched box clusters and record attrition.

    The hard-negative generator reconstructs manual and negative rows over one
    roster intersection. Keep only clusters with a manual row and a rung-negative
    row, then use this exact row population for all later fits and permutations.
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown EBM rung: {rung!r}")

    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("hard-negative records must contain only objects")
        grouped[_cluster_key(record)].append(record)

    selected_records: list[dict] = []
    selected_clusters: list[tuple[int, str, str]] = []
    missing_manual = 0
    missing_negative = 0
    for cluster, cluster_records in grouped.items():
        manual = [record for record in cluster_records if record.get("source") == "manual"]
        negatives = [
            record
            for record in cluster_records
            if _is_rung_negative(record.get("source"), rung)
        ]
        if not manual:
            missing_manual += 1
            continue
        if not negatives:
            missing_negative += 1
            continue
        for record in cluster_records:
            if record.get("source") == "manual" or _is_rung_negative(
                record.get("source"), rung
            ):
                selected_records.append(record)
                selected_clusters.append(cluster)

    return PreparedRung(
        records=selected_records,
        clusters=selected_clusters,
        attrition={
            "input_clusters": len(grouped),
            "missing_manual_clusters": missing_manual,
            "missing_negative_clusters": missing_negative,
            "retained_clusters": len(grouped) - missing_manual - missing_negative,
        },
    )


def build_design_matrix(
    records: Sequence[dict],
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[tuple[int, str, str]]]:
    """Build numeric X plus labels and grouping identities from retained rows."""
    if not records:
        raise ValueError("cannot build an EBM matrix from no records")

    flattened = [flatten(record) for record in records]
    columns = sorted(flattened[0])
    expected = set(columns)
    for row in flattened[1:]:
        if set(row) != expected:
            raise ValueError("hard-negative records flatten to inconsistent columns")
    basis_for_columns(columns)

    X = np.asarray([[row[column] for column in columns] for row in flattened], dtype=float)
    labels = np.asarray(
        [1 if record.get("source") == "manual" else 0 for record in records], dtype=int
    )
    offers = np.asarray([record["offer_id"] for record in records], dtype=int)
    clusters = [_cluster_key(record) for record in records]
    return X, columns, labels, offers, clusters
