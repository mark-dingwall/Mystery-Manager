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


@dataclass
class FittedRung:
    """A full-data EBM fit and the statistics used by the diagnostic."""

    model: object
    importances: dict[str, float]
    auc_insample: float


@dataclass
class MaxTResult:
    """Observed promotable importances and their cluster-permutation null."""

    observed_importances: dict[str, float]
    null_maxima: list[float]
    threshold: float
    family_columns: list[str]


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


def _validated_fit_inputs(
    X: np.ndarray, labels: np.ndarray, columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate the binary matrix contract shared by all fits."""
    matrix = np.asarray(X, dtype=float)
    target = np.asarray(labels, dtype=int)
    if matrix.ndim != 2:
        raise ValueError("EBM design matrix must be two-dimensional")
    if matrix.shape[0] == 0:
        raise ValueError("cannot fit an EBM with no rows")
    if matrix.shape[1] != len(columns):
        raise ValueError("EBM design matrix width does not match its columns")
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise ValueError("EBM labels must be a one-dimensional vector matching X")
    if set(target.tolist()) != {0, 1}:
        raise ValueError("EBM labels must contain both binary classes 0 and 1")
    if not np.isfinite(matrix).all():
        raise ValueError("EBM design matrix contains non-finite values")
    return matrix, target


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Give each class half of the total fitting weight.

    A rung can have many more generated negatives than manual rows.  Equal total
    class mass makes every full and cross-validated fit answer the same
    manual-versus-negative question instead of reflecting source multiplicity.
    """
    target = np.asarray(labels, dtype=int)
    if target.ndim != 1 or set(target.tolist()) != {0, 1}:
        raise ValueError("sample weights require both binary classes 0 and 1")
    positive_count = int(np.count_nonzero(target == 1))
    negative_count = int(np.count_nonzero(target == 0))
    return np.where(target == 1, 0.5 / positive_count, 0.5 / negative_count)


def _new_ebm(columns: Sequence[str], seed: int):
    """Construct the deterministic, main-effects-only EBM used by every fit."""
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as exc:  # pragma: no cover - exercised by dependency gate
        raise RuntimeError(
            "InterpretML is required for EBM diagnostics; install "
            "requirements-diagnostics.txt"
        ) from exc

    # Feature names belong in the constructor in InterpretML 0.7.x; passing
    # them to fit() is not supported.  Keep one worker so a seed reproduces the
    # serial permutation null across runs.
    return ExplainableBoostingClassifier(
        feature_names=list(columns),
        interactions=0,
        n_jobs=1,
        random_state=seed,
    )


def _positive_probabilities(model: object, X: np.ndarray) -> np.ndarray:
    """Return the probability column corresponding to the manual class (1)."""
    classes = list(np.asarray(model.classes_).tolist())
    if 1 not in classes:
        raise ValueError("EBM model did not retain the manual class")
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("EBM returned malformed class probabilities")
    return probabilities[:, classes.index(1)]


def fit_full_ebm(
    X: np.ndarray, labels: np.ndarray, columns: Sequence[str], seed: int
) -> FittedRung:
    """Fit one class-balanced, named, main-effects EBM on the full cohort."""
    matrix, target = _validated_fit_inputs(X, labels, columns)
    model = _new_ebm(columns, seed)
    model.fit(matrix, target, sample_weight=balanced_sample_weights(target))

    term_names = list(model.term_names_)
    if term_names != list(columns):
        raise ValueError(
            "main-effects EBM term names do not match the diagnostic feature columns"
        )
    importances = {
        name: float(importance)
        for name, importance in zip(term_names, model.term_importances("avg_weight"))
    }

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:  # pragma: no cover - exercised by dependency gate
        raise RuntimeError(
            "scikit-learn is required for EBM diagnostics; install "
            "requirements-diagnostics.txt"
        ) from exc
    return FittedRung(
        model=model,
        importances=importances,
        auc_insample=float(roc_auc_score(target, _positive_probabilities(model, matrix))),
    )


def auc_group_kfold(
    X: np.ndarray,
    labels: np.ndarray,
    offers: np.ndarray,
    columns: Sequence[str],
    seed: int,
) -> float:
    """Calculate OOF AUC with every validation fold holding out whole offers."""
    matrix, target = _validated_fit_inputs(X, labels, columns)
    groups = np.asarray(offers)
    if groups.ndim != 1 or groups.shape[0] != matrix.shape[0]:
        raise ValueError("offer groups must be a one-dimensional vector matching X")
    if len(np.unique(groups)) < 5:
        raise ValueError("GroupKFold requires at least five distinct offers")

    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
    except ImportError as exc:  # pragma: no cover - exercised by dependency gate
        raise RuntimeError(
            "scikit-learn is required for EBM diagnostics; install "
            "requirements-diagnostics.txt"
        ) from exc

    predictions = np.full(matrix.shape[0], np.nan, dtype=float)
    splitter = GroupKFold(n_splits=5)
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(matrix, target, groups)
    ):
        train_labels = target[train_indices]
        validation_labels = target[validation_indices]
        if set(train_labels.tolist()) != {0, 1} or set(validation_labels.tolist()) != {
            0,
            1,
        }:
            raise ValueError("each offer-held-out EBM fold must contain both classes")
        fitted = fit_full_ebm(
            matrix[train_indices], train_labels, columns, seed=seed + fold_index
        )
        predictions[validation_indices] = _positive_probabilities(
            fitted.model, matrix[validation_indices]
        )
    if not np.isfinite(predictions).all():
        raise ValueError("GroupKFold did not produce an OOF prediction for every row")
    return float(roc_auc_score(target, predictions))


def permute_labels_within_clusters(
    labels: np.ndarray,
    clusters: Sequence[tuple[int, str, str]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle labels only among rows sharing one matched roster cluster."""
    target = np.asarray(labels, dtype=int)
    if target.ndim != 1 or target.shape[0] != len(clusters):
        raise ValueError("clusters must provide exactly one identity per label")

    grouped_indices: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped_indices[cluster].append(index)

    permuted = target.copy()
    for indices in grouped_indices.values():
        shuffled_indices = rng.permutation(indices)
        permuted[indices] = target[shuffled_indices]
    return permuted


def run_maxt(
    X: np.ndarray,
    labels: np.ndarray,
    clusters: Sequence[tuple[int, str, str]],
    columns: Sequence[str],
    basis: dict[str, str],
    seed: int,
    n_permutations: int,
) -> MaxTResult:
    """Fit the observed EBM and its matched-cluster, promotable-family maxT null."""
    if n_permutations < 200:
        raise ValueError("maxT requires at least 200 permutations")
    matrix, target = _validated_fit_inputs(X, labels, columns)
    if set(basis) != set(columns):
        raise ValueError("maxT basis must classify exactly the diagnostic columns")
    family_columns = [
        column for column in columns if basis[column] in {"parent", "agnostic"}
    ]
    if not family_columns:
        raise ValueError("maxT requires at least one promotable parent or agnostic term")

    observed_fit = fit_full_ebm(matrix, target, columns, seed)
    observed_importances = {
        column: observed_fit.importances[column] for column in family_columns
    }
    rng = np.random.default_rng(seed)
    null_maxima: list[float] = []
    for permutation_index in range(n_permutations):
        permuted_labels = permute_labels_within_clusters(target, clusters, rng)
        permuted_fit = fit_full_ebm(
            matrix, permuted_labels, columns, seed=seed + permutation_index + 1
        )
        null_maxima.append(
            max(permuted_fit.importances[column] for column in family_columns)
        )

    return MaxTResult(
        observed_importances=observed_importances,
        null_maxima=null_maxima,
        threshold=float(np.quantile(np.asarray(null_maxima), 0.95, method="higher")),
        family_columns=family_columns,
    )
