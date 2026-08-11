#!/usr/bin/env python3
"""Prepare DB-free inputs for the EBM hard-negative diagnostic.

The diagnostic intentionally consumes only the validated artifact written by
``generate_hard_negatives.py``. It never imports allocation, comparison, or DB
code: those paths belong exclusively to artifact generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
_TIERS = ("small", "medium", "large")
_MIN_MANUAL_ROWS = 150
_MIN_OFFERS = 20
_RUNG_SEED_OFFSETS = {rung: index for index, rung in enumerate(RUNGS)}


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


def load_artifact(path: Path, *, raw_bytes: bytes | None = None) -> dict:
    """Load a successful, current hard-negative artifact or fail early.

    A failure report and an artifact generated under stale feature or roster
    assumptions are not training data. Detect them at the boundary so later
    matrix errors cannot be mistaken for an inference result.
    """
    try:
        payload = json.loads(path.read_bytes() if raw_bytes is None else raw_bytes)
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
    for index, record in enumerate(payload["records"]):
        _validate_artifact_record(record, index)
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
    if (
        isinstance(offer_id, bool)
        or not isinstance(offer_id, int)
        or not isinstance(tier, str)
        or not isinstance(box_name, str)
    ):
        raise ValueError("hard-negative record has invalid offer_id, tier, or box_name")
    return offer_id, tier, box_name.casefold()


def _validate_artifact_record(record: dict, index: int) -> None:
    """Verify every artifact row can form a valid diagnostic matrix row."""
    try:
        _cluster_key(record)
        if not isinstance(record.get("source"), str):
            raise ValueError("source must be a string")
        flatten(record)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"hard-negative artifact record {index} is not valid diagnostic input"
        ) from exc


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


def _validate_complete_matched_clusters(
    labels: np.ndarray, clusters: Sequence[tuple[int, str, str]]
) -> None:
    """Require the paired manual/negative structure assumed by the null test."""
    if len(clusters) != labels.shape[0]:
        raise ValueError("clusters must provide exactly one identity per label")
    grouped_labels: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for label, cluster in zip(labels.tolist(), clusters):
        grouped_labels[cluster].append(label)
    incomplete = [
        cluster
        for cluster, cluster_labels in grouped_labels.items()
        if set(cluster_labels) != {0, 1}
    ]
    if incomplete:
        raise ValueError(
            "maxT requires every row to belong to a complete manual/negative "
            f"cluster; incomplete clusters include {incomplete[:3]!r}"
        )


def _is_promotable_column(column: str, basis: str) -> bool:
    """Return whether a term can enter the primary maxT family and findings."""
    # Tag parents require an aggregate-preserving refit that is intentionally
    # deferred in this MVP.  Keep them in the fitted model, but do not treat a
    # plain parent classification as permission to promote them.
    return basis in {"parent", "agnostic"} and not column.startswith(
        "raw_tag_counts."
    )


def _clears_maxt(importance: float, threshold: float) -> bool:
    """Return whether a term strictly exceeds the discrete null cutoff."""
    return importance > threshold


def _rung_seed(seed: int, rung: str) -> int:
    """Derive a stable per-rung seed independent of CLI ordering."""
    return seed + _RUNG_SEED_OFFSETS[rung]


def run_maxt(
    X: np.ndarray,
    labels: np.ndarray,
    clusters: Sequence[tuple[int, str, str]],
    columns: Sequence[str],
    basis: dict[str, str],
    seed: int,
    n_permutations: int,
    observed_fit: FittedRung | None = None,
) -> MaxTResult:
    """Fit the observed EBM and its matched-cluster, promotable-family maxT null."""
    if n_permutations < 200:
        raise ValueError("maxT requires at least 200 permutations")
    matrix, target = _validated_fit_inputs(X, labels, columns)
    _validate_complete_matched_clusters(target, clusters)
    if set(basis) != set(columns):
        raise ValueError("maxT basis must classify exactly the diagnostic columns")
    family_columns = [
        column
        for column in columns
        if _is_promotable_column(column, basis[column])
    ]
    if not family_columns:
        raise ValueError("maxT requires at least one promotable parent or agnostic term")

    if observed_fit is None:
        observed_fit = fit_full_ebm(matrix, target, columns, seed)
    observed_importances = {
        column: observed_fit.importances[column] for column in family_columns
    }
    rng = np.random.default_rng(seed)
    null_maxima: list[float] = []
    for permutation_index in range(n_permutations):
        permuted_labels = permute_labels_within_clusters(target, clusters, rng)
        permuted_fit = fit_full_ebm(
            matrix, permuted_labels, columns, seed=seed
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


def _cohort_counts(records: Sequence[dict]) -> tuple[int, int, int]:
    """Return manual rows, negative rows, and distinct offers for one rung."""
    labels = [1 if record.get("source") == "manual" else 0 for record in records]
    return (
        labels.count(1),
        labels.count(0),
        len({record["offer_id"] for record in records}),
    )


def _is_underpowered(n_manual: int, n_offers: int) -> bool:
    """Apply the registered floor before any EBM inference is attempted."""
    return n_manual < _MIN_MANUAL_ROWS or n_offers < _MIN_OFFERS


def fit_without_value_pct(
    X: np.ndarray, labels: np.ndarray, columns: Sequence[str], seed: int
) -> FittedRung:
    """Refit after dropping all tier-sliced value columns as one group."""
    retained_indices = [
        index
        for index, column in enumerate(columns)
        if not column.startswith("value_pct_")
    ]
    if len(retained_indices) == len(columns):
        raise ValueError("EBM matrix has no tier-sliced value_pct columns to ablate")
    retained_columns = [columns[index] for index in retained_indices]
    return fit_full_ebm(
        np.asarray(X)[:, retained_indices], labels, retained_columns, seed
    )


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a defined Pearson r, leaving degenerate tier slices explicit."""
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _correlations_by_tier(
    X: np.ndarray,
    columns: Sequence[str],
    records: Sequence[dict],
    term: str,
    reference_term: str,
) -> dict[str, float | None]:
    """Calculate a transparent per-tier correlation for finding metadata."""
    if term not in columns or reference_term not in columns:
        return {tier: None for tier in _TIERS}
    matrix = np.asarray(X, dtype=float)
    if matrix.shape[0] != len(records):
        raise ValueError("finding records must align with the EBM matrix")
    term_index = list(columns).index(term)
    reference_index = list(columns).index(reference_term)
    result: dict[str, float | None] = {}
    for tier in _TIERS:
        indices = [
            index for index, record in enumerate(records) if record["tier"] == tier
        ]
        result[tier] = _pearson(
            matrix[indices, term_index], matrix[indices, reference_index]
        )
    return result


def _value_confound_correlations(
    X: np.ndarray, columns: Sequence[str], records: Sequence[dict], term: str
) -> dict[str, float | None]:
    """Measure each term against the matching tier-sliced value covariate."""
    if term not in columns:
        return {tier: None for tier in _TIERS}
    matrix = np.asarray(X, dtype=float)
    if matrix.shape[0] != len(records):
        raise ValueError("finding records must align with the EBM matrix")
    term_index = list(columns).index(term)
    result: dict[str, float | None] = {}
    for tier in _TIERS:
        value_term = f"value_pct_{tier}"
        if value_term not in columns:
            result[tier] = None
            continue
        indices = [
            index for index, record in enumerate(records) if record["tier"] == tier
        ]
        result[tier] = _pearson(
            matrix[indices, term_index],
            matrix[indices, list(columns).index(value_term)],
        )
    return result


def _parent_metadata(
    X: np.ndarray, columns: Sequence[str], records: Sequence[dict], term: str
) -> dict[str, object]:
    """Add the affordable group check and state the deferred tag-parent check."""
    if term.startswith("raw_group_totals."):
        aggregate_term = (
            f"capped_group_totals.{term.removeprefix('raw_group_totals.')}"
        )
        correlations = _correlations_by_tier(X, columns, records, term, aggregate_term)
        return {
            "aggregate_term": aggregate_term,
            "aggregate_r": correlations,
            "aggregate_explained": any(
                correlation is not None and abs(correlation) > 0.7
                for correlation in correlations.values()
            ),
            "aggregate_check": "per-tier Pearson correlation",
        }
    if term.startswith("raw_tag_counts."):
        return {
            "aggregate_term": None,
            "aggregate_r": None,
            "aggregate_explained": None,
            "aggregate_check": "deferred tag-parent aggregate-preserving refit",
        }
    return {
        "aggregate_term": None,
        "aggregate_r": None,
        "aggregate_explained": None,
        "aggregate_check": None,
    }


def _build_findings(
    ablated_fit: FittedRung,
    maxt: MaxTResult,
    X: np.ndarray,
    records: Sequence[dict],
    columns: Sequence[str],
    basis: dict[str, str],
) -> list[dict[str, object]]:
    """Turn maxT-clearing promotable terms into caveated hypotheses."""
    findings: list[dict[str, object]] = []
    for term, importance in maxt.observed_importances.items():
        if not _clears_maxt(importance, maxt.threshold):
            continue
        value_correlations = _value_confound_correlations(X, columns, records, term)
        finding: dict[str, object] = {
            "term": term,
            "importance": importance,
            "basis": basis[term],
            "value_confound_r": value_correlations,
            "ablation_drop_value_pct": {
                "importance": ablated_fit.importances.get(term),
                "interpretation": "descriptive full-fit importance only",
            },
        }
        finding.update(_parent_metadata(X, columns, records, term))
        findings.append(finding)
    return sorted(
        findings,
        key=lambda finding: (-float(finding["importance"]), str(finding["term"])),
    )


def _diagnostic_versions() -> dict[str, str]:
    """Record installed library versions without importing their runtime modules."""
    return {
        distribution: version(distribution)
        for distribution in (
            "interpret",
            "scikit-learn",
            "numpy",
        )
    }


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    """Publish findings only after a complete JSON file has reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run(
    features: Path,
    out: Path,
    *,
    seed: int,
    permutations: int,
    rungs: Sequence[str],
) -> dict[str, object]:
    """Run every requested EBM rung and write one deterministic findings report."""
    requested_rungs = list(rungs)
    unknown_rungs = [rung for rung in requested_rungs if rung not in RUNGS]
    if unknown_rungs:
        raise ValueError(f"unknown EBM rungs: {unknown_rungs!r}")
    if not requested_rungs:
        raise ValueError("at least one EBM rung is required")
    if permutations < 200:
        raise ValueError("maxT requires at least 200 permutations")

    artifact_bytes = features.read_bytes()
    artifact = load_artifact(features, raw_bytes=artifact_bytes)
    rung_results: dict[str, object] = {}
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact": {
            "feature_schema_version": artifact["feature_schema_version"],
            "config_hash": artifact["config_hash"],
            "roster_config_hash": artifact["roster_config_hash"],
            "input_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "run_metadata": artifact["run_metadata"],
            "source_counts": artifact["source_counts"],
            "attrition": artifact["attrition"],
        },
        "methodology": {
            "seed": seed,
            "permutations": permutations,
            "interactions_enabled": False,
            "balance": "equal total class weight",
            "validation": "five-fold GroupKFold by offer_id",
            "permutation": "cluster-local labels; fixed EBM seed; full-data refits",
            "maxT_family": (
                "agnostic plus group-parent terms; scored and deferred "
                "tag-parent terms excluded"
            ),
            "versions": _diagnostic_versions(),
            "caveats": (
                "Findings are hypothesis-generating only. value confounding is "
                "reported through a drop-value ablation but is not resolved by it. "
                "The tag-parent aggregate-preserving refit is deferred, so tag "
                "parents are not promotable."
            ),
        },
        "rungs": rung_results,
    }

    for rung in requested_rungs:
        prepared = prepare_rung(artifact["records"], rung)
        n_pos, n_neg, n_offers = _cohort_counts(prepared.records)
        rung_seed = _rung_seed(seed, rung)
        rung_result: dict[str, object] = {
            "underpowered": _is_underpowered(n_pos, n_offers),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_offers": n_offers,
            "model_seed": rung_seed,
            "attrition": prepared.attrition,
            "findings": [],
        }
        if not rung_result["underpowered"]:
            X, columns, labels, offers, clusters = build_design_matrix(prepared.records)
            _validate_complete_matched_clusters(labels, clusters)
            basis = basis_for_columns(columns)
            full_fit = fit_full_ebm(X, labels, columns, rung_seed)
            ablated_fit = fit_without_value_pct(X, labels, columns, rung_seed)
            maxt = run_maxt(
                X,
                labels,
                clusters,
                columns,
                basis,
                seed=rung_seed,
                n_permutations=permutations,
                observed_fit=full_fit,
            )
            null = np.asarray(maxt.null_maxima, dtype=float)
            rung_result.update(
                {
                    "column_count": len(columns),
                    "auc_oof": auc_group_kfold(X, labels, offers, columns, rung_seed),
                    "auc_insample": full_fit.auc_insample,
                    "maxt_threshold": maxt.threshold,
                    "maxT_family_count": len(maxt.family_columns),
                    "maxT_family_columns": maxt.family_columns,
                    "maxt_null_quantiles": {
                        "p50": float(np.quantile(null, 0.5)),
                        "p95": float(np.quantile(null, 0.95)),
                        "max": float(np.max(null)),
                    },
                    "findings": _build_findings(
                        ablated_fit,
                        maxt,
                        X,
                        prepared.records,
                        columns,
                        basis,
                    ),
                }
            )
        rung_results[rung] = rung_result

    _write_json_atomically(out, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the small, documented command-line interface for the diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("diagnostics/hard_negatives.json"),
        help="validated hard-negative artifact",
    )
    parser.add_argument("--seed", type=int, default=0, help="fixed EBM/randomization seed")
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="matched-cluster maxT draws (minimum: 200)",
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        choices=RUNGS,
        default=list(RUNGS),
        help="one or more manual-vs-negative rungs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("diagnostics/ebm_findings.json"),
        help="output findings JSON",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="accepted for compatibility; plots are deferred from this MVP",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the DB-free diagnostic."""
    args = build_parser().parse_args(argv)
    run(
        args.features,
        args.out,
        seed=args.seed,
        permutations=args.permutations,
        rungs=args.rungs,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
