#!/usr/bin/env python3
"""Generate and validate the hard-negative feature artifact.

The helpers in this section are deliberately DB-free. Runtime orchestration is
added below them separately so artifact consumers and tests can import the
contract without importing database or allocation infrastructure.
"""

import argparse
import copy
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from allocator.box_features import FEATURE_SCHEMA_VERSION, config_hash
from allocator.config import (
    DONATION_IDENTIFIERS,
    SKIP_COLUMN_IDENTIFIERS,
    STAFF_IDENTIFIERS,
)
from allocator.hard_negative_roster import (
    AmbiguousRosterIdentityError,
    correct_box_tiers,
    intersect_roster,
    roster_config_hash,
)


TIER_A_OFFER_IDS = frozenset(range(64, 110))
ADMITTING_ILP_STATUS = "Optimal Solution Found"
GENERATOR_VERSION = 1
BASELINE_SOURCES = (
    "baseline_deal_topup",
    "baseline_minmax_deficit",
    "baseline_greedy_best_fit",
)
RUNG_NEGATIVE_SELECTORS = {
    "manual_vs_synth": "synth_",
    "manual_vs_baseline": "baseline_",
    "manual_vs_ilp": "ilp_optimal",
}


def parse_requested_tier_a_offer_ids(value: str) -> list[int]:
    """Expand a comma-separated Tier-A offer selection into sorted IDs."""
    requested: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("requested Tier-A offer IDs must be non-empty")
        if "-" in part:
            try:
                low_text, high_text = part.split("-", 1)
                low = int(low_text)
                high = int(high_text)
            except ValueError as exc:
                raise ValueError(
                    f"invalid Tier-A offer selection: {part!r}"
                ) from exc
            outside_tier_a = [
                endpoint
                for endpoint in (low, high)
                if endpoint not in TIER_A_OFFER_IDS
            ]
            if outside_tier_a:
                raise ValueError(
                    f"range endpoints are outside Tier-A: {outside_tier_a}"
                )
            if high < low:
                raise ValueError(f"invalid Tier-A offer selection: {part!r}")
            requested.update(range(low, high + 1))
        else:
            try:
                requested.add(int(part))
            except ValueError as exc:
                raise ValueError(
                    f"invalid Tier-A offer selection: {part!r}"
                ) from exc

    if not requested:
        raise ValueError("requested Tier-A offer IDs must be non-empty")
    outside_tier_a = sorted(requested - TIER_A_OFFER_IDS)
    if outside_tier_a:
        raise ValueError(f"offer IDs are outside Tier-A: {outside_tier_a}")
    return sorted(requested)


def default_tier_a_offer_ids(available: set[int]) -> list[int]:
    """Return all available Tier-A offer IDs in deterministic order."""
    return sorted(available & TIER_A_OFFER_IDS)


def resolve_requested_offer_ids(
    requested: list[int], available: set[int]
) -> list[int]:
    """Resolve requested IDs against both availability and the Tier-A policy."""
    resolved = sorted(set(requested) & available & TIER_A_OFFER_IDS)
    if not resolved:
        raise ValueError("resolved Tier-A offer IDs must be non-empty")
    return resolved


def customer_csv_names(names: Iterable[str]) -> list[str]:
    """Remove known non-customer historical columns using exact-name policy."""
    excluded = DONATION_IDENTIFIERS | SKIP_COLUMN_IDENTIFIERS | STAFF_IDENTIFIERS
    return [name for name in names if name not in excluded]


def unextracted_row_reason(
    allocations: Mapping[int, int], item_lookup: Mapping[int, object]
) -> Literal["empty", "unextractable"]:
    """Classify why a feature extractor returned no row.

    Callers invoke this only after extraction returned ``None``. Under that
    precondition, a positive quantity cannot refer to a resolvable item.
    """
    if not any(quantity > 0 for quantity in allocations.values()):
        return "empty"
    return "unextractable"


def tags_for_preference(
    preference: str | None, variants: dict[str, dict[str, set[str]]]
) -> dict[str, set[str]]:
    """Select the exact tag denominator variant for a box preference."""
    if preference in ("fruit_only", "veg_only"):
        return variants[preference]
    return variants["all"]


def admits_to_ilp_class(status: str | None) -> bool:
    """Return whether a solver result may enter the ILP negative class."""
    return status == ADMITTING_ILP_STATUS


def paired_rung_coverage(
    records: list[dict], negative_selector: str
) -> dict[str, int]:
    """Count manual rows and offers only in cells paired with a negative class."""
    manual_cells = {
        (record["offer_id"], record["tier"])
        for record in records
        if record["source"] == "manual"
    }
    if negative_selector == "ilp_optimal":
        negative_cells = {
            (record["offer_id"], record["tier"])
            for record in records
            if record["source"] == "ilp_optimal"
        }
    else:
        negative_cells = {
            (record["offer_id"], record["tier"])
            for record in records
            if record["source"].startswith(negative_selector)
        }
    paired_cells = manual_cells & negative_cells
    paired_manual = [
        record
        for record in records
        if record["source"] == "manual"
        and (record["offer_id"], record["tier"]) in paired_cells
    ]
    return {
        "manual_boxes": len(paired_manual),
        "offers": len({record["offer_id"] for record in paired_manual}),
    }


def selected_roster_contract_failures(
    records: list[dict], expected: dict[tuple[int, str], dict]
) -> list[dict]:
    """Report rows that do not match the selected canonical roster contract."""
    failures: list[dict] = []
    for record in records:
        key = (record["offer_id"], record["box_name"].casefold())
        expectation = expected.get(key)
        context = {
            "gate": "selected_roster_contract",
            "offer_id": record["offer_id"],
            "box_name": record["box_name"],
            "source": record["source"],
        }
        if expectation is None:
            failures.append({**context, "reason": "unselected_box"})
            continue
        if record["tier"] != expectation["tier"]:
            failures.append({
                **context,
                "reason": "tier_mismatch",
                "expected": expectation["tier"],
                "actual": record["tier"],
            })
        if record.get("dim_available") != expectation["dim_available"]:
            failures.append({
                **context,
                "reason": "dim_available_mismatch",
                "expected": expectation["dim_available"],
                "actual": record.get("dim_available"),
            })
    return failures


def validation_failures(
    records: list[dict],
    source_counts: dict[str, int],
    roster_contract_failures: Sequence[dict] = (),
) -> list[dict]:
    """Return all failed write gates for a provisional artifact."""
    failures: list[dict] = []

    invalid_ilp_rows = [
        record
        for record in records
        if record["source"] == "ilp_optimal"
        and not admits_to_ilp_class(record.get("solver_status"))
    ]
    if invalid_ilp_rows:
        failures.append({
            "gate": "ilp_optimal.status",
            "required": ADMITTING_ILP_STATUS,
            "invalid_rows": len(invalid_ilp_rows),
            "statuses": dict(sorted(Counter(
                record.get("solver_status") or "<missing>"
                for record in invalid_ilp_rows
            ).items())),
        })

    for rung, selector in RUNG_NEGATIVE_SELECTORS.items():
        coverage = paired_rung_coverage(records, selector)
        if coverage["manual_boxes"] < 150 or coverage["offers"] < 20:
            failures.append({
                "gate": rung,
                "required": {"manual_boxes": 150, "offers": 20},
                "actual": coverage,
            })

    missing_sources = []
    if source_counts.get("manual", 0) <= 0:
        missing_sources.append("manual")
    if not any(
        source.startswith("synth_") and count > 0
        for source, count in source_counts.items()
    ):
        missing_sources.append("synth_*")
    missing_sources.extend(
        source for source in BASELINE_SOURCES if source_counts.get(source, 0) <= 0
    )
    if source_counts.get("ilp_optimal", 0) <= 0:
        missing_sources.append("ilp_optimal")
    if missing_sources:
        failures.append({
            "gate": "required_sources",
            "missing": missing_sources,
        })

    failures.extend(roster_contract_failures)
    return failures


def build_artifact(
    records: list[dict],
    source_counts: dict[str, int],
    roster_check: dict,
    attrition: dict,
    exclusions: list[dict],
    requested_offer_ids: list[int],
    resolved_offer_ids: list[int],
) -> dict:
    """Build the complete successful artifact with deterministic row ordering."""
    sorted_records = sorted(
        records,
        key=lambda record: (
            record["offer_id"],
            record["tier"],
            record["source"],
            record["box_name"],
            record.get("solver_status", ""),
        ),
    )
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "config_hash": config_hash(),
        "roster_config_hash": roster_config_hash(),
        "records": sorted_records,
        "source_counts": source_counts,
        "roster_check": roster_check,
        "attrition": attrition,
        "exclusions": exclusions,
        "run_metadata": {
            "requested_offer_ids": requested_offer_ids,
            "resolved_offer_ids": resolved_offer_ids,
            "generator_version": GENERATOR_VERSION,
        },
    }


def build_failure_report(
    status: str,
    failed_gates: list[dict],
    source_counts: dict[str, int],
    roster_check: dict,
    attrition: dict,
    exclusions: list[dict],
    errors: list[dict],
    requested_offer_ids: list[int],
    resolved_offer_ids: list[int],
) -> dict:
    """Build the exact audit-only payload for an unsuccessful run."""
    return {
        "status": status,
        "failed_gates": failed_gates,
        "source_counts": source_counts,
        "roster_check": roster_check,
        "attrition": attrition,
        "run_metadata": {
            "requested_offer_ids": requested_offer_ids,
            "resolved_offer_ids": resolved_offer_ids,
            "generator_version": GENERATOR_VERSION,
        },
        "exclusions": exclusions,
        "errors": errors,
    }


def write_json_atomically(path: Path, payload: dict) -> None:
    """Durably replace a JSON file only after a complete temporary write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def finalize_run(
    records: list[dict],
    requested_offer_ids: list[int],
    resolved_offer_ids: list[int],
    roster_check: dict,
    attrition: dict,
    exclusions: list[dict],
    errors: list[dict],
    roster_contract_failures: Sequence[dict],
    out_path: Path,
    report_path: Path,
) -> int:
    """Write either a complete valid artifact or an audit-only failure report."""
    if out_path.resolve() == report_path.resolve():
        raise ValueError("artifact and failure-report paths must be distinct")

    source_counts = dict(sorted(Counter(
        record["source"] for record in records
    ).items()))
    if errors:
        report = build_failure_report(
            "execution_failed",
            [],
            source_counts,
            roster_check,
            attrition,
            exclusions,
            errors,
            requested_offer_ids,
            resolved_offer_ids,
        )
        write_json_atomically(report_path, report)
        return 1

    failed_gates = validation_failures(
        records, source_counts, roster_contract_failures
    )
    if failed_gates:
        report = build_failure_report(
            "validation_failed",
            failed_gates,
            source_counts,
            roster_check,
            attrition,
            exclusions,
            errors,
            requested_offer_ids,
            resolved_offer_ids,
        )
        write_json_atomically(report_path, report)
        return 1

    artifact = build_artifact(
        records,
        source_counts,
        roster_check,
        attrition,
        exclusions,
        requested_offer_ids,
        resolved_offer_ids,
    )
    write_json_atomically(out_path, artifact)
    return 0
