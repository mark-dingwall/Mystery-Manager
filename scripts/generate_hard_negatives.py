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
import sys
import tempfile
from typing import Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


@dataclass
class OfferOutcome:
    offer_id: int
    records: list[dict]
    roster_entry: dict
    attrition: dict
    roster_contract_failures: list[dict] = field(default_factory=list)
    exclusion: dict | None = None
    error: dict | None = None


def empty_roster_entry(offer_id: int) -> dict:
    """Return the complete roster-audit shape for an unprocessed offer."""
    return {
        "offer_id": offer_id,
        "csv_only": [],
        "db_only": [],
        "selected_count": 0,
    }


def empty_offer_attrition() -> dict:
    """Return the fixed per-offer attrition shape used by orchestration."""
    return {
        "roster_candidates": {"csv": 0, "db": 0, "selected": 0},
        "solver_statuses": {},
        "row_attrition": {},
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

    if errors:
        source_counts = dict(sorted(Counter(
            record["source"]
            for record in records
            if isinstance(record.get("source"), str)
        ).items()))
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

    source_counts = dict(sorted(Counter(
        record["source"] for record in records
    ).items()))
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


def process_offer(offer_id: int) -> OfferOutcome:
    """Generate every hard-negative source for one reconciled customer roster."""
    import compare
    from allocator.allocator import allocate, build_boxes_from_db
    from allocator.box_features import (
        UnsupportedCategoryError,
        extract_box_features,
    )
    from scripts.extract_features import (
        EmptyPreferenceItemPoolError,
        SyntheticTemplate,
        generate_synthetic_boxes,
    )

    roster_entry = empty_roster_entry(offer_id)
    attrition = empty_offer_attrition()

    xlsx_path = compare._find_xlsx_path(offer_id)
    if xlsx_path is None:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "missing_xlsx",
                "detail": "shopping-list XLSX not found",
            },
        )

    item_lookup = compare.build_item_lookup(offer_id)
    if not item_lookup:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "missing_item_lookup",
                "detail": "item lookup is empty",
            },
        )

    raw_csv_names, historical_allocations = compare.load_historical_csv(offer_id)
    csv_names = customer_csv_names(raw_csv_names)
    attrition["roster_candidates"]["csv"] = len(csv_names)
    if not csv_names:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "missing_historical_csv",
                "detail": "historical CSV has no customer columns",
            },
        )

    try:
        db_boxes = correct_box_tiers(offer_id, build_boxes_from_db(offer_id))
        attrition["roster_candidates"]["db"] = len(db_boxes)
        intersection = intersect_roster(csv_names, [box.name for box in db_boxes])
    except AmbiguousRosterIdentityError as exc:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "ambiguous_roster_identity",
                "detail": str(exc),
            },
        )

    attrition["roster_candidates"]["selected"] = len(intersection.matches)
    roster_entry.update({
        "csv_only": list(intersection.csv_only),
        "db_only": list(intersection.db_only),
        "selected_count": len(intersection.matches),
    })
    if not intersection.matches:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "empty_roster_intersection",
                "detail": "CSV and DB customer rosters do not overlap",
            },
        )

    boxes_by_identity = {box.name.casefold(): box for box in db_boxes}
    selected = [
        (match.csv_name, copy.deepcopy(boxes_by_identity[match.db_name.casefold()]))
        for match in intersection.matches
    ]
    selected_boxes = [box for _csv_name, box in selected]
    variants = {
        "all": compare.compute_available_tags(item_lookup),
        "fruit_only": compare.compute_available_tags(
            item_lookup, preference="fruit_only"
        ),
        "veg_only": compare.compute_available_tags(
            item_lookup, preference="veg_only"
        ),
    }

    ilp_result = allocate(
        offer_id,
        xlsx_path,
        boxes=copy.deepcopy(selected_boxes),
        strategy="ilp-optimal",
    )
    solver_status = ilp_result.solver_status
    solver_status_key = solver_status or "<missing>"
    attrition["solver_statuses"][solver_status_key] = 1
    if not admits_to_ilp_class(solver_status):
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "nonoptimal_ilp",
                "detail": solver_status_key,
            },
        )

    baseline_results = []
    for strategy, source in (
        ("deal-topup", "baseline_deal_topup"),
        ("minmax-deficit", "baseline_minmax_deficit"),
        ("greedy-best-fit", "baseline_greedy_best_fit"),
    ):
        result = allocate(
            offer_id,
            xlsx_path,
            boxes=copy.deepcopy(selected_boxes),
            strategy=strategy,
        )
        baseline_results.append((source, result))

    provisional_records: list[dict] = []

    def extract_and_append(box, allocations: dict[int, int], source: str) -> None:
        tags = tags_for_preference(box.preference, variants)
        feature = extract_box_features(
            box.name,
            allocations,
            item_lookup,
            box.tier,
            tags,
            offer_id,
            source=source,
            preference=box.preference,
        )
        if feature is None:
            source_attrition = attrition["row_attrition"].setdefault(
                source, {"empty": 0, "unextractable": 0}
            )
            reason = unextracted_row_reason(allocations, item_lookup)
            source_attrition[reason] += 1
            return
        if source == "ilp_optimal":
            feature["solver_status"] = solver_status
        provisional_records.append(feature)

    try:
        for csv_name, box in selected:
            manual_allocations = {
                item_id: allocations[csv_name]
                for item_id, allocations in historical_allocations.items()
                if csv_name in allocations
            }
            extract_and_append(box, manual_allocations, "manual")

        for box in ilp_result.boxes:
            extract_and_append(box, box.allocations, "ilp_optimal")

        for source, result in baseline_results:
            for box in result.boxes:
                extract_and_append(box, box.allocations, source)

        templates = [
            SyntheticTemplate(
                box.name,
                box.tier,
                box.preference,
                tags_for_preference(box.preference, variants),
            )
            for box in selected_boxes
        ]
        provisional_records.extend(
            generate_synthetic_boxes(
                offer_id,
                item_lookup,
                variants["all"],
                templates=templates,
            )
        )
    except UnsupportedCategoryError as exc:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "unsupported_category",
                "detail": str(exc),
            },
        )
    except EmptyPreferenceItemPoolError as exc:
        return OfferOutcome(
            offer_id,
            [],
            roster_entry,
            attrition,
            exclusion={
                "offer_id": offer_id,
                "reason": "empty_preference_item_pool",
                "detail": str(exc),
            },
        )

    expected = {
        (offer_id, box.name.casefold()): {
            "tier": box.tier,
            "dim_available": {
                dimension: len(tags)
                for dimension, tags in tags_for_preference(
                    box.preference, variants
                ).items()
            },
        }
        for box in selected_boxes
    }
    return OfferOutcome(
        offer_id,
        provisional_records,
        roster_entry,
        attrition,
        roster_contract_failures=selected_roster_contract_failures(
            provisional_records, expected
        ),
    )


def execute(
    offer_ids: list[int],
    process_one: Callable[[int], OfferOutcome],
    *,
    out_path: Path,
    report_path: Path,
    requested_offer_ids: list[int],
) -> int:
    """Process offers sequentially and aggregate their DB-free outcomes."""
    records: list[dict] = []
    roster_entries: list[dict] = []
    exclusions: list[dict] = []
    errors: list[dict] = []
    roster_contract_failures: list[dict] = []
    eligible_offers = 0
    excluded_offers = 0
    aggregate_attrition = empty_offer_attrition()

    for offer_id in offer_ids:
        try:
            outcome = process_one(offer_id)
        except Exception as exc:
            errors.append({
                "offer_id": offer_id,
                "exception": type(exc).__name__,
                "message": str(exc),
            })
            continue

        if outcome.error is not None:
            errors.append(outcome.error)
            continue

        roster_entries.append(outcome.roster_entry)
        for counter_name in ("csv", "db", "selected"):
            aggregate_attrition["roster_candidates"][counter_name] += (
                outcome.attrition["roster_candidates"].get(counter_name, 0)
            )
        for status, count in outcome.attrition["solver_statuses"].items():
            aggregate_attrition["solver_statuses"][status] = (
                aggregate_attrition["solver_statuses"].get(status, 0) + count
            )
        for source, reasons in outcome.attrition["row_attrition"].items():
            aggregate_reasons = aggregate_attrition["row_attrition"].setdefault(
                source, {"empty": 0, "unextractable": 0}
            )
            for reason, count in reasons.items():
                aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + count

        roster_contract_failures.extend(outcome.roster_contract_failures)
        if outcome.exclusion is not None:
            excluded_offers += 1
            exclusions.append(outcome.exclusion)
        else:
            eligible_offers += 1
            records.extend(outcome.records)

    roster_check = {
        "offers": sorted(roster_entries, key=lambda entry: entry["offer_id"]),
        "totals": {
            "csv_only": sum(len(entry["csv_only"]) for entry in roster_entries),
            "db_only": sum(len(entry["db_only"]) for entry in roster_entries),
            "selected": sum(entry["selected_count"] for entry in roster_entries),
        },
    }
    attrition = {
        "requested_offers": len(requested_offer_ids),
        "resolved_offers": len(offer_ids),
        "eligible_offers": eligible_offers,
        "excluded_offers": excluded_offers,
        "roster_candidates": aggregate_attrition["roster_candidates"],
        "solver_statuses": aggregate_attrition["solver_statuses"],
        "row_attrition": aggregate_attrition["row_attrition"],
        "rung_coverage": {
            rung: paired_rung_coverage(records, selector)
            for rung, selector in RUNG_NEGATIVE_SELECTORS.items()
        },
    }
    return finalize_run(
        records,
        requested_offer_ids,
        list(offer_ids),
        roster_check,
        attrition,
        exclusions,
        errors,
        roster_contract_failures,
        out_path,
        report_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the deliberately small sequential-generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate reconciled hard-negative feature data"
    )
    parser.add_argument(
        "--only-offers",
        help="comma-separated IDs/ranges; default all available Tier-A offers",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("diagnostics/hard_negatives.json"),
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("diagnostics/hard_negatives_report.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the Tier-A audit trail and run the generator sequentially."""
    import compare

    parser = build_parser()
    args = parser.parse_args(argv)
    available = compare._discover_cleaned_offer_ids()
    try:
        if args.only_offers is None:
            requested_offer_ids = default_tier_a_offer_ids(available)
            offer_ids = list(requested_offer_ids)
        else:
            requested_offer_ids = parse_requested_tier_a_offer_ids(
                args.only_offers
            )
            offer_ids = resolve_requested_offer_ids(
                requested_offer_ids, available
            )
    except ValueError as exc:
        parser.error(str(exc))

    return execute(
        offer_ids,
        process_offer,
        out_path=args.out,
        report_path=args.report_out,
        requested_offer_ids=requested_offer_ids,
    )


if __name__ == "__main__":
    raise SystemExit(main())
