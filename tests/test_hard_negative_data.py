"""Unmarked hard-negative data contracts."""

import dataclasses
import os
from pathlib import Path
import shutil
import subprocess
import sys


_SYNTHETIC_OFFER_ID = 910_001


def test_allocation_result_defaults_solver_status_to_none(make_result):
    assert make_result().solver_status is None


def test_solver_status_is_final_defaulted_field():
    from allocator.models import AllocationResult

    fields = dataclasses.fields(AllocationResult)
    assert fields[-1].name == "solver_status"
    assert fields[-1].default is None


def test_solution_status_uses_pulp_solution_not_lp_status(make_result):
    from allocator.strategies.ilp_optimal import _record_solution_status

    class FakePulp:
        LpSolution = {7: "Solution Found"}

    class FakeProblem:
        sol_status = 7

    result = make_result()
    _record_solution_status(result, FakePulp, FakeProblem())
    assert result.solver_status == "Solution Found"


def test_solver_exception_preserves_nonoptimal_but_clears_optimal(make_result):
    from allocator.strategies.ilp_optimal import _record_fallback_solver_error

    result = make_result()
    for status, expected in (
        (None, "FallbackSolverError"),
        ("Optimal Solution Found", "FallbackSolverError"),
        ("Solution Found", "Solution Found"),
        ("No Solution Exists", "No Solution Exists"),
    ):
        result.solver_status = status
        _record_fallback_solver_error(result)
        assert result.solver_status == expected


def test_import_fallback_records_its_distinct_provenance(monkeypatch, make_result):
    import builtins

    import allocator.strategies.ilp_optimal as ilp

    real_import = builtins.__import__

    def no_pulp(name, *args, **kwargs):
        if name == "pulp":
            raise ImportError("synthetic missing PuLP")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pulp)
    result = make_result()
    ilp.run(result)
    assert result.solver_status == "FallbackImportError"


def test_ilp_no_work_keeps_the_existing_early_return(monkeypatch, make_result):
    import allocator.strategies.ilp_optimal as ilp

    result = make_result()
    result.boxes = []
    called = []
    monkeypatch.setattr(ilp, "_solve_ilp", lambda *_args: called.append(True))
    ilp.run(result)
    assert called == []


def test_correct_box_tiers_copies_refreshes_and_resorts(make_box, monkeypatch):
    import allocator.hard_negative_roster as roster
    from allocator.config import BOX_TIERS

    monkeypatch.setattr(
        roster,
        "PER_OFFER_BOX_SIZE_OVERRIDES",
        {str(_SYNTHETIC_OFFER_ID): {"z@example.com": "small"}},
    )
    original = [
        make_box(name="a@example.com", tier="small"),
        make_box(name="Z@Example.COM", tier="large"),
    ]
    corrected = roster.correct_box_tiers(_SYNTHETIC_OFFER_ID, original)

    assert original[1].tier == "large"
    assert [(box.name, box.tier) for box in corrected] == [
        ("a@example.com", "small"), ("Z@Example.COM", "small"),
    ]
    assert corrected[1].target_value == BOX_TIERS["small"]["target_value"]


def test_intersect_roster_retains_spelling_and_sorts_casefolded_identities():
    from allocator.hard_negative_roster import RosterMatch, intersect_roster

    intersection = intersect_roster(
        ["z@example.com", "C-only", "a@example.com", "B-only"],
        ["d-only", "A@example.com", "Z@example.com"],
    )
    assert intersection.matches == (
        RosterMatch(csv_name="a@example.com", db_name="A@example.com"),
        RosterMatch(csv_name="z@example.com", db_name="Z@example.com"),
    )
    assert intersection.csv_only == ("B-only", "C-only")
    assert intersection.db_only == ("d-only",)


def test_intersect_roster_rejects_casefold_collisions():
    import pytest
    from allocator.hard_negative_roster import (
        AmbiguousRosterIdentityError,
        intersect_roster,
    )

    with pytest.raises(AmbiguousRosterIdentityError, match="case-normalised"):
        intersect_roster(["Case@Example.com", "case@example.com"], [])


def test_correct_box_tiers_rejects_casefolded_override_collisions(make_box, monkeypatch):
    import pytest
    import allocator.hard_negative_roster as roster

    monkeypatch.setattr(roster, "PER_OFFER_BOX_SIZE_OVERRIDES", {
        str(_SYNTHETIC_OFFER_ID): {
            "case@example.com": "small", "CASE@example.com": "large"
        },
    })
    with pytest.raises(roster.AmbiguousRosterIdentityError, match="case-normalised"):
        roster.correct_box_tiers(
            _SYNTHETIC_OFFER_ID,
            [make_box(name="case@example.com", tier="medium")],
        )


def test_correct_box_tiers_rejects_invalid_applicable_override(make_box, monkeypatch):
    import pytest
    import allocator.hard_negative_roster as roster

    monkeypatch.setattr(
        roster,
        "PER_OFFER_BOX_SIZE_OVERRIDES",
        {str(_SYNTHETIC_OFFER_ID): {"box@example.com": "tiny"}},
    )
    with pytest.raises(ValueError, match="invalid tier override"):
        roster.correct_box_tiers(
            _SYNTHETIC_OFFER_ID,
            [make_box(name="BOX@example.com", tier="medium")],
        )


def test_roster_hash_uses_shared_feature_digest(monkeypatch):
    import allocator.hard_negative_roster as roster
    from allocator.box_features import stable_hash

    mapping = {
        str(_SYNTHETIC_OFFER_ID): {"a@example.com": "small"},
        "110": {"not-selected@example.com": "medium"},
    }
    monkeypatch.setattr(roster, "PER_OFFER_BOX_SIZE_OVERRIDES", mapping)
    assert roster.roster_config_hash() == stable_hash(mapping)


def test_hard_negative_roster_imports_from_isolated_root_without_db(tmp_path):
    """The roster contract must not reach DB or historical-data modules."""
    project_root = Path(__file__).resolve().parent.parent
    isolated_root = tmp_path / "isolated-project"
    shutil.copytree(
        project_root / "allocator",
        isolated_root / "allocator",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        project_root / "tests" / "fixtures" / "scoring_config.json",
        isolated_root / "scoring_config.json",
    )
    shutil.copy2(
        project_root / "tests" / "fixtures" / "identifiers.json",
        isolated_root / "identifiers.json",
    )

    env = os.environ.copy()
    env.update({
        "BOX_PRICE_SMALL": "1733",
        "BOX_PRICE_MEDIUM": "2867",
        "BOX_PRICE_LARGE": "4099",
        "BOX_TARGET_PCT": "108",
        "VALUE_SWEET_FROM": "101",
        "VALUE_SWEET_TO": "106",
        "VALUE_PENALTY_EXPONENT": "1.6",
        "PYTHONPATH": str(isolated_root),
    })
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name in ('allocator.db', 'compare', 'scripts.extract_features'):\n"
        "            raise AssertionError('hard_negative_roster imported ' + name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import allocator.hard_negative_roster as roster\n"
        "assert Path(roster.__file__).resolve().is_relative_to(\n"
        "    Path(sys.argv[1]).resolve())\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(isolated_root)],
        cwd=isolated_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_selected_synthetic_template_keeps_box_tier_and_tag_denominator():
    from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES
    from scripts.extract_features import SyntheticTemplate, generate_synthetic_boxes

    lookup = {
        1: {"name": "Apple", "price": 100, "category_id": CATEGORY_FRUIT, "fungible_group": "apple",
            "fungible_degree": 0.7, "sub_category": "pome", "usage": "snacking",
            "colour": "red", "shape": "round", "size": 1},
        2: {"name": "Carrot", "price": 100, "category_id": CATEGORY_VEGETABLES, "fungible_group": None,
            "fungible_degree": 0.0, "sub_category": "root", "usage": "cooking",
            "colour": "orange", "shape": "long", "size": 1},
    }
    tags = {"sub_category": {"pome"}, "usage": {"snacking"},
            "colour": {"red"}, "shape": {"round"}}
    template = SyntheticTemplate("fruit@example.com", "small", "fruit_only", tags)

    records = generate_synthetic_boxes(80, lookup, tags, templates=[template])
    assert len(records) == 5
    assert {(record["box_name"], record["tier"]) for record in records} == {
        ("fruit@example.com", "small"),
    }
    assert all(record["dim_available"]["sub_category"] == 1 for record in records)
    assert all(record["pref_violations"] == 0 for record in records)
    assert all(record["category_value_share"]["veg"] == 0.0 for record in records)


def test_synthetic_recipe_is_absent_when_its_required_fungible_group_is_missing():
    from random import Random
    from scripts.extract_features import _synthetic_allocations

    lookup = {1: {"price": 100, "fungible_group": None}}
    recipes = _synthetic_allocations(lookup, "small", Random(1))
    assert {source for source, _fragment, _allocations in recipes} == {
        "synth_monoculture", "synth_random", "synth_value_low", "synth_value_high",
    }


def test_synthetic_recipes_ignore_zero_priced_items():
    from random import Random

    from scripts.extract_features import _synthetic_allocations

    recipes = _synthetic_allocations(
        {
            1: {"price": 0, "fungible_group": "free"},
            2: {"price": 100, "fungible_group": "paid"},
        },
        "small",
        Random(1),
    )
    assert recipes
    assert all(1 not in allocations for _source, _fragment, allocations in recipes)


def test_selected_synthetic_template_rejects_free_only_pool():
    import pytest
    from allocator.config import CATEGORY_FRUIT
    from scripts.extract_features import (
        EmptyPreferenceItemPoolError, SyntheticTemplate, generate_synthetic_boxes,
    )

    lookup = {1: {"price": 0, "category_id": CATEGORY_FRUIT}}
    template = SyntheticTemplate("fruit@example.com", "small", "fruit_only", {})
    with pytest.raises(EmptyPreferenceItemPoolError, match="positive-priced"):
        generate_synthetic_boxes(80, lookup, {}, templates=[template])


def test_selected_synthetic_template_rejects_an_empty_preference_pool():
    import pytest
    from allocator.config import CATEGORY_VEGETABLES
    from scripts.extract_features import (
        EmptyPreferenceItemPoolError, SyntheticTemplate, generate_synthetic_boxes,
    )

    lookup = {1: {"price": 100, "category_id": CATEGORY_VEGETABLES}}
    template = SyntheticTemplate("fruit@example.com", "small", "fruit_only", {})
    with pytest.raises(EmptyPreferenceItemPoolError, match="fruit@example.com"):
        generate_synthetic_boxes(80, lookup, {}, templates=[template])


def test_synthetic_recipe_helper_preserves_the_empty_item_guard():
    from random import Random
    from scripts.extract_features import _synthetic_allocations

    assert _synthetic_allocations({}, "small", Random(1)) == []


def test_template_synthetics_propagate_unsupported_categories_but_legacy_skips(monkeypatch):
    import pytest
    import scripts.extract_features as extractor

    def raise_unsupported(*_args, **_kwargs):
        raise extractor.UnsupportedCategoryError("test category")

    monkeypatch.setattr(extractor, "extract_box_features", raise_unsupported)
    lookup = {1: {"price": 100}}
    assert extractor.generate_synthetic_boxes(80, lookup, {}) == []
    template = extractor.SyntheticTemplate("box@example.com", "small", None, {})
    with pytest.raises(extractor.UnsupportedCategoryError, match="test category"):
        extractor.generate_synthetic_boxes(80, lookup, {}, templates=[template])


def _gate_records(offers=20, manual_per_offer=8):
    records = []
    for offer_id in range(64, 64 + offers):
        for index in range(manual_per_offer):
            tier = ("small", "medium", "large")[index % 3]
            records.extend([
                {"offer_id": offer_id, "box_name": f"m{index}@{offer_id}", "tier": tier, "source": "manual"},
                {"offer_id": offer_id, "box_name": f"s{index}@{offer_id}", "tier": tier, "source": "synth_random"},
                {"offer_id": offer_id, "box_name": f"b{index}@{offer_id}", "tier": tier, "source": "baseline_deal_topup"},
                {"offer_id": offer_id, "box_name": f"i{index}@{offer_id}", "tier": tier,
                 "source": "ilp_optimal", "solver_status": "Optimal Solution Found"},
            ])
    return records


def test_paired_coverage_drops_unmatched_manual_cells():
    from scripts.generate_hard_negatives import paired_rung_coverage

    records = [
        record for record in _gate_records()
        if record["source"] != "synth_random" or record["offer_id"] != 64
    ]
    assert paired_rung_coverage(records, "synth_") == {
        "manual_boxes": 152, "offers": 19,
    }


def test_preference_tags_and_ilp_admission_are_exact():
    from scripts.generate_hard_negatives import admits_to_ilp_class, tags_for_preference

    variants = {
        "all": {"marker": {"all"}},
        "fruit_only": {"marker": {"fruit"}},
        "veg_only": {"marker": {"veg"}},
    }
    assert tags_for_preference(None, variants) is variants["all"]
    assert tags_for_preference("fruit_only", variants) is variants["fruit_only"]
    assert tags_for_preference("veg_only", variants) is variants["veg_only"]
    assert tags_for_preference("unrecognised", variants) is variants["all"]
    assert admits_to_ilp_class("Optimal Solution Found")
    assert not admits_to_ilp_class("Solution Found")
    assert not admits_to_ilp_class("FallbackSolverError")
    assert not admits_to_ilp_class(None)


def test_selected_roster_contract_rejects_an_unselected_or_wrong_denominator_row():
    from scripts.generate_hard_negatives import (
        selected_roster_contract_failures,
        validation_failures,
    )

    expected = {
        (64, "person@example.com"): {
            "tier": "small",
            "dim_available": {"sub_category": 1, "usage": 2, "colour": 3, "shape": 4},
        },
    }
    valid = {
        "offer_id": 64, "box_name": "Person@Example.com", "tier": "small",
        "source": "manual",
        "dim_available": {"sub_category": 1, "usage": 2, "colour": 3, "shape": 4},
    }
    assert selected_roster_contract_failures([valid], expected) == []

    bad = {**valid, "box_name": "not-selected@example.com"}
    wrong_denominator = {
        **valid, "dim_available": {"sub_category": 0, "usage": 2, "colour": 3, "shape": 4},
    }
    failures = selected_roster_contract_failures([bad, wrong_denominator], expected)
    assert {failure["gate"] for failure in failures} == {"selected_roster_contract"}

    assert any(
        failure["gate"] == "selected_roster_contract"
        for failure in validation_failures([], {}, failures)
    )


def test_unextracted_row_reason_distinguishes_empty_from_unknown_positive_ids():
    from scripts.generate_hard_negatives import unextracted_row_reason

    assert unextracted_row_reason({}, {1: object()}) == "empty"
    assert unextracted_row_reason({1: 0}, {1: object()}) == "empty"
    assert unextracted_row_reason({999: 1}, {1: object()}) == "unextractable"


def test_validation_rejects_nonoptimal_ilp_and_missing_baseline_source():
    from scripts.generate_hard_negatives import validation_failures

    records = _gate_records()
    records[3]["solver_status"] = "Solution Found"
    failures = validation_failures(records, {
        "manual": 160, "synth_random": 160,
        "baseline_deal_topup": 160, "baseline_minmax_deficit": 160,
        "ilp_optimal": 160,
    })
    gates = {failure["gate"] for failure in failures}
    assert "ilp_optimal.status" in gates
    assert "required_sources" in gates


def test_build_artifact_has_exact_stamped_shape():
    from collections import Counter

    from allocator.box_features import FEATURE_SCHEMA_VERSION, config_hash
    from allocator.hard_negative_roster import roster_config_hash
    from scripts.generate_hard_negatives import (
        GENERATOR_VERSION, build_artifact, validation_failures,
    )

    records = _gate_records()
    baseline_rows = [r for r in records if r["source"] == "baseline_deal_topup"]
    for source in ("baseline_minmax_deficit", "baseline_greedy_best_fit"):
        records.extend({**row, "source": source} for row in baseline_rows)
    # These are deliberately valid negative rows; gates are paired-cell coverage,
    # not a quality/fill proxy.
    for record in records:
        if record["source"].startswith(("baseline_", "synth_")):
            record["value_pct"] = 0.0
    source_counts = dict(Counter(r["source"] for r in records))
    assert validation_failures(records, source_counts) == []
    artifact = build_artifact(
        records, source_counts,
        {"offers": [], "totals": {}}, {}, [], list(range(64, 84)), list(range(64, 84)),
    )
    assert list(artifact) == [
        "feature_schema_version", "config_hash", "roster_config_hash", "records",
        "source_counts", "roster_check", "attrition", "exclusions", "run_metadata",
    ]
    assert artifact["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert artifact["config_hash"] == config_hash()
    assert artifact["roster_config_hash"] == roster_config_hash()
    assert artifact["run_metadata"]["generator_version"] == GENERATOR_VERSION
    assert artifact["records"] == sorted(
        records,
        key=lambda r: (
            r["offer_id"], r["tier"], r["source"], r["box_name"],
            r.get("solver_status", ""),
        ),
    )


def test_failed_run_preserves_existing_artifact_and_writes_report(tmp_path):
    from scripts.generate_hard_negatives import finalize_run

    out = tmp_path / "hard_negatives.json"
    report = tmp_path / "hard_negatives_report.json"
    out.write_text('{"prior": true}\n')

    assert finalize_run(
        records=[], requested_offer_ids=[64], resolved_offer_ids=[64],
        roster_check={"offers": [], "totals": {}}, attrition={},
        exclusions=[], errors=[], roster_contract_failures=[], out_path=out, report_path=report,
    ) == 1
    assert out.read_text() == '{"prior": true}\n'
    assert __import__("json").loads(report.read_text())["status"] == "validation_failed"


def test_execution_error_short_circuits_success_artifact_build(tmp_path):
    import json

    from scripts.generate_hard_negatives import finalize_run

    out = tmp_path / "hard_negatives.json"
    report = tmp_path / "hard_negatives_report.json"
    out.write_text('{"prior": true}\n')

    assert finalize_run(
        records=[{"source": "manual"}],
        requested_offer_ids=[64], resolved_offer_ids=[64],
        roster_check={"offers": [], "totals": {}}, attrition={}, exclusions=[],
        errors=[{"offer_id": 64, "type": "RuntimeError", "message": "boom"}],
        roster_contract_failures=[], out_path=out, report_path=report,
    ) == 1
    assert out.read_text() == '{"prior": true}\n'
    payload = json.loads(report.read_text())
    assert payload["status"] == "execution_failed"
    assert payload["failed_gates"] == []
    assert payload["source_counts"] == {"manual": 1}


def test_execution_error_report_omits_unlabeled_partial_records(tmp_path):
    import json

    from scripts.generate_hard_negatives import finalize_run

    report = tmp_path / "hard_negatives_report.json"
    assert finalize_run(
        records=[{}], requested_offer_ids=[64], resolved_offer_ids=[64],
        roster_check={"offers": [], "totals": {}}, attrition={}, exclusions=[],
        errors=[{"offer_id": 64, "type": "RuntimeError", "message": "boom"}],
        roster_contract_failures=[],
        out_path=tmp_path / "hard_negatives.json", report_path=report,
    ) == 1

    payload = json.loads(report.read_text())
    assert set(payload) == {
        "status", "failed_gates", "source_counts", "roster_check", "attrition",
        "run_metadata", "exclusions", "errors",
    }
    assert payload["status"] == "execution_failed"
    assert payload["source_counts"] == {}


def test_finalize_run_rejects_aliased_artifact_and_report_paths(tmp_path):
    import pytest

    from scripts.generate_hard_negatives import finalize_run

    destination = tmp_path / "hard_negatives.json"
    destination.write_text('{"prior": true}\n')

    with pytest.raises(ValueError, match="distinct"):
        finalize_run(
            records=[], requested_offer_ids=[64], resolved_offer_ids=[64],
            roster_check={"offers": [], "totals": {}}, attrition={},
            exclusions=[], errors=[], roster_contract_failures=[],
            out_path=destination, report_path=destination,
        )
    assert destination.read_text() == '{"prior": true}\n'


def test_requested_offer_range_validates_endpoint_before_expansion(monkeypatch):
    import pytest
    import scripts.generate_hard_negatives as generator

    real_range = range

    def reject_large_range(start, stop):
        if stop - start > 1_000:
            raise AssertionError("large range expanded before validation")
        return real_range(start, stop)

    monkeypatch.setattr(generator, "range", reject_large_range, raising=False)
    with pytest.raises(ValueError, match="Tier-A"):
        generator.parse_requested_tier_a_offer_ids("64-1000000000")


def test_requested_and_resolved_tier_a_offers_remain_distinct():
    import pytest
    from scripts.generate_hard_negatives import (
        default_tier_a_offer_ids,
        parse_requested_tier_a_offer_ids,
        resolve_requested_offer_ids,
    )

    requested = parse_requested_tier_a_offer_ids("64,66-67")
    assert requested == [64, 66, 67]
    assert resolve_requested_offer_ids(requested, {64, 65, 67}) == [64, 67]
    assert default_tier_a_offer_ids({63, 64, 109, 110}) == [64, 109]
    with pytest.raises(ValueError, match="Tier-A"):
        parse_requested_tier_a_offer_ids("63")
    with pytest.raises(ValueError, match="non-empty"):
        resolve_requested_offer_ids([64], {65})
    with pytest.raises(ValueError, match="resolved Tier-A offer IDs must be non-empty"):
        default_tier_a_offer_ids({63, 110})


def test_execute_discards_every_source_for_nonoptimal_offer(tmp_path):
    from scripts.generate_hard_negatives import OfferOutcome, execute

    outcome = OfferOutcome(
        offer_id=64,
        records=[],
        roster_entry={
            "offer_id": 64,
            "csv_only": [],
            "db_only": [],
            "selected_count": 2,
        },
        attrition={
            "roster_candidates": {"csv": 2, "db": 2, "selected": 2},
            "solver_statuses": {"Solution Found": 1},
            "row_attrition": {},
        },
        exclusion={
            "offer_id": 64,
            "reason": "nonoptimal_ilp",
            "detail": "Solution Found",
        },
    )
    code = execute(
        [64],
        lambda _offer_id: outcome,
        out_path=tmp_path / "hard_negatives.json",
        report_path=tmp_path / "hard_negatives_report.json",
        requested_offer_ids=[64],
    )
    report = __import__("json").loads(
        (tmp_path / "hard_negatives_report.json").read_text()
    )
    assert code == 1
    assert report["exclusions"] == [outcome.exclusion]
    assert report["source_counts"] == {}


def test_execute_reports_an_unexpected_error_without_classifying_that_offer(tmp_path):
    from scripts.generate_hard_negatives import execute

    def process_one(_offer_id):
        raise RuntimeError("test failure")

    assert execute(
        [64],
        process_one,
        out_path=tmp_path / "hard_negatives.json",
        report_path=tmp_path / "hard_negatives_report.json",
        requested_offer_ids=[64],
    ) == 1
    report = __import__("json").loads(
        (tmp_path / "hard_negatives_report.json").read_text()
    )
    assert report["status"] == "execution_failed"
    assert report["errors"] == [
        {"offer_id": 64, "exception": "RuntimeError", "message": "test failure"},
    ]
    assert report["attrition"]["resolved_offers"] == 1
    assert report["attrition"]["eligible_offers"] == 0
    assert report["attrition"]["excluded_offers"] == 0


def test_execute_aggregates_the_pinned_attrition_contract(tmp_path):
    from scripts.generate_hard_negatives import OfferOutcome, execute

    outcomes = {
        64: OfferOutcome(
            offer_id=64,
            records=[],
            roster_entry={
                "offer_id": 64,
                "csv_only": [],
                "db_only": [],
                "selected_count": 2,
            },
            attrition={
                "roster_candidates": {"csv": 3, "db": 2, "selected": 2},
                "solver_statuses": {"Solution Found": 1},
                "row_attrition": {"manual": {"empty": 1, "unextractable": 2}},
            },
            exclusion={
                "offer_id": 64,
                "reason": "nonoptimal_ilp",
                "detail": "Solution Found",
            },
        ),
        65: OfferOutcome(
            offer_id=65,
            records=[],
            roster_entry={
                "offer_id": 65,
                "csv_only": [],
                "db_only": [],
                "selected_count": 3,
            },
            attrition={
                "roster_candidates": {"csv": 4, "db": 3, "selected": 3},
                "solver_statuses": {"No Solution Exists": 1},
                "row_attrition": {
                    "manual": {"empty": 2, "unextractable": 1},
                    "synth_random": {"empty": 1, "unextractable": 0},
                },
            },
            exclusion={
                "offer_id": 65,
                "reason": "nonoptimal_ilp",
                "detail": "No Solution Exists",
            },
        ),
    }
    execute(
        [64, 65],
        lambda offer_id: outcomes[offer_id],
        out_path=tmp_path / "hard_negatives.json",
        report_path=tmp_path / "hard_negatives_report.json",
        requested_offer_ids=[64, 65],
    )
    report = __import__("json").loads(
        (tmp_path / "hard_negatives_report.json").read_text()
    )
    assert report["attrition"] == {
        "requested_offers": 2,
        "resolved_offers": 2,
        "eligible_offers": 0,
        "excluded_offers": 2,
        "roster_candidates": {"csv": 7, "db": 5, "selected": 5},
        "solver_statuses": {"Solution Found": 1, "No Solution Exists": 1},
        "row_attrition": {
            "manual": {"empty": 3, "unextractable": 3},
            "synth_random": {"empty": 1, "unextractable": 0},
        },
        "rung_coverage": {
            "manual_vs_synth": {"manual_boxes": 0, "offers": 0},
            "manual_vs_baseline": {"manual_boxes": 0, "offers": 0},
            "manual_vs_ilp": {"manual_boxes": 0, "offers": 0},
        },
    }


def test_empty_roster_entry_keeps_early_exclusions_aggregateable():
    from scripts.generate_hard_negatives import empty_roster_entry

    assert empty_roster_entry(64) == {
        "offer_id": 64,
        "csv_only": [],
        "db_only": [],
        "selected_count": 0,
    }


def test_generator_module_import_does_not_import_compare_or_db():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import scripts.generate_hard_negatives; "
                "assert 'compare' not in sys.modules; "
                "assert 'allocator.db' not in sys.modules"
            ),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_generator_cli_help_runs_from_project_root():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "scripts/generate_hard_negatives.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--only-offers" in proc.stdout
    assert "--out" in proc.stdout
    assert "--report-out" in proc.stdout


def test_generator_parser_exposes_only_mvp_options():
    from scripts.generate_hard_negatives import build_parser

    parser = build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {"--only-offers", "--out", "--report-out"} <= options
    assert "--workers" not in options


def test_generator_defaults_use_repository_diagnostics_directory():
    from scripts.generate_hard_negatives import build_parser

    root = Path(__file__).resolve().parent.parent
    args = build_parser().parse_args([])
    assert args.out == root / "diagnostics" / "hard_negatives.json"
    assert args.report_out == root / "diagnostics" / "hard_negatives_report.json"


def test_process_offer_uses_only_the_selected_customer_roster(
    monkeypatch, make_box, tmp_path
):
    from types import SimpleNamespace

    import allocator.allocator as allocator_module
    import allocator.box_features as box_features
    from allocator.config import (
        CATEGORY_FRUIT,
        CATEGORY_VEGETABLES,
        DONATION_IDENTIFIERS,
        SKIP_COLUMN_IDENTIFIERS,
        STAFF_IDENTIFIERS,
    )
    import compare
    import scripts.generate_hard_negatives as hard_negatives

    csv_name, db_name = "fruit@example.com", "FRUIT@example.com"
    filtered_names = [
        next(iter(DONATION_IDENTIFIERS)),
        next(iter(SKIP_COLUMN_IDENTIFIERS)),
        next(iter(STAFF_IDENTIFIERS)),
    ]
    raw_csv_names = [csv_name, *filtered_names]
    lookup = {
        1: {
            "name": "Apple",
            "price": 100,
            "category_id": CATEGORY_FRUIT,
            "fungible_group": "apple",
            "fungible_degree": 0.7,
            "sub_category": "pome",
            "usage": "snacking",
            "colour": "red",
            "shape": "round",
            "size": 1,
        },
        2: {
            "name": "Carrot",
            "price": 100,
            "category_id": CATEGORY_VEGETABLES,
            "fungible_group": None,
            "fungible_degree": 0.0,
            "sub_category": "root",
            "usage": "cooking",
            "colour": "orange",
            "shape": "long",
            "size": 1,
        },
    }
    monkeypatch.setattr(
        compare, "_find_xlsx_path", lambda _offer_id: tmp_path / "offer.xlsx"
    )
    monkeypatch.setattr(compare, "build_item_lookup", lambda _offer_id: lookup)
    monkeypatch.setattr(
        compare,
        "load_historical_csv",
        lambda _offer_id: (raw_csv_names, {1: {csv_name: 1.0}}),
    )
    monkeypatch.setattr(
        allocator_module,
        "build_boxes_from_db",
        lambda _offer_id: [
            make_box(name=db_name, tier="small", preference="fruit_only")
        ],
    )
    strategies, manual_inputs = [], []
    real_extract = box_features.extract_box_features

    def fake_allocate(_offer_id, _xlsx_path, *, boxes, strategy, **_kwargs):
        strategies.append(strategy)
        boxes[0].allocations = {1: 1}
        return SimpleNamespace(
            boxes=boxes, solver_status="Optimal Solution Found"
        )

    def spy_extract(box_name, allocations, *args, source="manual", **kwargs):
        if source == "manual":
            manual_inputs.append((box_name, allocations))
        return real_extract(
            box_name, allocations, *args, source=source, **kwargs
        )

    monkeypatch.setattr(allocator_module, "allocate", fake_allocate)
    monkeypatch.setattr(box_features, "extract_box_features", spy_extract)
    outcome = hard_negatives.process_offer(_SYNTHETIC_OFFER_ID)

    assert strategies == [
        "ilp-optimal",
        "deal-topup",
        "minmax-deficit",
        "greedy-best-fit",
    ]
    assert manual_inputs == [(db_name, {1: 1})]
    assert type(manual_inputs[0][1][1]) is int
    assert outcome.exclusion is None
    assert outcome.error is None
    assert outcome.roster_entry == {
        "offer_id": _SYNTHETIC_OFFER_ID,
        "csv_only": [],
        "db_only": [],
        "selected_count": 1,
    }
    assert not set(filtered_names) & set(outcome.roster_entry["csv_only"])
    assert outcome.roster_contract_failures == []
    assert outcome.records
    assert {record["box_name"] for record in outcome.records} == {db_name}
    assert {record["tier"] for record in outcome.records} == {"small"}
    assert all(
        record["dim_available"]
        == {"sub_category": 1, "usage": 1, "colour": 1, "shape": 1}
        for record in outcome.records
    )
    assert {
        record["solver_status"]
        for record in outcome.records
        if record["source"] == "ilp_optimal"
    } == {"Optimal Solution Found"}
    assert outcome.attrition == {
        "roster_candidates": {"csv": 1, "db": 1, "selected": 1},
        "solver_statuses": {"Optimal Solution Found": 1},
        "row_attrition": {},
    }

    report_path = tmp_path / "hard_negatives_report.json"
    assert hard_negatives.execute(
        [80],
        lambda _offer_id: outcome,
        out_path=tmp_path / "hard_negatives.json",
        report_path=report_path,
        requested_offer_ids=[80],
    ) == 1
    report = __import__("json").loads(report_path.read_text())
    assert report["status"] == "validation_failed"
    assert report["roster_check"] == {
        "offers": [outcome.roster_entry],
        "totals": {"csv_only": 0, "db_only": 0, "selected": 1},
    }
    assert report["attrition"]["roster_candidates"] == {
        "csv": 1,
        "db": 1,
        "selected": 1,
    }
    assert report["attrition"]["solver_statuses"] == {
        "Optimal Solution Found": 1,
    }
    assert report["attrition"]["row_attrition"] == {}
