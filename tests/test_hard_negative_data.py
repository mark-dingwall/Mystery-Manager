"""Unmarked hard-negative data contracts."""

import dataclasses
import os
from pathlib import Path
import shutil
import subprocess
import sys


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
        roster, "PER_OFFER_BOX_SIZE_OVERRIDES", {"80": {"z@example.com": "small"}}
    )
    original = [
        make_box(name="a@example.com", tier="small"),
        make_box(name="Z@Example.COM", tier="large"),
    ]
    corrected = roster.correct_box_tiers(80, original)

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
        "80": {"case@example.com": "small", "CASE@example.com": "large"},
    })
    with pytest.raises(roster.AmbiguousRosterIdentityError, match="case-normalised"):
        roster.correct_box_tiers(80, [make_box(name="case@example.com", tier="medium")])


def test_roster_hash_uses_shared_feature_digest(monkeypatch):
    import allocator.hard_negative_roster as roster
    from allocator.box_features import stable_hash

    mapping = {
        "80": {"a@example.com": "small"},
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
