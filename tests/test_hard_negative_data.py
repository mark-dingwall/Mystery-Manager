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
