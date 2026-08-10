"""Unmarked hard-negative data contracts."""

import dataclasses


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
