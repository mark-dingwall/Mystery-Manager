# Hard-Negative Data Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Generate a reproducible, hard-gated Tier-A diagnostics/hard_negatives.json that is safe input for the later three-rung EBM diagnostic.

**Architecture:** Solver provenance and shared feature identity form the lowest layer. A new DB-free roster module reconciles only the email population common to CSV and DB sources. The new sequential generator is the only DB-backed entry point; it rebuilds manual, baseline, ILP, and synthetic rows from that common roster, validates all three rungs, and writes either an atomic usable artifact or an audit-only report.

**Tech Stack:** Python 3.10, pytest, stdlib JSON/hashlib/tempfile, existing allocator strategies, PuLP/HiGHS when installed. No new dependencies.

**Source spec:** docs/superpowers/specs/2026-08-10-hard-negative-data-design.md

## Global Constraints

- This is PR #2 only. Do not import InterpretML, fit an EBM, add ordinal regression, alter scoring, or change allocator strategy selection.
- The generator supports Tier-A offer IDs 64–109 only. --only-offers accepts existing comma/range syntax but rejects an out-of-range ID and must resolve a non-empty available subset.
- The generator is deliberately sequential. Do not add --workers, ProcessPoolExecutor, or ThreadPoolExecutor; one process reuses the existing process-scoped DB tunnel.
- New allocator modules must be DB-free and have no import-time side effects. allocator.hard_negative_roster must not import compare, allocator.db, or either script.
- Preserve config_hash()'s thirteen input contract. Promote its existing digest behavior to public stable_hash() and call it from roster_config_hash(); do not create a second JSON/hash recipe.
- Export FEATURE_SCHEMA_VERSION = 2. Rebuild manual rows with plain DB prices, never XLSX pack-price overrides.
- A non-empty CSV-only/DB-only roster difference is audit-only. Missing XLSX/historical CSV/item lookup, empty roster intersection, non-optimal ILP, or UnsupportedCategoryError excludes the whole offer from every family.
- A duplicate CSV or DB identity after case-normalisation is ambiguous, never silently de-duplicated, and excludes that whole offer as `ambiguous_roster_identity`.
- Only the solution status "Optimal Solution Found" admits an ILP row. "Solution Found" and both fallback labels are non-optimal.
- Do not impose a fill floor on baseline or synthetic records. Empty/unextractable individual boxes reduce paired coverage instead.
- A normal artifact is all-or-nothing: each rung requires at least 150 paired manual rows across 20 offers. Failure never replaces --out and always writes an audit report.
- Tests use synthetic fixtures only; live DB work is operator validation after unit tests.

---

## File Structure

| File | Responsibility |
|---|---|
| allocator/models.py | Add the defaulted final AllocationResult.solver_status provenance field. |
| allocator/strategies/ilp_optimal.py | Record PuLP solution status and label fallback paths without losing meaningful non-optimal evidence. |
| allocator/box_features.py | Export FEATURE_SCHEMA_VERSION and public stable_hash() while retaining unchanged config-hash output. |
| allocator/hard_negative_roster.py | New DB-free tier correction, roster intersection, and roster config stamp. |
| scripts/extract_features.py | Refactor existing synthetic allocation construction behind a selected-box template without changing legacy no-template output. |
| scripts/generate_hard_negatives.py | New sequential DB entry point; owns per-offer assembly, gates, failure reports, and atomic writes. |
| tests/test_hard_negative_data.py | New unmarked test module for provenance, roster, template, validation, failure, and orchestration contracts. |
| tests/test_box_features.py | Extend public feature schema/hash and DB-free import tests. |
| tests/test_strategies.py | Extend ILP fallback regression with provenance assertions. |
| tests/test_tuning.py | Lock down legacy synthetic output and test template behavior. |
| CLAUDE.md | Document the normal and failure-report generator workflows. |

The generator may import compare and allocator.db only inside runtime functions. Importing scripts.generate_hard_negatives in a test must not require queries.json or open a tunnel.

---

### Task 1: Add stable feature identity and ILP solver provenance

**Files:**

- Modify: allocator/models.py:93-101
- Modify: allocator/strategies/ilp_optimal.py:90-115,460-495
- Modify: allocator/box_features.py:312-376
- Modify: tests/test_box_features.py:665-677,1087-1093
- Modify: tests/test_strategies.py:146-160
- Create: tests/test_hard_negative_data.py

**Interfaces:**

~~~python
# allocator.box_features
FEATURE_SCHEMA_VERSION = 2
def stable_hash(obj: object) -> str: pass

# allocator.models.AllocationResult, final/defaulted field
solver_status: str | None = None

# allocator.strategies.ilp_optimal
def _record_solution_status(result: AllocationResult, pulp: object, prob: object) -> None: pass
def _record_fallback_solver_error(result: AllocationResult) -> None: pass
~~~

stable_hash() has exactly the behavior of the existing private _digest():
json.dumps(obj, sort_keys=True, default=list), SHA-256 of default UTF-8 bytes,
and the first 16 hexadecimal characters. In particular, do not introduce compact
JSON separators, because stable unchanged config_hash() output is required.

- [ ] **Step 1: Write the failing tests**

Create tests/test_hard_negative_data.py with:

~~~python
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
~~~

Append to tests/test_box_features.py:

~~~python
def test_feature_schema_version_and_public_stable_hash():
    import allocator.box_features as features

    assert features.FEATURE_SCHEMA_VERSION == 2
    assert features.stable_hash({"b": [2], "a": {1}}) == (
        features.stable_hash({"a": {1}, "b": [2]})
    )
    assert features.config_hash() == features.stable_hash(features._hash_inputs())
~~~

Extend the existing test_ilp_falls_back_to_local_search assertion in
tests/test_strategies.py:

~~~python
    assert two_box_result.solver_status == "FallbackSolverError"
~~~

- [ ] **Step 2: Run the focused tests to confirm the red state**

Run:

~~~bash
python3 -m pytest tests/test_hard_negative_data.py tests/test_box_features.py tests/test_strategies.py -q
~~~

Expected: failures for the missing solver_status, status helpers,
FEATURE_SCHEMA_VERSION, and stable_hash interfaces.

- [ ] **Step 3: Implement the feature API without changing the existing stamp**

In allocator/box_features.py, add FEATURE_SCHEMA_VERSION alongside module
constants. Rename the existing _digest to stable_hash and replace each internal
call with stable_hash:

~~~python
FEATURE_SCHEMA_VERSION = 2


def stable_hash(obj: object) -> str:
    """Return the established 16-hex configuration digest for JSON-like data."""
    payload = json.dumps(obj, sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def config_hash() -> str:
    """Stamp over thirteen effective schema/scoring inputs."""
    return stable_hash(_hash_inputs())
~~~

Keep _hash_inputs() unchanged. Update config_snapshot() to use stable_hash for
its readable sub-digests.

- [ ] **Step 4: Implement status recording at the correct boundaries**

Add this final field after AllocationResult.items:

~~~python
    # None means ilp-optimal was never invoked for this result.
    solver_status: str | None = None
~~~

Add the helpers near run() in allocator/strategies/ilp_optimal.py:

~~~python
def _record_solution_status(result: AllocationResult, pulp, prob) -> None:
    """Persist the solution-status evidence before validation can raise."""
    result.solver_status = pulp.LpSolution[prob.sol_status]


def _record_fallback_solver_error(result: AllocationResult) -> None:
    """Label an exception fallback without losing a non-optimal cause."""
    if result.solver_status in (None, "Optimal Solution Found"):
        result.solver_status = "FallbackSolverError"
~~~

Modify run() in this order:

~~~python
    try:
        import pulp
    except ImportError:
        result.solver_status = "FallbackImportError"
        from allocator.strategies import FALLBACK_STRATEGY, get_strategy
        logger.warning(f"PuLP not installed, falling back to {FALLBACK_STRATEGY}")
        get_strategy(FALLBACK_STRATEGY)(result)
        return

    if not result.boxes or not result.items:
        return

    try:
        _solve_ilp(result, pulp)
    except Exception as exc:
        from allocator.strategies import FALLBACK_STRATEGY, get_strategy
        logger.warning(f"ILP solver failed ({exc}), falling back to {FALLBACK_STRATEGY}")
        for box in result.boxes:
            box.allocations.clear()
        _record_fallback_solver_error(result)
        get_strategy(FALLBACK_STRATEGY)(result)
~~~

Immediately after status = prob.solve(solver) in _solve_ilp, before either
status-checking raise branch, call:

~~~python
    _record_solution_status(result, pulp, prob)
~~~

Do not change the allocator's existing LpStatus behavior. The generator makes
the stricter admission decision from the persisted LpSolution string.

- [ ] **Step 5: Verify this slice**

Run:

~~~bash
python3 -m pytest tests/test_hard_negative_data.py tests/test_box_features.py tests/test_strategies.py -q
python3 -m pytest -q
~~~

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

~~~bash
git add allocator/models.py allocator/strategies/ilp_optimal.py allocator/box_features.py \
  tests/test_hard_negative_data.py tests/test_box_features.py tests/test_strategies.py
git commit -m "feat: stamp hard-negative solver provenance"
~~~

---

### Task 2: Create the DB-free roster reconciliation contract

**Files:**

- Create: allocator/hard_negative_roster.py
- Modify: tests/test_hard_negative_data.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class RosterMatch:
    csv_name: str
    db_name: str

@dataclass(frozen=True)
class RosterIntersection:
    matches: tuple[RosterMatch, ...]
    csv_only: tuple[str, ...]
    db_only: tuple[str, ...]

class AmbiguousRosterIdentityError(ValueError): pass

def _casefold_unique(names: Iterable[str], source: str) -> dict[str, str]: pass
def correct_box_tiers(offer_id: int, boxes: Sequence[MysteryBox]) -> list[MysteryBox]: pass
def intersect_roster(csv_box_names: Iterable[str], db_box_names: Iterable[str]) -> RosterIntersection: pass
def roster_config_hash() -> str: pass
~~~

RosterMatch preserves both names: csv_name retrieves manual allocations from the
historical row map, while db_name retrieves the canonical email, preference, and
corrected tier. Output is ordered by casefolded identity for deterministic JSON.
Each source must be one-to-one under that identity: a duplicate (including a
case-only duplicate) raises `AmbiguousRosterIdentityError` rather than causing a
dictionary overwrite and an incorrectly selected box.

- [ ] **Step 1: Write failing roster and import-boundary tests**

Append to tests/test_hard_negative_data.py:

~~~python
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
        AmbiguousRosterIdentityError, intersect_roster,
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
~~~

Add a subprocess test patterned after
test_box_features_module_imports_from_isolated_root_without_db. Its import hook
must reject allocator.db, compare, and scripts.extract_features, then
successfully import allocator.hard_negative_roster.

- [ ] **Step 2: Run the tests to confirm the red state**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py -q -k "roster or tier"
~~~

Expected: ModuleNotFoundError for allocator.hard_negative_roster.

- [ ] **Step 3: Implement the module**

Create allocator/hard_negative_roster.py. Its only non-stdlib imports may be
BOX_TIERS, PER_OFFER_BOX_SIZE_OVERRIDES, MysteryBox, and stable_hash.
Import copy, dataclass, Iterable, and Sequence from the standard library as
needed; no runtime DB import is permitted.

~~~python
def _casefold_unique(names: Iterable[str], source: str) -> dict[str, str]:
    """Return normalised identity -> original spelling, rejecting ambiguity."""
    result = {}
    for name in names:
        identity = name.casefold()
        if identity in result:
            raise AmbiguousRosterIdentityError(
                f"{source} has case-normalised collision for {identity!r}: "
                f"{result[identity]!r}, {name!r}"
            )
        result[identity] = name
    return result


def correct_box_tiers(offer_id: int, boxes: Sequence[MysteryBox]) -> list[MysteryBox]:
    corrected_boxes = copy.deepcopy(list(boxes))
    overrides = PER_OFFER_BOX_SIZE_OVERRIDES.get(str(offer_id), {})
    override_names = _casefold_unique(overrides, "tier override")
    for box in corrected_boxes:
        original_override_name = override_names.get(box.name.casefold())
        tier = overrides.get(original_override_name) if original_override_name else None
        if tier in BOX_TIERS:
            box.tier = tier
            box.target_value = BOX_TIERS[tier]["target_value"]
    return sorted(
        corrected_boxes, key=lambda box: (box.target_value, box.name.casefold())
    )


def roster_config_hash() -> str:
    # Stamp the complete source mapping; irrelevant changes only conservatively
    # invalidate an artifact and avoid a second, Tier-A-specific hash contract.
    return stable_hash(PER_OFFER_BOX_SIZE_OVERRIDES)
~~~

Use `_casefold_unique()` for each CSV, DB, and applicable tier-override mapping.
It returns a normalised identity-to-original-spelling map, or raises
`AmbiguousRosterIdentityError` with the source, normalised identity, and both
original names. intersect_roster() constructs matches and each difference by
sorted casefolded keys, preserving the original spelling in every RosterMatch and
difference tuple. Do not call infer_box_tier: its CSV/summary/name-matching
layers do not safely apply to DB email boxes. Do not reject a non-empty
difference: the generator must preserve a non-empty valid intersection.

`PER_OFFER_BOX_SIZE_OVERRIDES` is shared historical configuration. Some current
entries (offers 65 and 67) are standalone CSV labels, not DB emails, so they
cannot safely match the email-named `MysteryBox` objects and remain inapplicable
here. `correct_box_tiers()` matches only case-normalised DB-email keys; it must
not add historical name matching. A case-normalised collision in the override
mapping raises the same ambiguity error as a roster collision.
`roster_config_hash()` deliberately stamps the full mapping, rather than a
Tier-A projection: a non-Tier-A edit may cause a harmless conservative
regeneration but can never make an artifact stale.

- [ ] **Step 4: Verify this slice**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py tests/test_box_features.py -q
python3 -m pytest -q
~~~

Expected: both commands exit 0, including the DB-free subprocess import check.

- [ ] **Step 5: Commit**

~~~bash
git add allocator/hard_negative_roster.py tests/test_hard_negative_data.py
git commit -m "feat: reconcile hard-negative box rosters"
~~~

---

### Task 3: Refactor synthetic generation for selected preference templates

**Files:**

- Modify: scripts/extract_features.py:71-173
- Modify: tests/test_tuning.py:831-905
- Modify: tests/test_hard_negative_data.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class SyntheticTemplate:
    box_name: str
    tier: str
    preference: str | None
    available_tags: dict[str, set[str]]

class EmptyPreferenceItemPoolError(ValueError): pass

def generate_synthetic_boxes(
    offer_id: int,
    item_lookup: dict[int, dict],
    available_tags: dict[str, set[str]],
    templates: Sequence[SyntheticTemplate] | None = None,
) -> list[dict]: pass
~~~

The no-template branch must retain the current legacy behavior: five recipes per
small/medium/large tier when the offer has fungible groups, source order, legacy
box names, and one Random(offer_id) sequence. The template branch emits every
applicable recipe for each selected DB box, selects only items allowed by that
template's fruit/veg preference, and passes that template's tier, preference,
and tag denominator into feature extraction.

- [ ] **Step 1: Write failing legacy and template tests**

Append to tests/test_tuning.py:

~~~python
def test_legacy_synthetic_sequence_remains_five_per_tier():
    import hashlib
    import json

    from scripts.extract_features import generate_synthetic_boxes

    records = generate_synthetic_boxes(
        100, TestSyntheticBoxes()._item_lookup(),
        {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()},
    )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    assert len(records) == 15
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "2558780f53d6fe0386a943aa235d585769c9d09b5ec40b8f9a14a68a17fb66c5"
    )
~~~

Append to tests/test_hard_negative_data.py:

~~~python
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
~~~

- [ ] **Step 2: Run the tests to confirm the red state**

~~~bash
python3 -m pytest tests/test_tuning.py tests/test_hard_negative_data.py -q -k "synthetic"
~~~

Expected: the template, exception-policy, and recipe-helper tests fail because
SyntheticTemplate, the templates parameter, and _synthetic_allocations do not
exist. If the legacy expectation reveals a pre-existing fixture assumption,
correct the expectation before changing production code.

- [ ] **Step 3: Share the allocation recipes while keeping legacy behavior**

Move the five current allocation recipes into one private helper:

~~~python
def _synthetic_allocations(
    item_lookup: dict[int, dict], tier: str, rng: Random,
) -> list[tuple[str, str, dict[int, int]]]:
    """Return source, legacy name fragment, and allocations for applicable recipes."""
~~~

Add dataclass and Sequence imports from the standard library at the top of
scripts/extract_features.py.

Move the existing monoculture, random, over-fungible, low-value, and high-value
math unchanged into that helper. The over-fungible recipe remains conditionally
absent when no fungible-grouped item exists; do not invent an empty/placeholder
row or a new synthetic attrition category. Its absence is semantically correct,
visible in `source_counts`, and covered by the existing source-family and paired
coverage gates. The public function has two branches:

~~~python
if templates is None:
    rng = Random(offer_id)
    for tier in ("small", "medium", "large"):
        for source, fragment, allocations in _synthetic_allocations(item_lookup, tier, rng):
            emit_legacy(
                f"synth_{fragment}_{tier}", tier, source, None, available_tags, allocations,
            )
else:
    for template in templates:
        rng = Random(f"{offer_id}:{template.box_name.casefold()}:{template.tier}")
        allowed_lookup = {
            item_id: info for item_id, info in item_lookup.items()
            if _matches_preference(info, template.preference)
        }
        if not allowed_lookup:
            raise EmptyPreferenceItemPoolError(
                f"{template.box_name!r} has no items for {template.preference!r}"
            )
        for source, _fragment, allocations in _synthetic_allocations(
            allowed_lookup, template.tier, rng
        ):
            emit_template(
                template.box_name, template.tier, source, template.preference,
                template.available_tags, allocations,
            )
~~~

Add a private `_matches_preference(info, preference)` that returns true for all
items with `None`, only `CATEGORY_FRUIT` for `fruit_only`, and only
`CATEGORY_VEGETABLES` for `veg_only`; import those category constants alongside
the existing configuration imports. Both emitters pass the original complete
`item_lookup` to feature extraction, while recipe construction receives the
filtered lookup, so IDs and category checks remain canonical. Define
`emit_legacy()` as the existing `_extract_or_skip()` skip-and-append behavior,
and `emit_template()` as a direct `extract_box_features()` call that appends a
non-None result. The latter deliberately permits `UnsupportedCategoryError` to
propagate; the hard-negative generator will turn it into a whole-offer exclusion
rather than a silent missing synthetic row.
Likewise, an empty filtered template pool raises EmptyPreferenceItemPoolError;
the hard-negative offer processor excludes that whole offer as
`empty_preference_item_pool` instead of retaining manual/baseline rows with no
same-roster synthetic counterpart.

Keep the existing `if not sorted_items: return []` guard at the top of
`_synthetic_allocations()`. It is part of the legacy no-item behavior and must
run before selecting the cheapest item.

- [ ] **Step 4: Verify this slice**

~~~bash
python3 -m pytest tests/test_tuning.py tests/test_hard_negative_data.py -q
python3 -m pytest -q
~~~

Expected: both commands exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/extract_features.py tests/test_tuning.py tests/test_hard_negative_data.py
git commit -m "feat: template synthetic hard-negative boxes"
~~~

---

### Task 4: Build pure artifact assembly, validation, and failure-report helpers

**Files:**

- Create: scripts/generate_hard_negatives.py
- Modify: tests/test_hard_negative_data.py

**Interfaces:**

~~~python
TIER_A_OFFER_IDS = frozenset(range(64, 110))
ADMITTING_ILP_STATUS = "Optimal Solution Found"
GENERATOR_VERSION = 1
BASELINE_SOURCES = (
    "baseline_deal_topup",
    "baseline_minmax_deficit",
    "baseline_greedy_best_fit",
)

def parse_requested_tier_a_offer_ids(value: str) -> list[int]: pass
def default_tier_a_offer_ids(available: set[int]) -> list[int]: pass
def resolve_requested_offer_ids(requested: list[int], available: set[int]) -> list[int]: pass
def customer_csv_names(names: Iterable[str]) -> list[str]: pass
def unextracted_row_reason(
    allocations: Mapping[int, int], item_lookup: Mapping[int, object],
) -> Literal["empty", "unextractable"]: pass
def tags_for_preference(preference: str | None, variants: dict[str, dict[str, set[str]]]) -> dict[str, set[str]]: pass
def admits_to_ilp_class(status: str | None) -> bool: pass
def paired_rung_coverage(records: list[dict], negative_selector: str) -> dict[str, int]: pass
def selected_roster_contract_failures(
    records: list[dict], expected: dict[tuple[int, str], dict],
) -> list[dict]: pass
def validation_failures(
    records: list[dict], source_counts: dict[str, int],
    roster_contract_failures: Sequence[dict] = (),
) -> list[dict]: pass
def build_artifact(
    records: list[dict], source_counts: dict[str, int], roster_check: dict,
    attrition: dict, exclusions: list[dict], requested_offer_ids: list[int],
    resolved_offer_ids: list[int],
) -> dict: pass
def build_failure_report(
    status: str, failed_gates: list[dict], source_counts: dict[str, int],
    roster_check: dict, attrition: dict, exclusions: list[dict], errors: list[dict],
    requested_offer_ids: list[int], resolved_offer_ids: list[int],
) -> dict: pass
def write_json_atomically(path: Path, payload: dict) -> None: pass
def finalize_run(
    records: list[dict], requested_offer_ids: list[int], resolved_offer_ids: list[int],
    roster_check: dict, attrition: dict, exclusions: list[dict], errors: list[dict],
    roster_contract_failures: Sequence[dict],
    out_path: Path, report_path: Path,
) -> int: pass
~~~

All of these helpers must live above runtime imports so a clean test checkout can
validate artifact shape and gates without DB access.

- [ ] **Step 1: Write failing gates and atomic-output tests**

Append to tests/test_hard_negative_data.py:

~~~python
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
~~~

The first test proves a loose global manual total cannot satisfy a rung: after
one offer loses all synth rows, 152 manual records remain but only 19 offers
have paired cells.

- [ ] **Step 2: Run the tests to confirm the red state**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py -q -k "paired or validation or failed_run"
~~~

Expected: ModuleNotFoundError for scripts.generate_hard_negatives.

- [ ] **Step 3: Implement the artifact contract**

Create scripts/generate_hard_negatives.py with only stdlib and the allocator
modules allocator.box_features, allocator.hard_negative_roster, and
allocator.config imported at module scope. Its stdlib imports include argparse,
Callable, Counter, dataclass, field, Iterable,
Literal, Mapping, json, os, Path, Sequence, tempfile, and copy. Import DONATION_IDENTIFIERS,
SKIP_COLUMN_IDENTIFIERS, and STAFF_IDENTIFIERS from allocator.config so the
customer-column filter has the same exact-name policy as compare.py. Import the
roster names needed by both pure assembly and runtime processing here:

~~~python
from allocator.hard_negative_roster import (
    AmbiguousRosterIdentityError,
    correct_box_tiers,
    intersect_roster,
    roster_config_hash,
)
~~~

Define the three rungs as:

~~~python
RUNG_NEGATIVE_SELECTORS = {
    "manual_vs_synth": "synth_",
    "manual_vs_baseline": "baseline_",
    "manual_vs_ilp": "ilp_optimal",
}
~~~

paired_rung_coverage must identify cells from (offer_id, tier), intersect manual
and the requested negative family, then count only manual records whose cell is
present in that intersection:

~~~python
def paired_rung_coverage(records, negative_selector):
    manual_cells = {(r["offer_id"], r["tier"]) for r in records if r["source"] == "manual"}
    if negative_selector == "ilp_optimal":
        negative_cells = {
            (r["offer_id"], r["tier"]) for r in records if r["source"] == "ilp_optimal"
        }
    else:
        negative_cells = {
            (r["offer_id"], r["tier"]) for r in records
            if r["source"].startswith(negative_selector)
        }
    paired_cells = manual_cells & negative_cells
    paired_manual = [
        r for r in records
        if r["source"] == "manual" and (r["offer_id"], r["tier"]) in paired_cells
    ]
    return {
        "manual_boxes": len(paired_manual),
        "offers": len({r["offer_id"] for r in paired_manual}),
    }
~~~

`tags_for_preference()` returns `variants["fruit_only"]` or
`variants["veg_only"]` for those exact preferences; `None` and any unexpected
value return `variants["all"]`, matching `parse_preference()`'s unrestricted
fallback. It returns the selected variant itself, never a merged tag set.
`admits_to_ilp_class()` is exactly `status == ADMITTING_ILP_STATUS`; use it for
both ILP admission and the ILP-status validation gate so no near-optimal or
fallback status can leak into the ILP class.

When extract_box_features() returns None, call unextracted_row_reason() before
incrementing `row_attrition[source]`. `empty` means every allocation quantity is
non-positive; `unextractable` means at least one quantity is positive but none of
those positive item IDs is in item_lookup. The helper is called only after a None
result, so these two outcomes are exhaustive: any positive, resolvable ID would
have made extract_box_features() return a record.

`selected_roster_contract_failures()` is the remaining write gate. For each
selected DB box, process_offer builds an expectation keyed by
`(offer_id, db_name.casefold())`, containing its corrected tier and the four
`dim_available` cardinalities of `tags_for_preference(box.preference, variants)`.
Every emitted row—manual, baseline, ILP, and synthetic—uses the canonical DB box
name and must match that expectation. The helper returns descriptive
`selected_roster_contract` failures for an unselected name, wrong tier, or wrong
denominator. This keeps preference provenance out of the numeric feature schema
while hard-failing an artifact assembled from the wrong roster or tag variant.

validation_failures() returns every failure, rather than stopping at the first:

1. every included ilp_optimal record has the exact admitting solver status;
2. all three coverage dictionaries have manual_boxes >= 150 and offers >= 20;
3. source counts include manual, at least one synth source, each exact baseline
   source, and ilp_optimal;
4. all supplied selected-roster contract failures.

`finalize_run()` does not compare a newly built artifact's stamps back to the
same live values: that would be tautological. It writes the feature-schema,
feature-config, and roster-config stamps as provenance; PR #3's artifact loader
will compare them to its live values before fitting an EBM.

build_artifact() produces exactly these nine top-level fields:

~~~python
sorted_records = sorted(
    records,
    key=lambda r: (
        r["offer_id"], r["tier"], r["source"], r["box_name"],
        r.get("solver_status", ""),
    ),
)
{
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
~~~

`GENERATOR_VERSION` is the module-level integer for generation semantics. Bump
it whenever a change can alter emitted records, source eligibility, or validation
meaning; do not bump it for formatting-only changes. It is provenance for the
later artifact consumer, not a second feature-schema version.

build_failure_report() produces exactly:
status, failed_gates, source_counts, roster_check, attrition, run_metadata,
exclusions, errors. Use status execution_failed if errors is non-empty;
otherwise use validation_failed when gates fail.

finalize_run() builds the artifact, passes the aggregated per-offer
roster_contract_failures to validation_failures(), then chooses whether the
normal artifact can be written. Configuration stamps are generation provenance,
not an in-process config audit.

write_json_atomically() creates the output parent, then uses
`NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)`.
It writes formatted sorted JSON, flushes and fsyncs, closes the temporary file,
then `os.replace()`s its saved path into place. `delete=False` and closing before
replacement make this work on Windows as well as POSIX. finalize_run() writes
--out only when there are no errors and no failed gates; failures write only
--report-out.

- [ ] **Step 4: Verify this slice**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py -q
python3 -m pytest -q
~~~

Expected: both commands exit 0.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/generate_hard_negatives.py tests/test_hard_negative_data.py
git commit -m "feat: validate hard-negative artifacts atomically"
~~~

---

### Task 5: Implement sequential DB generation and its test seam

**Files:**

- Modify: scripts/generate_hard_negatives.py
- Modify: tests/test_hard_negative_data.py

**Interfaces:**

~~~python
@dataclass
class OfferOutcome:
    offer_id: int
    records: list[dict]
    roster_entry: dict
    attrition: dict
    roster_contract_failures: list[dict] = field(default_factory=list)
    exclusion: dict | None = None
    error: dict | None = None

def empty_roster_entry(offer_id: int) -> dict: pass
def empty_offer_attrition() -> dict: pass
def process_offer(offer_id: int) -> OfferOutcome: pass
def execute(
    offer_ids: list[int],
    process_one: Callable[[int], OfferOutcome],
    *,
    out_path: Path,
    report_path: Path,
    requested_offer_ids: list[int],
) -> int: pass
def build_parser() -> argparse.ArgumentParser: pass
def main(argv: Sequence[str] | None = None) -> int: pass
~~~

execute() is the DB-free orchestration seam: tests supply a fake process_one.
process_offer() is the sole function that imports DB-backed code. main() resolves
offers, processes them in sorted order, and calls execute() directly; it does not
submit anything to an executor. `empty_roster_entry()` returns exactly
`{"offer_id": offer_id, "csv_only": [], "db_only": [], "selected_count": 0}`;
process_offer() creates it before its first early-exit condition and returns that
same complete shape for every exclusion path. `empty_offer_attrition()` returns
the exact per-offer substructure used by execute() for aggregation:

~~~python
{
    "roster_candidates": {"csv": 0, "db": 0, "selected": 0},
    "solver_statuses": {},
    "row_attrition": {},  # source -> {"empty": int, "unextractable": int}
}
~~~

Every early return carries this structure (with known counters incremented), and
every normal return also carries `roster_contract_failures` calculated before
the OfferOutcome is returned.

- [ ] **Step 1: Write failing orchestration tests**

Append to tests/test_hard_negative_data.py:

~~~python
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


def test_execute_discards_every_source_for_nonoptimal_offer(tmp_path):
    from scripts.generate_hard_negatives import OfferOutcome, execute

    outcome = OfferOutcome(
        offer_id=64, records=[],
        roster_entry={"offer_id": 64, "csv_only": [], "db_only": [], "selected_count": 2},
        attrition={
            "roster_candidates": {"csv": 2, "db": 2, "selected": 2},
            "solver_statuses": {"Solution Found": 1},
            "row_attrition": {},
        },
        exclusion={"offer_id": 64, "reason": "nonoptimal_ilp", "detail": "Solution Found"},
    )
    code = execute(
        [64], lambda _offer_id: outcome,
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
        [64], process_one,
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
            offer_id=64, records=[],
            roster_entry={"offer_id": 64, "csv_only": [], "db_only": [], "selected_count": 2},
            attrition={
                "roster_candidates": {"csv": 3, "db": 2, "selected": 2},
                "solver_statuses": {"Solution Found": 1},
                "row_attrition": {"manual": {"empty": 1, "unextractable": 2}},
            },
            exclusion={"offer_id": 64, "reason": "nonoptimal_ilp", "detail": "Solution Found"},
        ),
        65: OfferOutcome(
            offer_id=65, records=[],
            roster_entry={"offer_id": 65, "csv_only": [], "db_only": [], "selected_count": 3},
            attrition={
                "roster_candidates": {"csv": 4, "db": 3, "selected": 3},
                "solver_statuses": {"No Solution Exists": 1},
                "row_attrition": {
                    "manual": {"empty": 2, "unextractable": 1},
                    "synth_random": {"empty": 1, "unextractable": 0},
                },
            },
            exclusion={"offer_id": 65, "reason": "nonoptimal_ilp", "detail": "No Solution Exists"},
        ),
    }
    execute(
        [64, 65], lambda offer_id: outcomes[offer_id],
        out_path=tmp_path / "hard_negatives.json",
        report_path=tmp_path / "hard_negatives_report.json",
        requested_offer_ids=[64, 65],
    )
    report = __import__("json").loads(
        (tmp_path / "hard_negatives_report.json").read_text()
    )
    assert report["attrition"] == {
        "requested_offers": 2, "resolved_offers": 2,
        "eligible_offers": 0, "excluded_offers": 2,
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
        "offer_id": 64, "csv_only": [], "db_only": [], "selected_count": 0,
    }


def test_generator_module_import_does_not_import_compare_or_db():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import scripts.generate_hard_negatives; "
            "assert 'compare' not in sys.modules; assert 'allocator.db' not in sys.modules"
        )],
        cwd=root, env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
~~~

Add one DB-free, monkeypatched
`test_process_offer_uses_only_the_selected_customer_roster()`. Patch the lazy
runtime collaborators—not process_offer’s API—with a
two-category item lookup, CSV names containing one matched customer plus donation,
staff, and skip identifiers, one case-differing DB email box, deterministic
allocation results for all four strategies, and deterministic synthetic output.
Assert that the real process_offer() result contains only the canonical selected
DB email for every source; manual allocation is read using the matched CSV name;
every emitted `dim_available` equals the fruit-only variant; the filtered CSV
columns do not appear in its roster_entry; and `roster_contract_failures == []`.
Pass that exact returned OfferOutcome through execute() and assert that its
roster/solver/row attrition counters and filtered roster survive aggregation in
the failure report.
This test must not introduce a production dependency-injection API or connect to
the DB.

Use this executable setup scaffold, then retain the assertions above plus the
final failure-report assertions:

~~~python
def test_process_offer_uses_only_the_selected_customer_roster(monkeypatch, make_box, tmp_path):
    from types import SimpleNamespace
    import allocator.allocator as allocator_module
    import allocator.box_features as box_features
    from allocator.config import (
        CATEGORY_FRUIT, CATEGORY_VEGETABLES, DONATION_IDENTIFIERS,
        SKIP_COLUMN_IDENTIFIERS, STAFF_IDENTIFIERS,
    )
    import compare
    import scripts.generate_hard_negatives as hard_negatives

    csv_name, db_name = "fruit@example.com", "FRUIT@example.com"
    raw_csv_names = [
        csv_name, next(iter(DONATION_IDENTIFIERS)),
        next(iter(SKIP_COLUMN_IDENTIFIERS)), next(iter(STAFF_IDENTIFIERS)),
    ]
    lookup = {
        1: {"name": "Apple", "price": 100, "category_id": CATEGORY_FRUIT,
            "fungible_group": "apple", "fungible_degree": 0.7,
            "sub_category": "pome", "usage": "snacking", "colour": "red",
            "shape": "round", "size": 1},
        2: {"name": "Carrot", "price": 100, "category_id": CATEGORY_VEGETABLES,
            "fungible_group": None, "fungible_degree": 0.0,
            "sub_category": "root", "usage": "cooking", "colour": "orange",
            "shape": "long", "size": 1},
    }
    monkeypatch.setattr(compare, "_find_xlsx_path", lambda _offer_id: tmp_path / "offer.xlsx")
    monkeypatch.setattr(compare, "build_item_lookup", lambda _offer_id: lookup)
    monkeypatch.setattr(
        compare, "load_historical_csv", lambda _offer_id: (raw_csv_names, {1: {csv_name: 1}}),
    )
    monkeypatch.setattr(
        allocator_module, "build_boxes_from_db",
        lambda _offer_id: [make_box(name=db_name, tier="small", preference="fruit_only")],
    )
    strategies, manual_inputs = [], []
    real_extract = box_features.extract_box_features

    def fake_allocate(_offer_id, _xlsx_path, *, boxes, strategy, **_kwargs):
        strategies.append(strategy)
        boxes[0].allocations = {1: 1}
        return SimpleNamespace(boxes=boxes, solver_status="Optimal Solution Found")

    def spy_extract(box_name, allocations, *args, source="manual", **kwargs):
        if source == "manual":
            manual_inputs.append((box_name, allocations))
        return real_extract(box_name, allocations, *args, source=source, **kwargs)

    monkeypatch.setattr(allocator_module, "allocate", fake_allocate)
    monkeypatch.setattr(box_features, "extract_box_features", spy_extract)
    outcome = hard_negatives.process_offer(80)
    assert strategies == ["ilp-optimal", "deal-topup", "minmax-deficit", "greedy-best-fit"]
    assert manual_inputs == [(db_name, {1: 1})]
    # Assert the selected-roster and failure-report contract described above.
~~~

- [ ] **Step 2: Run the tests to confirm the red state**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py -q -k "requested or execute or generator_module or process_offer"
~~~

Expected: failures for the absent requested/resolved parser, outcome, execute,
and process_offer contracts.

- [ ] **Step 3: Implement process_offer() around one selected roster**

Inside process_offer(), import lazily:

~~~python
import compare
from allocator.allocator import allocate, build_boxes_from_db
from allocator.box_features import UnsupportedCategoryError, extract_box_features
from scripts.extract_features import (
    EmptyPreferenceItemPoolError, SyntheticTemplate, generate_synthetic_boxes,
)
~~~

Perform this exact sequence for every offer:

1. Start with `roster_entry = empty_roster_entry(offer_id)` and
   `attrition = empty_offer_attrition()`. Resolve
   xlsx_path = compare._find_xlsx_path(offer_id). If absent, return a
   missing_xlsx exclusion and no records. Use this and every DB lookup through
   the existing production non-deleted path; do not add historical name matching
   or call fetch_offer_parts_by_name(include_deleted=True).
2. Load plain item_lookup = compare.build_item_lookup(offer_id). If empty,
   return missing_item_lookup. Load raw_csv_names and historical_allocations with
   compare.load_historical_csv(offer_id), then pass raw_csv_names through
   customer_csv_names(). That helper must use the same exact-name exclusion
   predicate as compare.py: remove DONATION_IDENTIFIERS,
   SKIP_COLUMN_IDENTIFIERS, and STAFF_IDENTIFIERS before roster intersection. If
   no customer columns remain, return missing_historical_csv; never report a
   donation, staff, or skip column as CSV-only. Set
   `attrition["roster_candidates"]["csv"] = len(csv_names)` immediately after
   filtering.
3. In one `try` block, first build
   `db_boxes = correct_box_tiers(offer_id, build_boxes_from_db(offer_id))`, then
   intersect its email names with CSV names. This one catch deliberately covers
   both correct_box_tiers()' case-normalised override collisions and
   intersect_roster() CSV/DB collisions: return an ambiguous_roster_identity
   exclusion whose detail is the error message. After successful tier correction,
   set `attrition["roster_candidates"]["db"] = len(db_boxes)`; after the
   intersection, set its `selected` counter to `len(intersection.matches)`, then
   put CSV-only, DB-only, and selected count in roster_entry. If there are no
   matches, return empty_roster_intersection; a non-empty difference continues
   normally. Every later early exclusion returns this complete roster_entry rather
   than `{}` or `None`.
4. Build selected boxes in RosterMatch order. Retain the pair of csv_name and a
   copied DB box so manual reconstruction uses the real historical column while
   allocation uses DB tier/preference. Every emitted feature record, including
   manual, uses the canonical `db_name` as box_name; csv_name is retrieval-only.
5. Build tags as follows; no preference selects all:

   ~~~python
   variants = {
       "all": compare.compute_available_tags(item_lookup),
       "fruit_only": compare.compute_available_tags(item_lookup, preference="fruit_only"),
       "veg_only": compare.compute_available_tags(item_lookup, preference="veg_only"),
   }
   ~~~
6. Call `allocate(offer_id, xlsx_path, boxes=copy.deepcopy(selected_boxes),
   strategy="ilp-optimal")` first. Any status other than the admitting string
   returns a nonoptimal_ilp whole-offer exclusion and records that status count.
   Add solver_status to every admitted ILP feature record.
7. For each of deal-topup, minmax-deficit, and greedy-best-fit, call
   `allocate()` with another independent `copy.deepcopy(selected_boxes)` and
   that `strategy=` value. Do not import or call a baseline strategy's `run()`
   function directly: allocate() retains the shared item, charity, and stock
   infrastructure required by CLAUDE.md. Emit source labels baseline_deal_topup,
   baseline_minmax_deficit, baseline_greedy_best_fit. The four `allocate()` calls
   deliberately reread the XLSX. Do not widen the production allocator with
   preloaded items/overage solely for this one-off diagnostic; measure it after
   MVP if runtime becomes material.
8. Rebuild manual records from historical_allocations[item_id][csv_name] with
   the matched corrected tier/preference and plain item_lookup. Do not call
   read_xlsx_pack_overrides.
9. Build one SyntheticTemplate per selected box using its canonical db_name,
   corrected tier, preference, and matching tag variant. Pass all templates to
   generate_synthetic_boxes(). It filters recipe inputs to the template
   preference and may propagate UnsupportedCategoryError or
   EmptyPreferenceItemPoolError.
10. extract_box_features() gets the matching per-box tag variant for every
    manual/baseline/ILP record. A None feature increments numeric row attrition.
    Use unextracted_row_reason() to select its exact `empty` or
    `unextractable` counter. Catch UnsupportedCategoryError around both steps 9
    and 10; return unsupported_category and discard all provisional records for
    the offer. Catch EmptyPreferenceItemPoolError from step 9 as the atomic
    empty_preference_item_pool exclusion, also discarding all provisional records.
11. Build selected-roster expectations from the selected DB boxes and variants,
    run selected_roster_contract_failures() over every provisional record, and
    return those failures with the OfferOutcome. They are validation failures,
    not ordinary attrition, so a contract failure prevents normal artifact write.

Catch unexpected exceptions in execute(), not as ordinary attrition: record
offer_id, exception class, and message in errors; continue to later sorted
offers; and make the final report execution_failed. Such an error is neither an
eligible nor an excluded outcome, because process_offer() did not return an
eligibility decision. It remains included only in resolved_offers and errors.

- [ ] **Step 4: Implement execute() and CLI**

Aggregate outcomes in input order. Keep roster_check as:

~~~python
roster_check = {
    "offers": sorted(roster_entries, key=lambda entry: entry["offer_id"]),
    "totals": {
        "csv_only": sum(len(entry["csv_only"]) for entry in roster_entries),
        "db_only": sum(len(entry["db_only"]) for entry in roster_entries),
        "selected": sum(entry["selected_count"] for entry in roster_entries),
    },
}
~~~

execute() aggregates each known OfferOutcome substructure by summation (never
dict.update): add each fixed roster counter, union-and-add solver-status keys,
then union-and-add every dynamic `row_attrition[source][reason]` pair. It pins
the normal/report `attrition` shape exactly as follows:

~~~python
{
    "requested_offers": len(requested_offer_ids),
    "resolved_offers": len(offer_ids),
    "eligible_offers": int,
    "excluded_offers": int,
    "roster_candidates": {"csv": int, "db": int, "selected": int},
    "solver_statuses": {"status string": int},
    "row_attrition": {"source": {"empty": int, "unextractable": int}},
    "rung_coverage": {"manual_vs_*": {"manual_boxes": int, "offers": int}},
}
~~~

It also concatenates `outcome.roster_contract_failures` in sorted offer order and
passes them to finalize_run(). exclusions remains the separate per-offer
reason/detail event list.

build_parser() exposes only:

~~~text
--only-offers    comma-separated IDs/ranges; default all available Tier-A offers
--out            default diagnostics/hard_negatives.json
--report-out     default diagnostics/hard_negatives_report.json
~~~

Do not use compare._build_offer_ids for explicit selection because it allows Tier
B–D values. In main(), lazily import compare and set
`available = compare._discover_cleaned_offer_ids()`: that existing narrow helper
is the authoritative scan of cleaned historical CSVs without collapsing the
requested-ID audit trail. For an explicit `--only-offers`,
parse_requested_tier_a_offer_ids() first expands the requested Tier-A range
without considering availability. Then resolve_requested_offer_ids() intersects
it with `available`, sorted, and rejects an empty resolved set. This preserves
unavailable but valid explicit IDs in run_metadata.requested_offer_ids and writes
only runnable ones to run_metadata.resolved_offer_ids. With no flag, both lists
are the sorted available Tier-A IDs. Keep this small parser local:
compare._build_offer_ids is a private, broader policy helper whose availability
intersection would erase the requested-ID audit trail; extracting a shared
production helper is not MVP work. `default_tier_a_offer_ids(available)` is the
no-flag path and returns exactly `sorted(available & TIER_A_OFFER_IDS)`.
resolve_requested_offer_ids() also intersects its result with TIER_A_OFFER_IDS
defensively. main() must call the default helper rather than passing all
discovered IDs through to execution.

Finish with:

~~~python
if __name__ == "__main__":
    raise SystemExit(main())
~~~

- [ ] **Step 5: Verify this slice**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py tests/test_tuning.py tests/test_strategies.py -q
python3 -m pytest -q
~~~

Expected: both commands exit 0. The subprocess test proves import remains DB-free.

- [ ] **Step 6: Commit**

~~~bash
git add scripts/generate_hard_negatives.py tests/test_hard_negative_data.py
git commit -m "feat: generate reconciled hard-negative data"
~~~

---

### Task 6: Document operator validation and verify the PR boundary

**Files:**

- Modify: CLAUDE.md:100-114
- Modify: tests/test_hard_negative_data.py

**Interfaces:**

~~~bash
python3 scripts/generate_hard_negatives.py
python3 scripts/generate_hard_negatives.py --only-offers 85-86 \
  --out diagnostics/hard_negatives.json \
  --report-out diagnostics/hard_negatives_report.json
~~~

A two-offer run should fail the 150-box/20-offer gate. It is a valid smoke test
only when it leaves --out untouched and writes a complete report.

- [ ] **Step 1: Add the final CLI-scope test**

~~~python
def test_generator_parser_exposes_only_mvp_options():
    from scripts.generate_hard_negatives import build_parser

    parser = build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert {"--only-offers", "--out", "--report-out"} <= options
    assert "--workers" not in options
~~~

- [ ] **Step 2: Verify parser scope**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py -q -k "parser_exposes"
~~~

Expected: exit 0. The parser helper has no DB imports.

- [ ] **Step 3: Document normal and audit workflows**

Add immediately after the existing extract_features.py examples in CLAUDE.md:

~~~markdown
python3 scripts/generate_hard_negatives.py                       # Tier-A EBM input; needs DB
python3 scripts/generate_hard_negatives.py --only-offers 85-86   # smoke test; writes failure report below gates
~~~

Add this sentence after the command block: hard_negatives.json is replaced only
after all three paired EBM rungs pass; inspect
diagnostics/hard_negatives_report.json after any non-zero generation run.

- [ ] **Step 4: Run final static and test verification**

~~~bash
python3 -m pytest tests/test_hard_negative_data.py tests/test_box_features.py tests/test_tuning.py tests/test_strategies.py -q
python3 -m pytest -q
git diff --check
~~~

Expected: every pytest command exits 0 and git diff --check produces no output.

- [ ] **Step 5: Run live validation in increasing scope**

With DB credentials and historical files available, run:

~~~bash
python3 scripts/generate_hard_negatives.py --only-offers 85-86
~~~

Verify the non-zero report has requested/resolved IDs, roster differences, the
pinned attrition keys, solver-status counts, selected-roster contract status, and
expected gate failures; verify no normal artifact was created or replaced. Then
run:

~~~bash
python3 scripts/generate_hard_negatives.py
~~~

On success, inspect the nine artifact fields, live schema/config/roster stamps,
stable record ordering, all required source families, each coverage count at or
above 150/20, and that no selected-roster contract failure was reported. On
failure, keep the report as the diagnosis; do not lower a gate or fit an EBM from
partial data.

- [ ] **Step 6: Commit and hand off**

~~~bash
git add CLAUDE.md tests/test_hard_negative_data.py
git commit -m "docs: document hard-negative generation"
~~~

If the full Tier-A run succeeds, PR #3 may consume only the guarded artifact.
If it fails, use its attrition cause rather than expanding scope or attempting
an exploratory model fit.

---

## Plan Self-Review

| Spec requirement | Plan coverage |
|---|---|
| Defaulted solver provenance, strict solution status, fallback behavior | Task 1 |
| Feature schema version and shared roster hash | Tasks 1–2 |
| DB-free tier correction and case-insensitive reconciliation | Task 2 |
| Preference-filtered template synthetics without tuning-output regression | Task 3 |
| Artifact/report fields, provenance and paired gates, atomic write | Task 4 |
| Tier-A sequential generation, filtered roster, plain-price manual rebuild, atomic exclusions | Task 5 |
| No workers, operator docs, test/live verification | Task 6 |

No EBM, ordinal regression, config-drift audit, Tier B–D support, historical name
matching, or executor/tunnel redesign appears in this plan; each is outside the
approved MVP.
