# Hard-Negative Data Set — Design

**Date:** 2026-08-10
**Status:** Approved for planning
**Scope:** PR #2 of the pre-survey diagnostics milestone

## Purpose

Create a trustworthy, reproducible EBM input artifact before packer-survey
responses arrive. The artifact distinguishes manually packed boxes from a
difficulty ladder of machine-generated boxes without leaking class identity
through solver fallbacks, roster mismatches, price bases, or diversity-feature
denominators.

The artifact enables a later EBM implementation (PR #3), but this PR does not
fit a model or change production scoring.

## Delivery sequence

The pre-survey diagnostics milestone is deliberately split into independent,
reviewable pull requests:

1. **PR #2 — hard-negative data set:** this design.
2. **PR #3 — EBM diagnostic:** load the guarded artifact, fit the three EBM
   rungs, run the permutation gate, and emit hypotheses.
3. **PR #4 — ordinal pre-survey machinery:** pre-ship scenario gate, pure
   features, simulated-response fit, recovery tests, and power analysis.

Packer-survey responses are not required by any of these implementation slices.
They are required only to produce the ordinal model's real coefficients and
empirical value-band result, and to corroborate or challenge EBM hypotheses.

## Non-goals

- No EBM model, InterpretML import, model finding, or score/config change.
- No ordinal-regression code or survey-file change.
- No item/config-drift audit; that remains a separate backlog item.
- No changes to `scripts/extract_features.py`'s tuning output or its use of
  XLSX pack-price overrides.
- No Tier B–D inputs. The generator supports only the existing Tier-A offer set
  (64–109); `--only-offers` must resolve to a non-empty subset of that set.
  Historical name matching and its `include_deleted=True` fallback are not part
  of this MVP.
- No change to allocation strategy selection or production allocator results,
  apart from recording solver provenance on `AllocationResult`.

## Data flow

```text
historical CSV + DB buyer roster + XLSX input
                    |
                    v
      pure tier correction and roster intersection
                    |
                    v
 manual reconstruction + four allocation strategies + regenerated synthetics
                    |
                    v
       PR #1 DB-free box-feature extraction and config stamping
                    |
                    v
 hard-negative validation gates ---- fail ---> audit-only report
                    |
                    +---- pass ---> usable hard_negatives.json
```

## Components and contracts

### ILP status provenance

Add `solver_status: str | None = None` as the final, defaulted field of
`AllocationResult`. `None` means the ILP was never run.

`allocator.strategies.ilp_optimal` records the PuLP *solution-status* string
immediately after `prob.solve(...)`, before any exception path. It must not use
`LpStatus`: a time-limited integer-feasible incumbent may report `LpStatus` as
`"Optimal"` even though its solution status is not a proof of optimality.

The generator admits an `ilp_optimal` record only for:

```text
"Optimal Solution Found"
```

It rejects and reports all other solution statuses, including time-limited
`"Solution Found"`, no-solution states, `FallbackImportError`, and
`FallbackSolverError`.

`run()` sets `FallbackImportError` before invoking the no-PuLP fallback. Its
solver-exception path clears partial allocations, calls
`_record_fallback_solver_error(result)`, then invokes the fallback strategy.
That helper replaces `None` or `"Optimal Solution Found"` with
`FallbackSolverError`; it preserves any other recorded non-optimal solution
status so the attrition report retains its cause. The optimal status must be
cleared because a post-solve extraction error replaces the ILP allocation with
fallback boxes.

### Pure roster reconciliation

Create `allocator/hard_negative_roster.py`, with no DB imports or import-time
side effects.

- `correct_box_tiers(offer_id, boxes)` applies only
  `PER_OFFER_BOX_SIZE_OVERRIDES` to the DB-derived email roster, refreshes each
  corrected box's `target_value`, then sorts by target value. Other historical
  box-tier resolution layers are keyed by standalone CSV names and cannot be
  reached safely from email-named DB boxes.
- `intersect_roster(csv_box_names, db_box_names)` case-normalises names and
  returns retained email matches plus CSV-only and DB-only differences.

The manual class contains only the CSV/DB email intersection. Standalone CSV
names have no reliable email bridge, and their preferences cannot be resolved
against the email-keyed DB roster; retaining them would create class-dependent
diversity denominators.

For each offer, the generator first builds and tier-corrects the DB roster,
intersects it with historical email columns, and creates one selected roster
from those matched boxes. Every allocation strategy receives a fresh copy of
that selected roster. Manual rows use the matched box's corrected tier and DB
preference directly. This gives manual, baseline, and ILP rows the same box
population, tier, and preference distribution.

### Generator

Create `scripts/generate_hard_negatives.py` as the only DB-backed entry point.
It accepts the existing Tier-A offer range syntax, `--workers`, `--out`, and
`--report-out`. Defaults are: all Tier-A offers for `--only-offers`, `4` for
`--workers`, `diagnostics/hard_negatives.json` for `--out`, and
`diagnostics/hard_negatives_report.json` for `--report-out`.

Each top-level, picklable per-offer worker:

1. Validates that the offer is Tier A, then resolves the source XLSX and DB item
   lookup using the production non-deleted lookup path.
2. Builds three `available_tags` variants: unrestricted, fruit-only, and
   veg-only. Preference-less sources use the unrestricted variant.
3. Builds, tier-corrects, and intersects the selected roster once; passes a
   fresh copy of it to `allocate()` for `ilp-optimal`, `deal-topup`,
   `minmax-deficit`, and `greedy-best-fit`.
4. Emits `ilp_optimal` rows only from a proven-optimal ILP run and emits the
   three baseline source labels for their respective allocation results.
5. Rebuilds the retained manual boxes using **plain DB prices**, not XLSX
   overrides, and the selected roster's matching preference/tag denominator.
6. Generates every standard `synth_*` negative for every selected box, using
   that box's tier and matching preference/tag variant. This mirrors the manual
   preference distribution exactly; synthetics are not an unrestricted,
   class-specific denominator population.

The output therefore supports the three later EBM rungs:

| Rung | Positive source | Negative sources |
|---|---|---|
| manual vs synth | `manual` | `synth_*` |
| manual vs baseline | `manual` | `baseline_*` |
| manual vs ILP | `manual` | `ilp_optimal` |

Manual records must be rebuilt rather than reused from
`diagnostics/tuning_features.json`: its XLSX pack-price overrides would provide
a systematic class marker, and it predates the PR #1 feature schema.

## Artifact and validation rules

`allocator.box_features` defines and exports `FEATURE_SCHEMA_VERSION = 2`; the
generator stamps that constant rather than an unowned literal. The usable
artifact has that PR #1 schema version, the current `config_hash`, and a
separate `roster_config_hash` over `PER_OFFER_BOX_SIZE_OVERRIDES`. The separate
roster hash preserves PR #1's documented 13-input feature hash while detecting
changes to the generator's tier-correction input.

Its required top-level shape is:

```text
feature_schema_version, config_hash, roster_config_hash, records,
source_counts, roster_check, attrition, exclusions, run_metadata
```

`source_counts` maps each concrete source label to its record count.
`roster_check` holds per-offer CSV-only and DB-only identities plus aggregate
counts. `attrition` contains only numeric sample-loss accounting: roster
intersection counts, solver-status counts, and paired-cell/final counts for
**each** EBM rung. `exclusions` is the per-offer event list, with a reason and
detail for each skipped offer or discarded source. `run_metadata` records the
requested offer IDs, resolved offer IDs, worker count, and deterministic
generator version.

Each rung's coverage is computed after matching its manual and negative rows by
`(offer_id, tier)` and dropping cells that lack either class. A loose count of
all manual rows must never satisfy any rung's gate.

`hard_negatives.json` is written atomically and only when every gate passes:

1. Every included ILP row has `solver_status == "Optimal Solution Found"`.
2. Each paired rung—manual-vs-synth, manual-vs-baseline, and manual-vs-ILP—
   retains at least **150 manual boxes** across at least **20 offers**.
3. All retained manual rows come from the case-normalised email roster
   intersection and use the matching preference/tag variant.
4. The run carries the current PR #1 schema version, feature config hash, and
   roster config hash; each must match its live value. Every required family is
   represented: `manual`, at least one `synth_*` source, all three named
   `baseline_*` sources, and `ilp_optimal`.

On a failed gate, the command exits non-zero, does not create or replace the
normal artifact, and writes the audit-only report. If a prior successful normal
artifact exists at `--out`, it remains untouched; the failed run never marks it
as current or substitutes it for this run's result. Its required top-level
shape is:

```text
status, failed_gates, source_counts, roster_check, attrition,
run_metadata, exclusions, errors
```

`status` is `"validation_failed"` or `"execution_failed"`; the latter identifies
an unexpected worker/code error. A `validation_failed` report has one or more
`failed_gates`; an `execution_failed` report may have none, but has one or more
`errors`. The report lists each failed gate, source counts, roster differences,
worker/data exclusions, non-optimal solver statuses by cause, and paired-rung
counts.

Known data exclusions (missing XLSX, no item lookup, roster mismatch,
and non-optimal solver status) are reported per offer. An unsupported category
feature means `extract_box_features()` raised `UnsupportedCategoryError` because
a positive-quantity resolved item is neither the configured fruit nor vegetable
category; the generator excludes that entire offer from every source family to
avoid a class-specific row loss. Unexpected worker or code errors also fail the
run and identify the exception; they must not be silently converted into
ordinary attrition.

Workers are collected in deterministic offer order. Records have a stable
sorting order before output so changing `--workers` cannot change the artifact.

## Testing and verification

The test suite remains DB-free and uses synthetic fixtures only. It covers:

- the defaulted final `solver_status` field and every fallback/provenance path;
- strict ILP admission status;
- tier correction, target-value refresh, resorting, and case-insensitive roster
  intersection;
- selected-roster copying and identical per-box preference/tag variants across
  manual, generated, and synthetic rows;
- plain helper behavior for source labels, all three paired-rung gates,
  attrition/exclusion separation, feature/roster config stamps, and Tier-A-only
  argument validation;
- failure reports and atomic non-emission when any validation gate fails;
- deterministic aggregation order and unexpected-worker-error reporting.

Operator validation, with live DB access, runs first on an explicit offer subset
and then across Tier A. The review checks the attrition report, roster mismatch
counts, and that the output is accepted by the future PR #3 schema guard.

## Handoff to PR #3

PR #3 accepts only `hard_negatives.json`; it never consumes a failure report.
Its loader hard-fails on an unsupported feature schema version, feature-config
hash mismatch, or roster-config hash mismatch. PR #3 will fit separate balanced,
leave-offer-out EBM models for the three rungs and report findings as hypotheses
only. No score or ILP change is authorised from EBM output before survey evidence
is available for review.
