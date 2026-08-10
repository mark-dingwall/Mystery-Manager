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
`FallbackSolverError`. A fallback after a post-solve exception must not retain
an earlier optimal status, because the result's boxes have been replaced by the
fallback allocation.

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

### Generator

Create `scripts/generate_hard_negatives.py` as the only DB-backed entry point.
It accepts the existing Tier-A offer range syntax, `--workers`, `--out`, and
`--report-out`. Their defaults are respectively
`diagnostics/hard_negatives.json` and
`diagnostics/hard_negatives_report.json`.

Each top-level, picklable per-offer worker:

1. Resolves the source XLSX and DB item lookup.
2. Builds three `available_tags` variants: unrestricted, fruit-only, and
   veg-only. Preference-less sources use the unrestricted variant.
3. Builds and tier-corrects a fresh DB roster for each strategy, then calls
   `allocate()` directly for `ilp-optimal`, `deal-topup`, `minmax-deficit`, and
   `greedy-best-fit`.
4. Emits `ilp_optimal` rows only from a proven-optimal ILP run and emits the
   three baseline source labels for their respective allocation results.
5. Rebuilds the retained manual boxes using **plain DB prices**, not XLSX
   overrides, and the same preference/tag denominator rule as allocation rows.
6. Regenerates the existing `synth_*` records with the unrestricted tag set.

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

The usable artifact has the PR #1 feature schema version and current
`config_hash`, plus normalised records, source counts, roster differences, and
per-cause attrition. Its required top-level shape is:

```text
feature_schema_version, config_hash, records, source_counts,
roster_check, attrition, run_metadata
```

`source_counts` maps each concrete source label to its record count.
`roster_check` holds per-offer CSV-only and DB-only identities plus aggregate
counts. `attrition` holds per-cause solver-status counts, offer-level exclusions,
and paired manual-vs-ILP counts. `run_metadata` records the requested offer IDs,
resolved offer IDs, worker count, and deterministic generator version.

The primary rung's coverage is computed only after matching manual and ILP rows
by `(offer_id, tier)` and dropping cells that lack either class. A loose count of
all manual rows must never satisfy this gate.

`hard_negatives.json` is written atomically and only when every gate passes:

1. Every included ILP row has `solver_status == "Optimal Solution Found"`.
2. The paired manual-vs-ILP rung retains at least **150 manual boxes** across at
   least **20 offers**.
3. All retained manual rows come from the case-normalised email roster
   intersection and use the matching preference/tag variant.
4. The run carries the current PR #1 schema version and config hash, and each
   required family is represented: `manual`, at least one `synth_*` source, all
   three named `baseline_*` sources, and `ilp_optimal`.

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
an unexpected worker/code error. `failed_gates` is empty only on a successful
run. The report lists each failed gate, source counts, roster differences,
worker/data exclusions, non-optimal solver statuses by cause, and paired-rung
counts.

Known data exclusions (missing XLSX, no item lookup, roster mismatch,
unsupported category feature, and non-optimal solver status) are reported per
offer. Unexpected worker or code errors also fail the run and identify the
exception; they must not be silently converted into ordinary attrition.

Workers are collected in deterministic offer order. Records have a stable
sorting order before output so changing `--workers` cannot change the artifact.

## Testing and verification

The test suite remains DB-free and uses synthetic fixtures only. It covers:

- the defaulted final `solver_status` field and every fallback/provenance path;
- strict ILP admission status;
- tier correction, target-value refresh, resorting, and case-insensitive roster
  intersection;
- plain helper behavior for source labels, tag variants, paired-rung counting,
  attrition, and schema/config stamps;
- failure reports and atomic non-emission when any validation gate fails;
- deterministic aggregation order and unexpected-worker-error reporting.

Operator validation, with live DB access, runs first on an explicit offer subset
and then across Tier A. The review checks the attrition report, roster mismatch
counts, and that the output is accepted by the future PR #3 schema guard.

## Handoff to PR #3

PR #3 accepts only `hard_negatives.json`; it never consumes a failure report.
Its loader hard-fails on an unsupported feature schema version or a config-hash
mismatch. PR #3 will fit separate balanced, leave-offer-out EBM models for the
three rungs and report findings as hypotheses only. No score or ILP change is
authorised from EBM output before survey evidence is available for review.
