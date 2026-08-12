# EBM Diagnostic MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a DB-free, deterministic EBM diagnostic that turns the validated hard-negative artifact into three interpretable, confound-caveated hypothesis reports.

**Architecture:** `scripts/ebm_diagnostic.py` is the only new production module. It loads the atomically-published hard-negative artifact, validates all provenance stamps, prepares one matched box-cluster cohort for each rung, flattens it through `allocator.box_features`, and runs EBM inference without importing allocation or DB modules. Diagnostic-only tests use synthetic feature records and the isolated dependency stack.

**Tech Stack:** Python 3.10, InterpretML `ExplainableBoostingClassifier`, scikit-learn grouped cross-validation/AUC, NumPy, pytest diagnostics marker.

## Global Constraints

- The sole input is `diagnostics/hard_negatives.json`; never read `tuning_features.json`.
- Require `feature_schema_version == FEATURE_SCHEMA_VERSION`, live `config_hash()`, and live `roster_config_hash()` before reading records. Reject generator failure reports and malformed top-level payloads early.
- The script must remain DB-free at import time and must not import `compare`, `allocator.db`, or allocation strategies.
- Use `flatten()` unchanged. Never flatten `item_quantities`, `group_totals`, source labels, roster data, scores, or penalties.
- Every rung uses one prepared cohort of complete `(offer_id, tier, box_name)` clusters for OOF AUC, in-sample fit, permutations, and reported attrition. The latest hard-negative generator makes this stronger identity contract available; it supersedes the older parked plan's cell-only cohort wording.
- Rungs are `manual_vs_synth` (`synth_*`), `manual_vs_baseline` (`baseline_*`), and `manual_vs_ilp` (`ilp_optimal` exactly). A cluster needs one manual row and at least one rung-negative row.
- EBM primary fits are class-weighted, use `GroupKFold(n_splits=5)` grouped on `offer_id`, and report OOF and in-sample AUC separately.
- Permutation inference uses at least 200 full-data fits and shuffles labels only inside complete matched clusters. maxT is per rung and includes promotable columns only; scored columns never become findings.
- Findings are ranked hypotheses only. State that maxT controls chance association, not the known value-confound pathway. The diagnostic must not change allocation/scoring parameters.
- Output JSON records the seed, resolved diagnostic-library versions, input provenance, cohort attrition, column/family counts, and deferred-MVP limitations.
- Tests are synthetic-only, marked `diagnostics`, use `tests.conftest.require_dep`, and make no DB/network calls.

## Frozen MVP boundary

Included: provenance/shape guards, deterministic matrix and basis classification, complete-cluster preparation, grouped OOF + in-sample AUC, serial 200-permutation maxT, scored-term exclusion, group-parent aggregate correlation, value-confound metadata/ablation, machine-readable findings, and CLI/docs.

Deferred: plot generation, the three exploratory interaction models, multi-seed stability, process-parallel permutations, tag-parent refit permutations, and leave-one-negative-source-out ablations. Parent tag terms are reported but cannot be promoted until their specific aggregate-explained protocol ships. This is an internal, non-deploying MVP; the output makes every deferral visible.

---

### Task 1: Artifact contract, basis map, and matched-cohort preparation

**Files:**
- Create: `scripts/ebm_diagnostic.py`
- Create: `tests/test_ebm_diagnostic.py`

**Interfaces:**
- `load_artifact(path: Path) -> dict`
- `basis_for_columns(columns: Sequence[str]) -> dict[str, str]`
- `prepare_rung(records: Sequence[dict], rung: str) -> PreparedRung`
- `build_design_matrix(records: Sequence[dict]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[tuple[int, str, str]]]`

- [ ] **Step 1: Write failing diagnostics tests**

Cover a valid stamped artifact; rejection of a missing/incorrect schema, feature hash, roster hash, and failure report; exact source selection; unknown flattened columns; and removal of incomplete box clusters. Construct records only from the existing synthetic feature helpers so `flatten()` supplies the live-config column set.

- [ ] **Step 2: Verify the tests fail for missing module symbols**

Run:

```bash
PYTHONPATH=.venv-diagnostics/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -S -m pytest tests/test_ebm_diagnostic.py -m diagnostics \
  --strict-diagnostics-deps -k 'artifact or basis or prepare or design_matrix' -v
```

Expected: collection/import failure because `scripts.ebm_diagnostic` does not exist.

- [ ] **Step 3: Implement the minimal DB-free contract layer**

Use `FEATURE_SCHEMA_VERSION`, `config_hash`, `flatten`, and `roster_config_hash` from their owning modules. Validate required top-level fields before reading `records`. Build dynamic scored/parent/agnostic basis sets from the actual flattened column names; assert every column has exactly one basis. Canonicalise `box_name` with `casefold()` and preserve the complete cluster key for every retained row. Return counted attrition for missing-manual, missing-negative, and retained clusters.

- [ ] **Step 4: Verify the targeted tests pass**

Run the command from Step 2 and confirm the contract/basis/cohort subset passes.

- [ ] **Step 5: Commit the completed task**

```bash
git add scripts/ebm_diagnostic.py tests/test_ebm_diagnostic.py
git commit -m "feat: prepare validated EBM diagnostic cohorts"
```

### Task 2: EBM fitting and cluster-local maxT inference

**Files:**
- Modify: `scripts/ebm_diagnostic.py`
- Modify: `tests/test_ebm_diagnostic.py`

**Interfaces:**
- `fit_full_ebm(X, labels, columns, seed) -> FittedRung`
- `auc_group_kfold(X, labels, offers, columns, seed) -> float`
- `permute_labels_within_clusters(labels, clusters, rng) -> np.ndarray`
- `run_maxt(X, labels, clusters, columns, basis, seed, n_permutations) -> MaxTResult`
- `_build_findings(ablated_fit, maxt, X, records, columns, basis) -> list[dict]`

- [ ] **Step 1: Write failing diagnostics tests**

Use at least five synthetic offers with a complete manual/negative cluster per offer. Assert inverse-frequency sample weights, held-out-offer and full-fit AUC fields, preservation of label counts within each cluster, rejection of fewer than 200 permutations, maxT exclusion of scored terms, deterministic output for a fixed seed, and no findings from an underpowered prepared rung.

- [ ] **Step 2: Verify the tests fail for the missing fit/gate symbols**

Run:

```bash
PYTHONPATH=.venv-diagnostics/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -S -m pytest tests/test_ebm_diagnostic.py -m diagnostics \
  --strict-diagnostics-deps -k 'weights or fit or auc or permutation or maxt or findings' -v
```

Expected: import failure for `fit_full_ebm` or `run_maxt`.

- [ ] **Step 3: Implement the serial statistical engine**

Fit `ExplainableBoostingClassifier(interactions=0, random_state=seed)` with balanced sample weights. Use `GroupKFold` by offer for OOF probabilities and `roc_auc_score`; fit a separate full-data model for the in-sample statistic and importances. In every permutation retain the same prepared cohort and shuffle labels only inside each complete cluster. Compute the 95th percentile over the maximum `term_importances("avg_weight")` among parent/agnostic terms only. Include per-tier correlations against tier-specific value columns and a drop-value full-fit ablation. Parent group findings receive their raw-versus-capped Pearson metadata; parent tag terms remain unpromotable with an explicit deferred aggregate check.

- [ ] **Step 4: Verify the targeted tests pass**

Run the command from Step 2 and confirm the fitting/inference subset passes with the isolated stack.

- [ ] **Step 5: Commit the completed task**

```bash
git add scripts/ebm_diagnostic.py tests/test_ebm_diagnostic.py
git commit -m "feat: add clustered EBM permutation diagnostic"
```

### Task 3: CLI, results contract, and operator documentation

**Files:**
- Modify: `scripts/ebm_diagnostic.py`
- Modify: `tests/test_ebm_diagnostic.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- `run(features: Path, out: Path, seed: int, permutations: int, rungs: Sequence[str]) -> dict`
- `build_parser() -> argparse.ArgumentParser`

- [ ] **Step 1: Write failing diagnostics/CLI tests**

Assert default paths/options, invalid rung rejection, deterministic JSON serialization, all required provenance/methodology/rung fields, empty findings for a prepared cohort below 150 manual rows or 20 offers, and explicit caveats for deferred checks and value confounding.

- [ ] **Step 2: Verify the tests fail for missing CLI/output behavior**

Run:

```bash
PYTHONPATH=.venv-diagnostics/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -S -m pytest tests/test_ebm_diagnostic.py -m diagnostics \
  --strict-diagnostics-deps -k 'cli or test_main_wires or test_run or seed or findings_json or documented_isolated' -v
```

Expected: assertions fail because the result contract and CLI are incomplete.

- [ ] **Step 3: Implement the MVP CLI and JSON output**

Expose `--features`, `--seed`, `--permutations`, `--rungs`, `--out`, and `--no-plots`; default to all rungs and `diagnostics/ebm_findings.json`. Record `interpret`, `scikit-learn`, and `numpy` versions through `importlib.metadata`. Emit deterministic JSON with artifact provenance, cohort attrition, dynamic column counts, AUCs, maxT summaries, sorted findings, and the caveat that outputs are hypothesis-generating only. Document the isolated diagnostic command and the deferred MVP capabilities in `CLAUDE.md`.

- [ ] **Step 4: Verify the diagnostics suite passes**

Run:

```bash
PYTHONPATH=.venv-diagnostics/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -S -m pytest -m diagnostics --strict-diagnostics-deps \
  -W error::pytest.PytestUnhandledThreadExceptionWarning -rs
```

Expected: all collected diagnostics tests pass with no skips.

- [ ] **Step 5: Commit the completed task**

```bash
git add scripts/ebm_diagnostic.py tests/test_ebm_diagnostic.py CLAUDE.md
git commit -m "feat: expose EBM diagnostic findings CLI"
```

## Verification and handoff

```bash
PYTHONPATH=.venv-diagnostics/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -S -m pytest -m diagnostics --strict-diagnostics-deps \
  -W error::pytest.PytestUnhandledThreadExceptionWarning -rs
python3 scripts/generate_hard_negatives.py
PYTHONPATH=.venv-diagnostics/lib python3 -S scripts/ebm_diagnostic.py --no-plots
```

The full unmarked suite currently has four unrelated configuration/fixture failures in this worktree. Do not change them inside the EBM PR; report their fresh output separately.
