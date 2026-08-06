# PR #1 Review Repairs Design

## Goal

Resolve the confirmed review findings in PR #1 without broadening its diagnostics-foundation scope or weakening the two-category feature-matrix contract.

## Dependency gating

`requirements-diagnostics.txt` is the single source of diagnostic distribution names and minimum versions. `tests/conftest.py` parses its simple `distribution>=version` lines and keeps only the exceptional distribution-to-module mapping (`scikit-learn` imports as `sklearn`) in Python. `packaging>=22` is declared because the gate directly uses `packaging.version.Version`, but that import occurs lazily only when a declared diagnostic dependency is checked. Ordinary pytest collection therefore does not import an undeclared package.

Tests exercise parsing behavior from a temporary manifest rather than repeating the production floor dictionary. Existing strict-versus-skip behavior remains unchanged.

## Configuration identity

`allocator.box_features` retains its documented import-time-frozen configuration model. Extraction, vocabulary generation, flattening, hashing, and snapshotting all read the same module-local bindings; live `allocator.config` and `allocator.categorizer` lookups are removed from the guards.

`CATEGORY_FRUIT` and `CATEGORY_VEGETABLES` are added to both identity contracts because extraction uses them for category shares, preference violations, and schema validation. The snapshot grows from 12 to 14 explicit keys, and documentation and exact-contract tests change with it.

## Unsupported categories

`extract_box_features()` continues to raise `ValueError` when a resolved positive-quantity item has a category outside the configured fruit/vegetable pair. This protects the invariant that the two category shares are complementary and allows `flatten()` to emit only fruit share.

The batch script catches only this `ValueError` at the manual-box and synthetic-box extraction boundaries. It prints a contextual `[SKIP]` message and continues processing later boxes and offers. Other exceptions still fail fast. A small wrapper centralizes this behavior so all six extraction call sites cannot drift.

## Documentation and cleanup

`CLAUDE.md` no longer instructs users to run the nonexistent `scripts/ebm_diagnostic.py`; it documents only commands currently present in the repository. Imports made obsolete by relocating extraction are removed from `scripts/extract_features.py`.

## Verification

Regression tests cover:

- manifest-derived dependency floors and the `scikit-learn` import-name exception;
- importing/collecting the ordinary suite without a top-level `packaging` import;
- hash and snapshot changes when either category ID changes;
- guards observing the same frozen bindings used by extraction and flattening;
- manual and synthetic unsupported-category records producing `[SKIP]` while subsequent records continue;
- the existing direct extractor `ValueError` invariant.

Run the focused box-feature tests first, then the complete pytest suite.
