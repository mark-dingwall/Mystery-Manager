# PR #1 Review Repairs Design

## Goal

Resolve the confirmed review findings in PR #1 without broadening its diagnostics-foundation scope or weakening the two-category feature-matrix contract.

## Dependency gating

`requirements-diagnostics.txt` is the single source of diagnostic distribution names and minimum versions. `tests/conftest.py` parses its simple `distribution>=version` lines and derives ordinary import names by replacing hyphens with underscores; Python retains only the exceptional `scikit-learn` → `sklearn` mapping. Unknown names passed to `require_dep()` remain import-only checks so its existing use with stdlib/test modules is unchanged.

Add `packaging>=22` because the gate directly uses `packaging.version.Version`. The `packaging.version` import remains lazy: ordinary pytest startup and collection do not import it. When a declared dependency needs a floor comparison, absence or a below-floor version of `packaging` follows the same non-strict-skip/strict-error policy as the requested dependency.

`pytest` and `packaging` are bootstrap dependencies rather than modules requested by diagnostic tests. When `--strict-diagnostics-deps` is enabled, `pytest_configure()` validates both distributions against the manifest before collection proceeds; this uses the lazily imported `packaging.version.Version`, validating `packaging` itself first and then the running pytest. Other declared distributions remain demand-checked by `require_dep()`. Thus the documented strict diagnostics command cannot pass under either bootstrap floor, while ordinary collection remains independent of the diagnostics stack.

Tests exercise parsing behavior from a temporary manifest rather than repeating the production floor dictionary. Subprocess tests cover an absent and below-floor `packaging`, a below-floor pytest in strict mode, and the existing non-strict skip versus strict error behavior.

## Configuration identity

The identity contracts represent the effective import-time-frozen bindings that actually produce a feature record. Live `allocator.config` and `allocator.categorizer` lookups are removed from the guards, but the existing shared allowance helper remains the owner of its three frozen inputs; it is not copied or widened for this repair:

| Inputs | Effective owner read by extraction/schema code and identity contracts |
|---|---|
| `BOX_TIERS`, `GROUP_ALLOWANCES`, `CATEGORY_FRUIT`, `CATEGORY_VEGETABLES`, `ITEM_CLASSIFICATIONS`, `CLASSIFICATION_FALLBACK`, `BOX_TARGET_PCT`, `VALUE_SWEET_FROM`, `VALUE_SWEET_TO`, `VALUE_PENALTY_EXPONENT` | direct import-time bindings in `allocator.box_features` |
| `DEFAULT_CLASSIFICATION` | the direct import-time binding in `allocator.box_features` used by `tag_vocabulary()` |
| `FUNGIBLE_GROUPS`, `QUANTITY_CLASSES`, `QTY_CLASS_PRICE_THRESHOLDS` | import-time bindings in `allocator.strategies._scoring`, which `_resolve_item_allowance_from_lookup()` actually reads |

`allocator.box_features` imports the `_scoring` module only to read those three effective bindings for hashing and snapshotting; this adds no DB path or new implementation. Extraction, vocabulary generation, flattening, hashing, and snapshotting therefore agree on the binding that owns each value, and post-import reassignment of `allocator.config` or `allocator.categorizer` cannot make a guard describe configuration the feature code did not observe.

`CATEGORY_FRUIT` and `CATEGORY_VEGETABLES` are added to both identity contracts because extraction uses them for category shares, preference violations, and schema validation. The snapshot grows from 12 to 14 explicit keys, and documentation and exact-contract tests change with it.

## Unsupported categories

`allocator.box_features` defines `UnsupportedCategoryError(ValueError)`. `extract_box_features()` raises that subclass when a resolved positive-quantity item has a category outside the configured fruit/vegetable pair. Existing direct callers and tests that treat the condition as `ValueError` remain compatible. This protects the invariant that the two category shares are complementary and allows `flatten()` to emit only fruit share.

The batch script catches only `UnsupportedCategoryError` at the manual-box and synthetic-box extraction boundaries. It prints a contextual `[SKIP]` message and continues processing later boxes and offers. Other exceptions, including an unrelated `ValueError`, still fail fast. A small wrapper centralizes this behavior so all six extraction call sites cannot drift.

## Documentation and cleanup

`CLAUDE.md` no longer instructs users to run the nonexistent `scripts/ebm_diagnostic.py`; it documents only commands currently present in the repository. Imports made obsolete by relocating extraction are removed from `scripts/extract_features.py`.

The authoritative parent design (`docs/superpowers/specs/2026-08-03-ebm-ordinal-diagnostics-design.md`) and foundation plan (`docs/superpowers/plans/2026-08-04-diagnostics-foundation.md`) are amended wherever they enumerate the dependency manifest or exact identity contracts. They add `packaging>=22` and its bootstrap role, change the hash from 11 to 13 named inputs by adding both category IDs, and change the snapshot from 12 to 14 keys by adding `category_fruit` and `category_vegetables`. Their example objects, key lists, test expectations, and prose counts change together. These companion documents live in the planning branch rather than the PR #1 worktree, so the repair implementation must not be considered documented until the corresponding planning-branch amendments land.

## Verification

Regression tests cover:

- manifest-derived dependency floors and the `scikit-learn` import-name exception;
- importing/collecting the ordinary suite without a top-level `packaging` import, including non-strict skip behavior when it is absent;
- strict startup failing clearly below the declared `packaging` or pytest floor;
- hash and snapshot changes when either category ID changes;
- post-import live config/categorizer reassignment not changing frozen feature or identity behavior;
- changing each allowance-driving `_scoring` binding changing both extracted `item_quantities` and the relevant identity contract;
- manual and synthetic unsupported-category records producing `[SKIP]` while subsequent records continue;
- an unrelated `ValueError` at the batch extraction boundary propagating instead of being skipped;
- the existing direct extractor `ValueError` invariant.

## Second review-loop hardening

The final review loop keeps the existing schema and dependency-gate design,
but closes five fail-open edges:

- `config_snapshot()` returns detached nested data. Mutating a returned stamp
  cannot alter the live scoring bindings or a previously captured stamp.
- the diagnostics manifest rejects duplicate canonical distribution names
  instead of allowing a later, weaker floor to replace an earlier floor;
- dependency floors always validate installed distribution metadata and also
  validate the version reported by the imported module when it exposes one;
- `flatten()` rejects a record whose tier is outside the fixed
  small/medium/large matrix schema instead of emitting three zero value slices;
- configuration loading rejects equal fruit and vegetable category IDs because
  the two-category share and preference contracts require distinct identities.

Full numeric validation is deliberately out of scope. The adversarial
non-finite reproductions pass floats where the public allocation contract
requires integers; production extraction converts CSV quantities and all price
sources to integers before calling the extractor. No production source path to
the reported state was found.

The follow-up review tightens those same boundaries without widening scope.
Plain manifest parsing retains the strongest floor when duplicate canonical or
import-module names appear; strict parsing rejects either collision. Installed
and running versions that cannot be parsed follow the ordinary skip/strict
error policy instead of escaping as parser exceptions. Category IDs must be
exact integers (not booleans, strings, or floats) as well as distinct, and an
isolated import test proves that validation remains connected to config loading.
An explicit module `__version__ = None` is malformed; only a genuinely absent
version attribute falls back to distribution metadata alone.

Run the focused box-feature tests first, then the complete pytest suite.
