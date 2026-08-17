# Mystery Manager

We run a fruit & veggie box business. Customers place orders, we buy bulk boxes from farmers, then split & pack produce. This project is a Python application that allocates fruit & veggie items from bulk purchase overage to "mystery" boxes bought by customers each week. Mystery boxes are better value than buying piecemeal, but without the latter's flexibility.

## Running

### Primary tools (root)

```bash
python3 run.py                                                     # Textual operations UI
python3 run.py <offer_id> <shopping_list.xlsx>                    # weekly CLI (Rich review + LLM review)
python3 run.py <offer_id> <xlsx> --no-tui --no-llm               # quick run
python3 run.py <offer_id> <xlsx> --no-tui --no-llm -v            # enable DEBUG logging
python3 run.py <offer_id> <xlsx> --no-tui --no-llm --algorithm deal-topup

python3 compare.py                                                # validate against the local Tier-A archive
python3 compare.py --algorithm deal-topup                         # specific strategy
python3 compare.py --only-offers 120-123                          # illustrative subset
python3 compare.py --all-strategies                               # full benchmark (canonical vs baselines)
python3 compare.py --detail                                       # per-offer breakdown + detailed JSON
python3 compare.py --csv                                          # implies --detail; writes manual + algorithm CSVs
python3 compare.py --workers 4                                    # override CPU-count parallelism
python3 compare.py --sequential                                   # disable parallelism

python3 web/app.py                                                # comparison web app (localhost:5000)
python3 web/app.py --port 8080                                    # custom port
```

Offer IDs 120–123 used in this document are synthetic examples and do not
describe the private historical archive.

Argument-mode `run.py` also supports `--parse-notes`, repeated `--charity NAME`,
`--charity-target DOLLARS`, and `--output FILE`; see `--help` for the complete
contract. `compare.py --all-strategies` prints the leaderboard, including manual,
and returns without producing single-strategy detail/CSV artifacts.

The comparison web app requires a cleaned mystery CSV, its source XLSX, and DB
connectivity for the selected offer.

### Library tools (allocator/)

```bash
python3 allocator/clean_history.py                                # clean historical XLSX → CSVs
python3 allocator/clean_history.py --no-older                     # historical/ only
python3 allocator/clean_history.py --llm-extract                  # LLM extraction (run outside Claude Code)
python3 allocator/clean_history.py --llm-extract --llm-method sonnet-low
python3 -m allocator.fill_workbook 123 offer_123_shopping_list.xlsx   # synthetic example; modifies XLSX in place
python3 allocator/benchmark_extraction.py 5                       # benchmark LLM extraction (outside Claude Code)
```

### Tests

```bash
python3 -m pytest                                                 # run full suite
python3 -m pytest tests/test_strategies.py -v                     # single module, verbose
python3 -m pytest -k "test_value_penalty"                         # run by name pattern
```

Tests require no DB. They set synthetic environment defaults and provision
synthetic root JSON only when ignored local configuration is absent; existing
local configuration takes precedence. See `tests/conftest.py` for the bootstrap
and factory fixtures. `tests/fixtures/desirability_items.csv` provides isolated
desirability data.

### Diagnostics (isolated dependency stack)

```bash
python3 -m venv .venv-diagnostics
.venv-diagnostics/bin/python -m pip install -r requirements.txt -r requirements-diagnostics.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-diagnostics/bin/python -m pytest -m diagnostics --strict-diagnostics-deps -W error::pytest.PytestUnhandledThreadExceptionWarning -rs
```

Use the venv form on hosts with `ensurepip`. On the current host,
`python3.10-venv` is unavailable, so use the hermetic `--target` fallback below.
Python's `-S` flag excludes user and global site-packages, while
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` prevents host-installed pytest plugins from
entering the run.

```bash
python3 -m pip install --target .venv-diagnostics/lib -r requirements.txt -r requirements-diagnostics.txt
PYTHONPATH=.venv-diagnostics/lib PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -S -m pytest -m diagnostics --strict-diagnostics-deps -W error::pytest.PytestUnhandledThreadExceptionWarning -rs
```

`pytest>=9.0.0` and `packaging>=22` are bootstrap dependencies for diagnostic
tests, not production runtime dependencies. Under
`--strict-diagnostics-deps`, pytest validates `packaging` and the running
pytest's `__version__` against those floors before collection. Other diagnostic
libraries remain demand-checked by `require_dep()` when their test modules load.

**Diagnostic tests run under their own command, which is the enforcement point:**

The checked-in diagnostics dependency test demand-checks every non-bootstrap
distribution in the manifest at module scope. Under strict mode, a missing or
below-floor manifest dependency fails collection instead of skipping. Skips for
unrelated reasons retain normal pytest semantics.

Plain `python3 -m pytest` deliberately does **not** enforce this: without the
isolated stack installed, every diagnostics module skips and the plain command
still reports green. The `-m diagnostics` expression selects the diagnostic
tests only; `--strict-diagnostics-deps` independently arms `require_dep()` so a
missing or below-floor dependency raises instead of skipping. Malformed
bootstrap requirements, or missing or below-floor bootstrap dependencies, fail
startup with a concise pytest usage error. Exit code 5 (`EXIT_NOTESTSCOLLECTED`)
is also a failure of this command — it means the diagnostics suite collected
nothing, which is the disappearance being guarded against.

### Utility scripts (scripts/)

```bash
python3 scripts/diagnose_scoring.py --no-plots                    # penalty breakdown diagnostics
python3 scripts/validate_cleaned.py                               # structural + DB checks on cleaned CSVs
python3 scripts/validate_cleaned.py --no-db                       # offline structural checks only
python3 scripts/validate_cleaned.py --only-offers 120-123         # illustrative subset
python3 scripts/validate_prices.py --offers 120,123                # XLSX vs DB price validation
python3 scripts/standardize_filenames.py                          # dry-run filename normalization
python3 scripts/standardize_filenames.py --apply                  # apply renames
python3 scripts/compare_llm_outputs.py                            # side-by-side LLM extraction comparison
python3 scripts/analyze_offer_values.py                           # per-offer value targets by size tier
python3 scripts/analyze_offer_values.py --only-offers 120-123     # illustrative subset
python3 scripts/analyze_desirability.py                           # item desirability from packing history
python3 scripts/analyze_desirability.py --csv --no-plots          # export CSV, skip visuals
python3 scripts/extract_features.py                               # extract box features for tuning (needs DB)
python3 scripts/extract_features.py --only-offers 120-123         # illustrative subset
python3 scripts/extract_features.py --no-synthetics               # manual boxes only
python3 scripts/generate_hard_negatives.py                       # Tier-A EBM input; needs DB
python3 scripts/generate_hard_negatives.py --only-offers 120-121 # illustrative smoke test
PYTHONPATH=.venv-diagnostics/lib python3 -S scripts/ebm_diagnostic.py --no-plots    # DB-free EBM hypotheses
python3 scripts/tune_scoring.py                                   # parameter tuning (needs features JSON)
python3 scripts/tune_scoring.py --trials 200 --folds 3            # quick run
python3 scripts/tune_scoring.py --trials 3000 --repeats 25        # overnight stability run
python3 scripts/tune_scoring.py --features path/to/features.json  # custom features file
python3 scripts/generate_survey_scenarios.py                      # generate packer survey scenarios (needs DB)
python3 scripts/generate_survey_scenarios.py --seed 42            # reproducible
python3 scripts/process_survey_results.py responses.json          # analyze survey responses vs scoring function
```

`scripts/score_offer.py` is currently stale against the seven-component score
contract; use `compare.py --only-offers <id> --all-strategies` for per-offer
benchmarking until the legacy report is repaired. `analyze_desirability.py`
requires NumPy and SciPy, which are not fully represented by the checked-in
dependency manifests.

`hard_negatives.json` is atomically replaced only after generation validates the
required sources and the manual-vs-synthetic, manual-vs-baseline, and
manual-vs-ILP rungs (at least 150 manual boxes across 20 offers per rung).
Generation does not run EBM inference. Inspect
`diagnostics/hard_negatives_report.json` after any non-zero generation run.

`ebm_diagnostic.py` consumes only the validated hard-negative artifact and
checks its feature-schema, feature-config, and roster hashes before fitting. It reports
class-balanced, offer-held-out AUC separately from full-fit/maxT statistics, and
never adjusts scoring or the ILP. A rung below 150 manual boxes or 20 offers is
reported as underpowered with no findings. Findings are hypotheses only: value
confounding is caveated through a drop-value refit, and tag-parent promotion,
plots, interactions, multi-seed stability, parallel permutations, and
leave-one-source-out ablations remain deliberately deferred.
`--no-plots` is accepted for compatibility; the current diagnostic emits JSON,
not plots.

`compare.py` is the primary validation tool — it compares algorithm output against cleaned historical CSVs and prints per-box and aggregate metrics with a composite score. Default run uses Tier A offers only; use `--only-offers` for others.

## Project Direction

The committed allocation direction is **Optuna-informed ILP**, with a
hypothesis-generating EBM diagnostic. Tuning writes a candidate parameter report;
applying it is a separate reviewed update to `.env` / `scoring_config.json`. It
does not update runtime configuration or the ILP automatically.

Directional allocator code uses three states:

- **canonical** — the committed direction: `ilp-optimal`, shared scoring/infra, and
  the tuning/diagnostic pipeline. Extend and maintain here.
- **baseline** — runnable regression benchmarks the canonical model must beat
  (`deal-topup`, `greedy-best-fit`, `round-robin`, `minmax-deficit`, `discard-worst`,
  `local-search`). Not a production direction and not to be extended.
  `local-search` is also the ILP fallback (load-bearing).
- **superseded** — replaced as a direction, though a compatibility path may still
  use it (the argument-mode CLI still uses `allocator/tui.py` for Rich review).

Standalone maintenance utilities may instead be marked `# STATUS: dev-tool`.
These headers identify intentionally non-weekly or non-canonical code; they are
not an exhaustive classification of every module.

`ilp-optimal` is the default everywhere. The Textual weekly-allocation wizard
presents it alone; argument-mode `run.py` retains an expert `--algorithm <name>`
override. Baselines remain runnable through explicit overrides and comparison
surfaces such as `compare.py`, workbook fills, and the Flask UI.

## Architecture

Allocation framework: one canonical strategy (ilp-optimal) plus runnable baselines, over shared infrastructure. See § Project Direction.

**Data flow:** XLSX overage + DB items/buyers → `AllocationResult` → strategy fills boxes → charity allocation → stock. Tab-delimited serialization happens afterwards through `excel_io.format_output()`.

### Pipeline (in `allocator/allocator.py`)

```
allocate()
  ├── shared: build_items, build_boxes, create AllocationResult
  ├── optional: apply bootstrap_allocations (pre-fill boxes from prior run)
  ├── STRATEGY(result)          ← canonical strategy or a baseline (fills box.allocations in place)
  ├── shared: _allocate_charity() (when recipients are configured)
  └── shared: remaining → stock
```

### Strategies (in `allocator/strategies/`)

A strategy is a callable `(AllocationResult) -> None`. Strategies are registered in `allocator/strategies/__init__.py` and lazy-loaded to avoid circular imports.

**`ilp-optimal`** (canonical, default) — ILP-based multi-objective optimiser via PuLP/HiGHS. Solves a linearised surrogate of the scalar composite evaluation (see `docs/SCORING.md`, gitignored). Falls back to local-search if PuLP is missing or the solver/model fails.

**`local-search`** — Bootstraps from discard-worst, then iteratively relocates and swaps items between boxes to minimise composite penalty.

**`discard-worst`** — Subtractive. Seeds all boxes via greedy draft, then trims items whose removal most reduces penalty.

**`deal-topup`** (baseline) — Additive. Deals items round-robin to all eligible boxes, then tops up under-target boxes.

**`greedy-best-fit`**, **`round-robin`**, **`minmax-deficit`** — Simpler additive strategies. See module docstrings.

Charity allocation (remaining overage to charity toward computed target, then stock) is shared infrastructure, not part of any strategy.

To add a baseline benchmark: create `allocator/strategies/my_strat.py` with a `run(result)` function, then register it in `_REGISTRY` in `allocator/strategies/__init__.py`. Select it with `--algorithm my-strat`.

### Key modules (`allocator/`)

- **`strategies/`** — pluggable allocation strategies. `__init__.py` has the registry; `ilp_optimal.py` is the canonical strategy; `deal_topup.py` is a baseline. `_scoring.py` holds scalar penalty functions used by strategies; `compare.py` reuses selected helpers but applies its own final scalar evaluation. `_helpers.py` has shared constraint checks and diversity scoring.
- **`tuning.py`** — re-scoring module for parameter tuning. Precomputed box features + params dict → composite score. It has no top-level DB/config imports or import-time side effects; `default_params()` lazily snapshots current config for parity checks. Used by `scripts/tune_scoring.py`.
- **`box_features.py`** — box-feature extraction (relocated from `scripts/extract_features.py`). Hard-negative/EBM artifacts enforce schema, config-hash, roster-hash, and flatten-column consistency. `config_snapshot()` is a tested provenance helper, not currently emitted or checked by survey tooling. It has no DB imports, but it imports runtime configuration at module load, so required local configuration must be present and its values are frozen for the process.
- **`models.py`** — `Item`, `MysteryBox`, `CharityBox`, `AllocationResult`, `ExclusionRule`. `AllocationResult.solver_status` records ILP solution/fallback evidence. Persisted prices are integer cents.
- **`config.py`** — tier definitions (from `.env`), identifier sets (from `identifiers.json`), scoring/classification config (from `scoring_config.json`, gitignored). Exposes `BOX_TIERS`, `FUNGIBLE_GROUPS`, `ITEM_CLASSIFICATIONS`, and composite scoring constants (full model in `docs/SCORING.md`).
- **`desirability.py`** — loads item desirability scores from `diagnostics/desirability_items.csv`, applies Bayesian shrinkage, and normalises to [0,1] for survey scenario construction. It is not a production scoring dimension. The CSV is produced only by `scripts/analyze_desirability.py --csv`; that analysis script does not import this module.
- **`scorer.py`** — deal-topup specific scoring. `prioritize_items_for_deal()` sorts items for deal phase; `score_topup_candidate()` scores top-up additions with hard constraints and soft scoring.
- **`db.py`** — SSH tunnel (via paramiko) to MySQL DB. Singleton `TunnelManager` with reference counting; the tunnel remains alive until process exit. For direct connections, `DB_SOCKET` overrides `DB_HOST` / `DB_PORT`; SSH connections always use the local tunnel. All query functions are `@functools.cache`-decorated for within-run deduplication. SQL is loaded from `queries.json` (gitignored).
- **`excel_io.py`** — reads `ID` + `Overage` columns from XLSX; writes tab-delimited output for import.
- **`categorizer.py`** — assigns fungible groups and diversity classifications (sub-category, usage, colour, shape) by item name prefix matching.
- **`app.py`** — Textual application launched by `python3 run.py` with no arguments. Implements a 5-section main menu with DB status badge and section screens.
- **`screens/`** — TUI screen modules: wizard (early steps, box review, progress/results), strategy comparison, historical data, clean history, glossary, and help overlay.
- **`services/`** — service layer for the TUI: allocation, comparison, historical data, clean history, and DB connectivity services.
- **`tui.py`** — legacy Rich box-review UI, retained and used by argument-mode `run.py` unless `--no-tui` is supplied. Its strategy score-breakdown view is currently stale; see `BACKLOG.md`.
- **`llm_review.py`** — optional Claude CLI integration for note parsing and post-allocation review.
- **`clean_history.py`** — multi-tier historical data processing across Tiers A–D from `historical/` and `historical/older/`. Discovers files, selects sheets, detects transposed layouts, and classifies columns via `box_parser.py`. Historical layouts without IDs—including Tier C and older Tier-D sheets—use `name_matcher.py`. `--llm-extract` runs a selectable `benchmark_extraction.STRATEGY_RUNNERS` method (default `haiku-whole`) for non-standard Tier C/D workbooks. Method outputs go to `cleaned_llm/{method}/`; those alternatives are not read by `compare.py`. `mappings/` holds both name-to-ID maps and extraction caches.
- **`fill_workbook.py`** — runs all strategies and modifies the supplied XLSX in place by copying worksheet index 1 into result sheets. Copy the workbook first. Re-running against a workbook that already has strategy sheets is not idempotent. The legacy Rich TUI imports this command; the Textual app does not.
- **`benchmark_extraction.py`** — benchmarks LLM extraction strategies for non-standard historical workbooks. Must be run outside Claude Code.
- **`sheet_analyzer.py`** — legacy single-Sonnet extractor and shared prompt helper for non-standard historical offers. It owns the unsuffixed legacy cache; current `--llm-extract` execution uses benchmark strategy runners and method-suffixed caches.
- **`box_parser.py`** — parses box column headers across all historical naming conventions (`?Sm Name`, `(?) Lg Name`, `Size - Name`, `M Box N`, `Lge Charity`, etc.) into `(cleaned_name, size_tier, box_type)`.
- **`name_matcher.py`** — item name → DB ID matching for historical layouts without IDs. It loads cache first, tries live and then soft-deleted DB parts, performs exact/prefix matching, and uses Claude CLI (Haiku) for unresolved names. When the DB is unavailable, only an existing nonempty cache can provide fallback.
- **`claude_cli.py`** — subprocess wrapper for `claude -p` CLI calls.

### Packer survey tool (cross-repo)

Livewire v2 component in the Jointly.Shop codebase (`app/Http/Livewire/PackerSurvey.php`), admin-only. Scenarios generated here (`scripts/generate_survey_scenarios.py`), JSON manually copied to Jointly.Shop `storage/app/survey/`. Responses exported as JSON from the UI, processed here (`scripts/process_survey_results.py`). See `docs/OPTIMISATION_PLAN.md` Phase 3b.

### Web app (`web/`)

Flask app for side-by-side comparison of algorithm vs manual packing at the box level.

- **`app.py`** — Flask application and routes. Landing page (offer/algorithm selector) and comparison view.
- **`comparison.py`** — `build_comparison_data()` bridges `compare.py` functions to templates. `compute_item_diff()` and `build_box_pairs()` are pure functions for box matching and item-level diffs.
- **`templates/`** — Jinja2: `base.html` (shell), `index.html` (selector), `compare.html` (side-by-side cards with colour-coded metrics and item diffs).
- **`static/`** — `style.css` (grid layout, colour coding), `compare.js` (expand/collapse unchanged items, sort box cards).

### Tests (`tests/`)

Tests cover models/config, categorization/scoring, desirability/tuning/box features, hard-negative diagnostics, strategies, the allocator pipeline, parsing/I/O, wizard helpers and services, survey scenarios, and web comparison. They require no DB or network. Tests set synthetic environment defaults and provision synthetic JSON files only when ignored root configuration is absent; an existing local `scoring_config.json` / `identifiers.json` is reused.

- **`conftest.py`** — test config bootstrap (sets env vars before allocator import), factory fixtures for Item/MysteryBox/CharityBox/AllocationResult.
- **`tests/fixtures/`** — synthetic `identifiers.json` and `scoring_config.json` for CI portability.

### Utility scripts (`scripts/`)

- **`score_offer.py`** — legacy single-offer benchmark. Its report still expects removed score keys and is not operational against the current seven-component composite; use `compare.py --only-offers <id> --all-strategies` until the backlog repair lands.
- **`diagnose_scoring.py`** — penalty breakdowns, pricing anomaly detection, and visualisations across all historical tiers.
- **`validate_cleaned.py`** — structural integrity, DB consistency, and cross-file checks on cleaned CSVs. (underlying library for `HistoricalService` — validate_cleaned logic is now accessible via the TUI Historical Data screen via `python3 run.py` → Historical Data → Validate All. Keep as standalone tool for direct CLI use.)
- **`validate_prices.py`** — SUMPRODUCT validation comparing XLSX prices against DB prices.
- **`standardize_filenames.py`** — renames historical XLSX files to canonical `offer_{N}_shopping_list.xlsx` format. (dev tool — infrequent use, kept as standalone)
- **`compare_llm_outputs.py`** — side-by-side comparison of LLM extraction methods with Jaccard similarity and optional Claude investigation. (dev tool — infrequent use, kept as standalone)
- **`analyze_offer_values.py`** — per-offer, per-size-tier average box values. Writes `diagnostics/offer_value_targets.json` for training data.
- **`analyze_desirability.py`** — per-item desirability analysis from historical packing decisions. OLS regression + distribution stats; requires NumPy and SciPy. Writes `diagnostics/desirability_items.csv` only with `--csv`.
- **`extract_features.py`** — extracts precomputed box features from historical CSVs + DB, generates synthetic bad boxes, writes `diagnostics/tuning_features.json`. Requires DB connection.
- **`tune_scoring.py`** — parameter tuning over precomputed features JSON (no DB needed). Writes candidate parameters to `diagnostics/tuning_results.json`; application to live config is manual and reviewed. Local diagnostics artifacts are unversioned and must be regenerated after scoring-contract changes.
- **`ebm_diagnostic.py`** — provenance-guarded, DB-free EBM diagnostic over hard negatives. Writes `diagnostics/ebm_findings.json`; results never change scoring automatically.
- **`generate_survey_scenarios.py`** — constructs packer survey scenarios from historical boxes + overage. Tier 1 (random calibration) + Tier 2 (dimension-targeted). Writes `diagnostics/survey_scenarios.json`. Requires DB.
- **`process_survey_results.py`** — analyses survey responses against the scoring function. Writes `diagnostics/survey_analysis.json`.

## Database gotchas

- **Always filter soft deletes**: the relevant tables have a soft-delete column. Every query joining these tables must filter for non-deleted records — **except** `fetch_offer_parts_by_name(include_deleted=True)` which is used for historical name matching on older offers where parts are soft-deleted but prices are still valid.
- **User names are encrypted**. Use `email` as the identifier, never name fields.
- Persisted DB, model, configuration, and allocation values use integer cents.
  Spreadsheet price cells and `--charity-target` are dollar inputs converted at
  the boundary. Import output contains item IDs and quantities, not prices.
- SSH key path and connection config in `.env` (set `SSH_ENABLED=true` to use tunnel).
- **SQL queries** are loaded from `queries.json` (gitignored). See `queries.json.example` for the expected structure and column aliases.

## Conventions

- **Scoring model — single source of truth: `docs/SCORING.md`** (gitignored; full model, formulas, and what was dropped). `docs/OPTIMISATION_PLAN.md` is the roadmap/history. Code is ground truth: `allocator/strategies/_scoring.py` + `compare.py`.
- Concentration is penalised in two complementary layers (same-item and fungible-group), with a hard allocation ceiling enforced in `_helpers.py` (not a scoring cap). Full mechanism, allowances, and formulas: `docs/SCORING.md` (gitignored).
- Merged boxes (emails) get added to the customer's existing order. Standalone boxes (`?Name` prefix in output) ship separately.
- `BOX_TIERS` target values are `BOX_TARGET_PCT`% of price (configured in `.env`).
- Category IDs are configured in `scoring_config.json`.

## Strategy Benchmark (canonical vs baselines)

The current private benchmark supports `ilp-optimal` as the production choice;
the other strategies are regression baselines. Refresh the local result by
running `python3 compare.py --all-strategies`.

Rank order: ilp-optimal > local-search > discard-worst > round-robin >
greedy-best-fit ≈ manual > deal-topup > minmax-deficit. (`greedy-best-fit` and
`manual` sit within noise of each other — treat as tied, not ranked.)

This ordering is specific to the private benchmark and current configuration;
recalculate it after scoring or tuning changes. Except for `local-search`, which
remains the ILP fallback, the baselines are scheduled for removal after the
planned Optuna and EBM work is integrated.

Score = 100 minus composite penalties (full breakdown in `docs/SCORING.md`, gitignored).

## Historical data tiers

| Tier | Typical layout | IDs available? | Source dir | Handling |
|------|----------------|----------------|------------|----------|
| A | Current standard | Yes | `historical/` | Default algorithm-comparison cohort |
| B | Earlier standard | Yes | `historical/older/` | Mostly standalone boxes |
| C | Name-based | No | `historical/older/` | Programmatic extraction plus cached name matching |
| D | Sparse or irregular | Varies | `historical/older/` | Per-layout overrides and optional LLM extraction |

These tiers describe capabilities, not public archive contents. The cleaner and
historical browser use archive-quality labels, while comparison and validation
retain a legacy analytical split. Use `--only-offers` with locally discovered
IDs when a task needs an explicit cohort. Historical name matching can use
soft-deleted DB records; without DB access, it requires an existing nonempty
cache.

Archive counts and offer identifiers are intentionally kept out of public
documentation. Use the comparison and historical-data tools to inspect the
private local archive.
