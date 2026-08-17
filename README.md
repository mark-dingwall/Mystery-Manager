# Mystery Manager

Allocates bulk produce overage into customer "mystery boxes" for a fruit & veggie box business. It reads weekly shopping-list spreadsheets and item/buyer data from the Laravel database, allocates overage with the canonical ILP optimiser, then produces a tab-delimited item-ID/quantity matrix for import back into the app.

## Setup

Install the runtime dependencies and pytest:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install pytest
```

For a new checkout, copy the tracked examples to their ignored runtime locations, then replace every placeholder with the real business values:

```bash
cp .env.example .env
cp identifiers.json.example identifiers.json
cp scoring_config.json.example scoring_config.json
cp queries.json.example queries.json
```

Box prices must be positive integer cents. The example scoring values and SQL are synthetic placeholders, not production configuration. Copy `tuning_bounds.json.example` to `tuning_bounds.json` only when running Optuna tuning.

## Quick start

```bash
python3 run.py                                                   # Textual operations UI
python3 run.py 123 offer_123_shopping_list.xlsx                 # weekly CLI (Rich review + LLM review)
python3 run.py 123 offer_123_shopping_list.xlsx --no-tui --no-llm # non-interactive quick run
python3 compare.py                                              # validate against the local Tier-A archive
python3 compare.py --all-strategies                             # canonical-vs-baseline benchmark
python3 web/app.py                                              # comparison UI on 127.0.0.1:5000
python3 -m pytest                                               # standard test suite
```

Offer `123` is a synthetic documentation example, not a real historical offer.
The comparison UI also needs a cleaned mystery CSV, its source XLSX, and database
access for the selected offer. Diagnostic tests use an isolated dependency stack;
see `CLAUDE.md` for the enforced command.

## Project structure

```text
Mystery-Manager/
├── run.py                       # Textual entry point and argument-mode weekly CLI
├── compare.py                   # Algorithm validation against historical allocations
├── allocator/                   # Allocation, scoring, I/O, TUI, and service modules
│   ├── allocator.py             # XLSX + DB → strategy → charity/stock result pipeline
│   ├── strategies/              # Canonical ILP strategy and regression baselines
│   ├── models.py                # Item, box, exclusion, and result models
│   ├── config.py                # Runtime configuration loaders
│   ├── db.py                    # Direct/SSH MySQL access and cached queries
│   ├── excel_io.py              # XLSX reader and tab-delimited serializer
│   ├── app.py                   # No-argument Textual application
│   ├── tui.py                   # Rich review used by argument-mode run.py
│   ├── screens/                 # Textual screens
│   └── services/                # Textual service layer
├── web/                         # Flask manual-vs-algorithm comparison UI
├── scripts/                     # Tuning, diagnostics, and maintenance utilities
├── tests/                       # pytest suite; no DB or network required
├── docs/                        # Local implementation/operational docs (gitignored)
├── historical/                 # Source workbooks (gitignored)
├── cleaned/                    # Canonical processed CSVs (gitignored)
├── cleaned_llm/                # Alternative LLM extraction outputs (gitignored)
├── mappings/                   # Name maps and LLM extraction caches (gitignored)
├── BACKLOG.md                  # Deferred work and known issues
├── CLAUDE.md                   # Detailed architecture, commands, and conventions
├── requirements.txt            # Runtime dependencies
└── requirements-diagnostics.txt # Isolated diagnostics/test dependencies
```

## Local configuration and data

These ignored files are required for normal allocation work:

- `.env` — DB connection, box pricing/target, charity settings, value-band parameters, and optional SSH settings. Bare `load_dotenv()` loads `.env`; `.env.local` is not loaded automatically.
- `identifiers.json` — donation/staff identifiers, standalone-name handling, charity keywords, and box-size overrides.
- `scoring_config.json` — category IDs, classification data, composite weights, allowances, and thresholds.
- `queries.json` — SQL adapted to the Laravel schema, using the aliases shown in the example.

Historical comparison work additionally needs `historical/`, `cleaned/`, and any required cached maps under `mappings/`. Weekly allocation needs the relevant `offer_*.xlsx` input. `tuning_bounds.json` is required only by the tuning script.

Tests set synthetic environment defaults and provision synthetic JSON configuration on a clean checkout. If ignored root configuration already exists, pytest reuses it.

See `CLAUDE.md` for the current command inventory, architecture, diagnostics contract, and project conventions.
