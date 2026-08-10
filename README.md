# Mystery Manager

Allocates bulk produce overage into customer "mystery boxes" for a fruit & veggie box business. Reads weekly shopping-list spreadsheets and item/buyer data from the Laravel DB, allocates overage to mystery boxes using an ILP optimiser, then outputs tab-delimited box assignments for import back into the app.

## Quick start

```bash
python3 run.py 106 offer_106_shopping_list.xlsx                   # full run (TUI + LLM review)
python3 run.py 106 offer_106_shopping_list.xlsx --no-tui --no-llm # quick run
python3 compare.py                                                # validate against 45 Tier-A historical offers
python3 compare.py --all-strategies                               # strategy benchmark
python3 -m pytest                                                 # run test suite
```

## Project structure

```
Mystery-Manager/
├── run.py                  # Weekly allocation entry point
├── compare.py              # Algorithm validation against historical data
├── pyproject.toml          # pytest config
├── allocator/              # Core library
│   ├── allocator.py        #   Pipeline: XLSX + DB → strategy → output
│   ├── strategies/         #   Canonical ILP strategy + benchmark baselines
│   ├── models.py           #   Item, MysteryBox, CharityBox, AllocationResult
│   ├── config.py           #   Tiers, weights, scoring constants, identifiers
│   ├── db.py               #   SSH tunnel + MySQL queries
│   ├── excel_io.py         #   XLSX reader + tab-delimited writer
│   ├── app.py              #   Textual TUI application
│   ├── screens/            #   TUI screen modules (wizard, review, history, etc.)
│   ├── services/           #   Service layer (allocation, comparison, history, DB)
│   ├── tui.py              #   Legacy Rich interactive review UI
│   ├── clean_history.py    #   Historical XLSX → CSV pipeline
│   ├── fill_workbook.py    #   Write strategy results into XLSX
│   └── benchmark_extraction.py  # LLM extraction benchmarks
├── tests/                  # pytest suite (synthetic fixtures; no DB required)
│   ├── conftest.py         #   Config bootstrap + factory fixtures
│   ├── fixtures/           #   Synthetic identifiers + scoring config
│   └── test_*.py           #   test modules
├── scripts/                # Maintenance utilities
│   ├── score_offer.py      #   Per-offer strategy benchmark
│   ├── diagnose_scoring.py #   Penalty breakdown diagnostics
│   ├── validate_cleaned.py #   Structural + DB checks on cleaned CSVs
│   ├── validate_prices.py  #   XLSX vs DB price validation
│   ├── standardize_filenames.py  # Canonical XLSX filenames
│   ├── compare_llm_outputs.py   # Side-by-side LLM extraction comparison
│   └── analyze_offer_values.py  # Per-offer value targets by size tier
├── docs/                   # Design docs (gitignored)
├── historical/             # Source XLSX files (gitignored)
├── cleaned/                # Processed CSVs (gitignored)
├── mappings/               # Cached LLM name maps (gitignored)
├── CLAUDE.md               # Full architecture and conventions
└── requirements.txt
```

## Gitignored items a new user needs to provide

**Secrets / config**
- `.env` / `.env.local` — DB credentials, SSH tunnel config, pricing params (box target %, penalty weights)
- `identifiers.json` — customer/donor/staff email lists (see `identifiers.json.example`)

**Business data**
- `historical/` and `cleaned/` — historical shopping lists and processed CSVs used by `compare.py`
- `offer_*.xlsx` — weekly input files (overage spreadsheets)
- `mappings/` — cached item-name-to-DB-ID mappings for older offers without ID columns

See `CLAUDE.md` for full architecture, commands, and conventions.
