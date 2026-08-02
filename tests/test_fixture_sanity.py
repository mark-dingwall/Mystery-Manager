"""Guard against tracked files mirroring the real (gitignored) scoring config.

The production ``scoring_config.json`` is gitignored — it holds tuned penalty
weights and business-intelligence values. ``tests/fixtures/scoring_config.json``
is force-tracked (via ``!`` in ``.gitignore``) for CI portability, and
``scoring_config.json.example`` documents the schema. Both MUST hold synthetic
values only. LLMs have repeatedly repopulated them with the real values copied
from production; this test fails loudly when that happens. It skips where the
real config is absent (e.g. CI), so it only guards on machines that have it.
"""

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL = _REPO_ROOT / "scoring_config.json"
_TRACKED = [
    _REPO_ROOT / "tests" / "fixtures" / "scoring_config.json",
    _REPO_ROOT / "scoring_config.json.example",
]

# Keys whose VALUES are tuned or business intelligence and must never match
# production. (Iteration caps, max_composite_score, tolerances may match — they
# are generic/methodology, not sensitive.)
_SENSITIVE_KEYS = [
    "same_item_multiplier",
    "group_concentration_multiplier",
    "group_qty_exponent",
    "diversity_penalty_multiplier",
    "max_value_share_threshold",
    "max_value_share_multiplier",
    "size_floor_multiplier",
    "pref_violation_penalty",
    "diversity_weights",
    "quantity_classes",
    "group_allowances",
    "size_floor_targets",
    "target_item_counts",
    "max_slot_qty",
    "scoring_weights",
    "slot_degree_threshold",
    "qty_class_price_thresholds",
    "cheap_item_threshold",
    "category_fruit",
    "category_vegetables",
    "ilp_coverage_weight",
    "ilp_balance_weight",
]


@pytest.mark.parametrize("tracked", _TRACKED, ids=lambda p: p.name)
def test_tracked_file_does_not_mirror_real_config(tracked):
    if not _REAL.exists():
        pytest.skip("real scoring_config.json absent — nothing to compare against")
    real = json.loads(_REAL.read_text())
    public = json.loads(tracked.read_text())
    leaked = [
        k for k in _SENSITIVE_KEYS if k in real and k in public and real[k] == public[k]
    ]
    assert not leaked, (
        f"{tracked.name} mirrors REAL production values for sensitive keys {leaked} "
        "— replace with synthetic values (see memory/feedback_opensource_sensitivity.md)"
    )
