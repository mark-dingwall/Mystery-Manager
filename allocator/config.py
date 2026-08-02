"""Configuration for mystery box allocation."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Detect pack items (e.g. "Avocado - Hass (3 value pack)", "Lemons - 5 pack")
# and return the pack size so we can divide the DB price to get per-unit price.
_PACK_RE = re.compile(r"(\d+)\s*(?:value\s*)?pack", re.IGNORECASE)


def detect_pack_size(name: str) -> int:
    """Return the pack size from an item name, or 1 if not a pack."""
    m = _PACK_RE.search(name)
    return int(m.group(1)) if m else 1


# Box size tiers (all values in cents, loaded from .env)
def _load_box_tiers() -> dict:
    """Build BOX_TIERS from environment variables."""
    missing = [v for v in ("BOX_PRICE_SMALL", "BOX_PRICE_MEDIUM", "BOX_PRICE_LARGE")
               if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required pricing env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the real values."
        )
    target_pct = int(os.environ.get("BOX_TARGET_PCT", "115")) / 100
    tiers = {}
    for size, env_key in [("small", "BOX_PRICE_SMALL"),
                          ("medium", "BOX_PRICE_MEDIUM"),
                          ("large", "BOX_PRICE_LARGE")]:
        price = int(os.environ[env_key])
        tiers[size] = {"price": price, "target_value": round(price * target_pct)}
    return tiers

BOX_TIERS = _load_box_tiers()

# Hard ceiling: never exceed this % of target value
VALUE_CEILING_PCT = 1.30

# ---------------------------------------------------------------------------
# Scoring config (loaded from gitignored scoring_config.json)
# ---------------------------------------------------------------------------

def _load_scoring_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "scoring_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy scoring_config.json.example to scoring_config.json "
            "and fill in the real values."
        )
    with open(path) as f:
        return json.load(f)

_SCORING = _load_scoring_config()

CHEAP_ITEM_THRESHOLD = _SCORING["cheap_item_threshold"]
TARGET_ITEM_COUNTS = {k: tuple(v) for k, v in _SCORING["target_item_counts"].items()}
SCORING_WEIGHTS = _SCORING["scoring_weights"]
SLOT_DEGREE_THRESHOLD = _SCORING["slot_degree_threshold"]
MAX_SLOT_QTY = _SCORING["max_slot_qty"]
FUNGIBLE_GROUPS = {
    k: (v[0], v[1], v[2] if len(v) > 2 else "portioned")
    for k, v in _SCORING["fungible_groups"].items()
}

# ---------------------------------------------------------------------------
# PII-bearing identifiers (loaded from gitignored identifiers.json)
# ---------------------------------------------------------------------------

def _load_identifiers() -> dict:
    """Load identifier sets from identifiers.json at project root."""
    path = Path(__file__).resolve().parent.parent / "identifiers.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy identifiers.json.example to identifiers.json "
            "and fill in the real values."
        )
    with open(path) as f:
        return json.load(f)

_IDENTIFIERS = _load_identifiers()

# Non-customer boxes: donations and charity recipients (exclude from scoring)
DONATION_IDENTIFIERS = set(_IDENTIFIERS["donation_identifiers"])

# Backwards compat alias
CHARITY_IDENTIFIERS = DONATION_IDENTIFIERS

# Staff who self-pick from overs (not a standard mystery box, exclude from
# historical training data and algorithm auto-detection)
STAFF_IDENTIFIERS = set(_IDENTIFIERS["staff_identifiers"])

# Pre-account standalone names → email mappings for merged-box treatment
STANDALONE_NAME_TO_EMAIL = _IDENTIFIERS["standalone_name_to_email"]

# Explicit size overrides for non-standard box names
BOX_SIZE_OVERRIDES = _IDENTIFIERS.get("box_size_overrides", {})

# Charity keyword patterns for column matching in workbook fills
CHARITY_KEYWORDS = set(_IDENTIFIERS.get("charity_keywords", []))

# Per-offer size overrides (keyed by offer ID string)
PER_OFFER_BOX_SIZE_OVERRIDES = _IDENTIFIERS.get("per_offer_box_size_overrides", {})

# Special columns to skip in historical data
STOCK_IDENTIFIERS = {"Stock", "stock", "STOCK"}
BUFFER_IDENTIFIERS = {"Buffer", "buffer", "Volunteers", "volunteers"}
SUM_IDENTIFIERS = {"SUM", "Sum", "sum"}

# Additional skip-column patterns for older historical data
SKIP_COLUMN_IDENTIFIERS = {
    "Ov - Mys", "Ov-Mys", "Ov - mys", "Over Mys", "Overs", "Sum Mys",
    "Md Actual", "Staff Actual",
    "Total", "Unallocated",
    "Price Ea", "JS Price Ea", "JS Cost Ea", "Cost Ea", "RRP Ea",
    "Qty Sold", "Required Buy", "Overage", "Expected Purchase", "Expected Cost",
    "Pack Order", "Supplier #", "Supplier Name", "Supplier",
    "Qty (after mys)",
    "Buyer Notes", "buyer notes",
}

# Charity allocation target (loaded from .env)
CHARITY_NAME = os.environ.get("CHARITY_NAME", "Charity")
CHARITY_BASE_PERCENT = float(os.environ.get("CHARITY_BASE_PERCENT", "0.0"))
CHARITY_GIVING_MULTIPLIER = float(os.environ.get("CHARITY_GIVING_MULTIPLIER", "2.0"))

# Category IDs
CATEGORY_FRUIT = _SCORING["category_fruit"]
CATEGORY_VEGETABLES = _SCORING["category_vegetables"]

# Structured preference keywords
PREFERENCE_FRUIT_ONLY = _SCORING["preference_fruit_only"]
PREFERENCE_VEG_ONLY = _SCORING["preference_veg_only"]
PREFERENCE_MIX = _SCORING["preference_mix"]

# Diversity dimension weights
DIVERSITY_WEIGHTS = _SCORING["diversity_weights"]

# Item classifications: key -> (prefixes, sub_category, usage, colour, shape)
ITEM_CLASSIFICATIONS = {
    k: (v[0], v[1], v[2], v[3], v[4])
    for k, v in _SCORING["item_classifications"].items()
}

# ---------------------------------------------------------------------------
# Quantity classes and group allowances
# ---------------------------------------------------------------------------

QUANTITY_CLASSES = _SCORING["quantity_classes"]
GROUP_ALLOWANCES = _SCORING["group_allowances"]
# Price cut-points (cents) for usage->quantity-class refinement.
QTY_CLASS_PRICE_THRESHOLDS = _SCORING["qty_class_price_thresholds"]

# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

# Value penalty: power function parameters (env-overridable)
VALUE_SWEET_FROM = int(os.environ.get("VALUE_SWEET_FROM", "114"))
VALUE_SWEET_TO = int(os.environ.get("VALUE_SWEET_TO", "117"))
VALUE_PENALTY_EXPONENT = float(os.environ.get("VALUE_PENALTY_EXPONENT", "1.25"))

SAME_ITEM_MULTIPLIER = _SCORING["same_item_multiplier"]
GROUP_CONCENTRATION_MULTIPLIER = _SCORING["group_concentration_multiplier"]
GROUP_QTY_EXPONENT = _SCORING["group_qty_exponent"]
DIVERSITY_PENALTY_MULTIPLIER = _SCORING["diversity_penalty_multiplier"]
MAX_VALUE_SHARE_THRESHOLD = _SCORING["max_value_share_threshold"]
MAX_VALUE_SHARE_MULTIPLIER = _SCORING["max_value_share_multiplier"]
SIZE_FLOOR_TARGETS = _SCORING["size_floor_targets"]
SIZE_FLOOR_MULTIPLIER = _SCORING["size_floor_multiplier"]
PREF_VIOLATION_PENALTY = _SCORING["pref_violation_penalty"]
MAX_COMPOSITE_SCORE = _SCORING["max_composite_score"]

PACK_PRICE_TOLERANCE_CENTS = _SCORING["pack_price_tolerance_cents"]
DIVERSITY_FALLBACK_SCORE = _SCORING["diversity_fallback_score"]

# ---------------------------------------------------------------------------
# Top-up scorer
# ---------------------------------------------------------------------------

FUNGIBLE_NEUTRAL_SCORE = _SCORING["fungible_neutral_score"]
FUNGIBLE_NEW_GROUP_BONUS = _SCORING["fungible_new_group_bonus"]

# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------

TOPUP_MAX_PASSES = _SCORING["topup_max_passes"]
LOCAL_SEARCH_MAX_ITERATIONS = _SCORING["local_search_max_iterations"]
GREEDY_MAX_ROUNDS = _SCORING["greedy_max_rounds"]
ROUND_ROBIN_MAX_PASSES = _SCORING["round_robin_max_passes"]
MINMAX_MAX_PASSES = _SCORING["minmax_max_passes"]

ILP_HHI_BREAKPOINTS = _SCORING["ilp_hhi_breakpoints"]
ILP_COVERAGE_WEIGHT = _SCORING["ilp_coverage_weight"]
ILP_BALANCE_WEIGHT = _SCORING["ilp_balance_weight"]

# Fallback classification for unrecognized items (by category_id)
_cf = _SCORING["classification_fallback"]
CLASSIFICATION_FALLBACK = {
    CATEGORY_FRUIT:      tuple(_cf["fruit"]),
    CATEGORY_VEGETABLES: tuple(_cf["veg"]),
}

