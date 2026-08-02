"""
Main allocation orchestrator.

Shared infrastructure (data loading, box building, charity, output) lives here.
The box-filling strategy is pluggable — see allocator/strategies/.
"""

import json
import logging
from pathlib import Path

from allocator.categorizer import assign_classification, assign_fungible_group, category_name
from allocator.config import (
    BOX_TIERS,
    CHARITY_BASE_PERCENT,
    CHARITY_GIVING_MULTIPLIER,
    CHARITY_NAME,
    DONATION_IDENTIFIERS,
    STAFF_IDENTIFIERS,
    PREFERENCE_FRUIT_ONLY,
    PREFERENCE_VEG_ONLY,
    detect_pack_size,
)
from allocator.db import (
    fetch_buyer_existing_categories,
    fetch_categories,
    fetch_customer_giving,
    fetch_mystery_box_buyers,
    fetch_offer_gross_retail,
    fetch_offer_items,
)
from allocator.excel_io import read_overage_from_xlsx
from allocator.models import (
    AllocationResult,
    CharityBox,
    ExclusionRule,
    Item,
    MysteryBox,
)
from allocator.strategies import DEFAULT_STRATEGY, get_strategy

logger = logging.getLogger(__name__)


def infer_tier_from_name(name: str) -> str | None:
    """Infer box tier from offer_part name like 'A medium mystery box!'.

    Uses simple substring matching because the size keyword appears in the
    middle of the string (not as a prefix/suffix), so parse_box_name() would
    return None for all these names.

    Returns None if no tier keyword is found.
    """
    lower = name.lower()
    if "small" in lower:
        return "small"
    if "medium" in lower:
        return "medium"
    if "large" in lower:
        return "large"
    logger.warning("Cannot infer tier from offer_part name '%s' — skipping buyer", name)
    return None


def parse_preference(selected_option: str | None) -> str | None:
    """Convert structured buy option to internal preference key."""
    if not selected_option:
        return None
    if "no veg" in selected_option.lower() or selected_option == PREFERENCE_FRUIT_ONLY:
        return "fruit_only"
    if "no fruit" in selected_option.lower() or selected_option == PREFERENCE_VEG_ONLY:
        return "veg_only"
    return None  # "Mix of everything" or unknown = no constraint


def build_items(
    offer_id: int,
    overage: dict[int, int],
    categories: dict[int, str],
) -> dict[int, Item]:
    """
    Build Item objects by joining DB data with XLSX overage.

    Only includes items that have overage > 0.
    """
    db_items = fetch_offer_items(offer_id)
    items = {}

    for row in db_items:
        item_id = row["item_id"]
        if item_id not in overage or overage[item_id] <= 0:
            continue

        # Parse components JSON if present
        components = None
        if row["components"]:
            try:
                components = json.loads(row["components"]) if isinstance(row["components"], str) else row["components"]
            except (json.JSONDecodeError, TypeError):
                pass

        fg_group, fg_degree = assign_fungible_group(row["item_name"])
        sub_cat, usage, colour, shape = assign_classification(
            row["item_name"], row["category_id"]
        )

        pack_size = detect_pack_size(row["item_name"])
        price = row["item_price"] // pack_size

        items[item_id] = Item(
            id=item_id,
            name=row["item_name"],
            price=price,
            category_id=row["category_id"],
            category_name=category_name(row["category_id"], categories),
            size=row["size"] or 1,
            pack_order=row["sort_order"] or 0,
            overage=overage[item_id],
            fungible_group=fg_group,
            fungible_degree=fg_degree,
            sub_category=sub_cat,
            usage_type=usage,
            colour=colour,
            shape=shape,
            buy_qty=row["wholesale_qty"],
            buy_price=row["wholesale_price"],
            components=components,
        )

    logger.info(f"Built {len(items)} items with overage")
    return items


def build_boxes_from_db(offer_id: int) -> list[MysteryBox]:
    """
    Auto-detect mystery box buyers from DB and build MysteryBox objects.

    Returns boxes sorted ascending by target value (small first).
    """
    buyers = fetch_mystery_box_buyers(offer_id)
    boxes = []

    for buyer in buyers:
        email = buyer["user_email"]

        # Skip known donation buyers and staff self-picks
        if email in DONATION_IDENTIFIERS or email in STAFF_IDENTIFIERS:
            continue

        tier = infer_tier_from_name(buyer["product_name"])
        if tier is None:
            continue
        tier_config = BOX_TIERS[tier]

        preference = parse_preference(buyer["selected_option"])
        exclusions = []

        # Build exclusion rules from structured preference
        if preference == "fruit_only":
            exclusions.append(ExclusionRule(
                pattern="__category_exclude_veg__",
                source="preference",
            ))
        elif preference == "veg_only":
            exclusions.append(ExclusionRule(
                pattern="__category_exclude_fruit__",
                source="preference",
            ))

        # Get existing categories for merged boxes
        existing_cats = fetch_buyer_existing_categories(offer_id, email)

        box = MysteryBox(
            name=email,
            tier=tier,
            merged=True,  # default: merged with customer's regular order
            target_value=tier_config["target_value"],
            preference=preference,
            notes=buyer["buyer_note"],
            exclusions=exclusions,
            existing_categories=existing_cats,
        )
        boxes.append(box)

    # Sort by target value ascending (small first)
    boxes.sort(key=lambda b: b.target_value)

    logger.info(f"Built {len(boxes)} mystery boxes from DB")
    return boxes


def compute_charity_target(offer_id: int) -> int:
    """Compute charity allocation target in cents."""
    gross_retail = fetch_offer_gross_retail(offer_id)
    giving = fetch_customer_giving(offer_id)
    target = int(float(gross_retail) * CHARITY_BASE_PERCENT + float(giving) * CHARITY_GIVING_MULTIPLIER)
    logger.info(
        f"Charity target: ${target/100:.2f} "
        f"(gross=${gross_retail/100:.2f}, giving=${giving/100:.2f})"
    )
    return target


# ---------------------------------------------------------------------------
# Charity allocation (shared infrastructure, not strategy-specific)
# ---------------------------------------------------------------------------

def _allocate_charity(result: AllocationResult, charity_target: int) -> None:
    """
    Allocate remaining overage to charity toward target.

    Deliberately fills toward the charity target, preferring higher-value items.
    Remaining overage after charity target → stock.
    """
    if not result.charity:
        return

    # Use the first charity box as primary target
    primary = result.charity[0]
    primary.target_value = charity_target

    # Split target across charity recipients proportionally if multiple
    if len(result.charity) > 1:
        per_charity = charity_target // len(result.charity)
        for c in result.charity:
            c.target_value = per_charity

    for charity in result.charity:
        current_value = result.charity_value(charity)
        needed = charity.target_value - current_value

        if needed <= 0:
            continue

        # Sort available items by price descending (fill value quickly)
        available = [
            (item_id, item)
            for item_id, item in result.items.items()
            if result.remaining_overage(item_id) > 0
        ]
        available.sort(key=lambda x: x[1].price, reverse=True)

        for item_id, item in available:
            remaining = result.remaining_overage(item_id)
            if remaining <= 0:
                continue

            current_value = result.charity_value(charity)
            still_needed = charity.target_value - current_value
            if still_needed <= 0:
                break

            # Allocate up to what's needed
            max_qty_by_value = max(1, still_needed // max(item.price, 1))
            qty = min(remaining, max_qty_by_value)
            if qty <= 0:
                continue

            charity.allocations[item_id] = charity.allocations.get(item_id, 0) + qty

    # Remaining overage → stock
    for item_id, item in result.items.items():
        remaining = result.remaining_overage(item_id)
        if remaining > 0:
            result.stock[item_id] = remaining


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def allocate(
    offer_id: int,
    xlsx_path: Path,
    boxes: list[MysteryBox] | None = None,
    charity_names: list[str] | None = None,
    strategy: str = DEFAULT_STRATEGY,
    bootstrap_allocations: list[dict[int, int]] | None = None,
    charity_target_cents: int | None = None,
) -> AllocationResult:
    """
    Run the full allocation pipeline.

    Args:
        offer_id: The offer ID to allocate for
        xlsx_path: Path to the tweaked shopping list XLSX
        boxes: Pre-configured mystery boxes (from TUI). If None, auto-detect.
        charity_names: Charity recipient names. Defaults to [CHARITY_NAME].
        strategy: Name of the allocation strategy (default: ilp-optimal).
        bootstrap_allocations: Pre-computed box allocations to seed the strategy
            with (one dict per box, matching box order). Used by compare.py to
            feed discard-worst results into local-search without recomputing.
        charity_target_cents: Override charity target in cents (bypasses DB query).

    Returns:
        AllocationResult with all allocations
    """
    if charity_names is None:
        charity_names = [CHARITY_NAME]

    # Load data
    categories = fetch_categories()
    overage = read_overage_from_xlsx(xlsx_path)
    items = build_items(offer_id, overage, categories)

    if not items:
        logger.warning("No items with overage found")
        return AllocationResult(
            offer_id=offer_id,
            boxes=[],
            charity=[],
            stock={},
            items={},
        )

    # Build boxes
    if boxes is None:
        boxes = build_boxes_from_db(offer_id)

    # Build charity
    charity = [CharityBox(name=name) for name in charity_names]

    # Create result
    result = AllocationResult(
        offer_id=offer_id,
        boxes=boxes,
        charity=charity,
        stock={},
        items=items,
    )

    logger.info(
        f"Allocating {len(items)} items across {len(boxes)} boxes "
        f"+ {len(charity)} charity"
    )

    # Pre-fill box allocations if provided (e.g. from a prior discard-worst run)
    if bootstrap_allocations is not None:
        for box, allocs in zip(boxes, bootstrap_allocations):
            box.allocations = dict(allocs)

    # Run pluggable strategy (fills box.allocations in place)
    strategy_fn = get_strategy(strategy)
    logger.info(f"Running strategy: {strategy}")
    strategy_fn(result)

    # Charity allocation (shared infrastructure)
    logger.info("Phase 3: Charity allocation")
    if charity_target_cents is not None:
        charity_target = charity_target_cents
        logger.info(f"Charity target (override): ${charity_target/100:.2f}")
    else:
        charity_target = compute_charity_target(offer_id)
    _allocate_charity(result, charity_target)

    return result


def print_summary(result: AllocationResult) -> None:
    """Print a human-readable summary of the allocation."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Box summary table
    table = Table(title=f"Allocation Summary - Offer {result.offer_id}")
    table.add_column("Box", style="cyan")
    table.add_column("Tier", style="magenta")
    table.add_column("Type", style="blue")
    table.add_column("Items", justify="right")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Target", justify="right")
    table.add_column("Diff", justify="right")

    for box in result.boxes:
        value = result.box_value(box)
        target = box.target_value
        diff = value - target
        diff_str = f"+${diff/100:.2f}" if diff >= 0 else f"-${abs(diff)/100:.2f}"
        diff_style = "green" if diff >= 0 else "red"

        item_count = sum(1 for q in box.allocations.values() if q > 0)
        box_type = "merged" if box.merged else "standalone"
        pref = f" [{box.preference}]" if box.preference else ""

        table.add_row(
            box.name,
            box.tier,
            box_type + pref,
            str(item_count),
            f"${value/100:.2f}",
            f"${target/100:.2f}",
            f"[{diff_style}]{diff_str}[/]",
        )

    # Charity rows
    for charity in result.charity:
        value = result.charity_value(charity)
        target = charity.target_value
        diff = value - target
        diff_str = f"+${diff/100:.2f}" if diff >= 0 else f"-${abs(diff)/100:.2f}"
        diff_style = "green" if diff >= 0 else "red"
        item_count = sum(1 for q in charity.allocations.values() if q > 0)

        table.add_row(
            charity.name,
            "charity",
            "charity",
            str(item_count),
            f"${value/100:.2f}",
            f"${target/100:.2f}",
            f"[{diff_style}]{diff_str}[/]",
        )

    console.print(table)

    # Stock summary
    if result.stock:
        stock_value = sum(
            result.items[iid].price * qty
            for iid, qty in result.stock.items()
            if iid in result.items
        )
        stock_items = sum(1 for q in result.stock.values() if q > 0)
        console.print(f"\n[yellow]Stock:[/] {stock_items} items, ${stock_value/100:.2f}")

    # Constraint check
    violations = []
    for item_id, item in result.items.items():
        total = result.total_allocated_qty(item_id)
        if total > item.overage:
            violations.append(
                f"  {item.name} (ID {item_id}): allocated {total} > overage {item.overage}"
            )

    if violations:
        console.print(f"\n[red bold]CONSTRAINT VIOLATIONS:[/]")
        for v in violations:
            console.print(f"[red]{v}[/]")
    else:
        console.print(f"\n[green]All constraints satisfied.[/]")
