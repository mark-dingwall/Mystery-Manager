"""
Item categorization and fungible group assignment.

Uses DB-powered category data and config-based fungible groups.
"""

import logging

from allocator.config import (
    CLASSIFICATION_FALLBACK,
    FUNGIBLE_GROUPS,
    ITEM_CLASSIFICATIONS,
)

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFICATION = ("other", "cooking", "green", "round")


def assign_fungible_group(item_name: str) -> tuple[str | None, float]:
    """
    Assign an item to a fungible group based on name prefix matching.

    Returns (group_name, degree) tuple, or (None, 0.0) if no match.
    Degree indicates how interchangeable items in the group are:
    1.0 = near-identical, 0.7 = same type, 0.5 = related, 0.3 = similar role.
    """
    for group_name, (degree, prefixes, *_rest) in FUNGIBLE_GROUPS.items():
        for prefix in prefixes:
            if item_name.startswith(prefix) or item_name.lower().startswith(prefix.lower()):
                return group_name, degree
    return None, 0.0


def assign_classification(
    item_name: str, category_id: int
) -> tuple[str, str, str, str]:
    """
    Assign diversity classification tags to an item via prefix matching.

    Returns (sub_category, usage, colour, shape).
    Falls back to other_fruit/other_veg based on category_id if no match.
    """
    for _key, (prefixes, sub_cat, usage, colour, shape) in ITEM_CLASSIFICATIONS.items():
        for prefix in prefixes:
            if item_name.startswith(prefix) or item_name.lower().startswith(prefix.lower()):
                return sub_cat, usage, colour, shape

    fallback = CLASSIFICATION_FALLBACK.get(category_id, DEFAULT_CLASSIFICATION)
    logger.warning(f"No classification match for {item_name!r} (cat={category_id}), using fallback: {fallback[0]}")
    return fallback


def category_name(category_id: int, categories: dict[int, str]) -> str:
    """Get human-readable category name from ID."""
    name = categories.get(category_id, "unknown")
    return name.lower()
