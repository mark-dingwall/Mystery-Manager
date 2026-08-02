# STATUS: baseline — regression benchmark only; not the production direction (see CLAUDE.md § Project Direction). Do not extend.
"""
Round-robin allocation strategy.

Simple baseline: sort items by price descending, cycle through boxes assigning
1 unit each. Minimal top-up for boxes still under target.
"""

import logging

from allocator.config import ROUND_ROBIN_MAX_PASSES, VALUE_CEILING_PCT
from allocator.models import AllocationResult
from allocator.strategies._helpers import (
    assign_item,
    box_deficit,
    can_assign,
    has_hard_fungible_conflict,
    would_exceed_ceiling,
)

logger = logging.getLogger(__name__)


def run(result: AllocationResult) -> None:
    """Round-robin strategy: cycle items through boxes, then fill deficits."""
    _round_robin_phase(result)
    _deficit_fill_phase(result)


def _round_robin_phase(result: AllocationResult) -> None:
    """Phase 1: items by price desc, cycle boxes assigning 1 unit each."""
    items_by_price = sorted(
        result.items.values(),
        key=lambda i: i.price,
        reverse=True,
    )
    num_boxes = len(result.boxes)
    if num_boxes == 0:
        return

    box_idx = 0

    for item in items_by_price:
        while result.remaining_overage(item.id) > 0:
            assigned = False
            for attempt in range(num_boxes):
                bi = (box_idx + attempt) % num_boxes
                box = result.boxes[bi]

                if box.is_excluded(item):
                    continue
                if box_deficit(box, result) <= 0:
                    continue
                if has_hard_fungible_conflict(item, box, result):
                    continue
                if would_exceed_ceiling(box, item, 1, result):
                    continue

                assign_item(item.id, 1, box)
                box_idx = (bi + 1) % num_boxes
                assigned = True
                break

            if not assigned:
                break

    logger.info("Round-robin phase complete")


def _deficit_fill_phase(result: AllocationResult) -> None:
    """Phase 2: assign remaining overage to box with largest deficit."""
    for _ in range(ROUND_ROBIN_MAX_PASSES):
        # Find box with largest positive deficit
        best_box = None
        best_deficit = 0
        for box in result.boxes:
            d = box_deficit(box, result)
            if d > best_deficit:
                best_deficit = d
                best_box = box

        if best_box is None:
            break

        # Find any assignable item
        assigned = False
        for item in sorted(result.items.values(), key=lambda i: i.price, reverse=True):
            if can_assign(item, 1, best_box, result):
                assign_item(item.id, 1, best_box)
                assigned = True
                break

        if not assigned:
            break

    logger.info("Deficit-fill phase complete")
