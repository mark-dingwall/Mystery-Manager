# STATUS: baseline — regression benchmark only; not the production direction (see CLAUDE.md § Project Direction). Do not extend.
"""
Min-max deficit allocation strategy.

Bin-packing approach: items sorted by scarcity (lowest overage first so scarce
items get fair distribution). Each unit assigned to the box with the largest
positive deficit. Falls back to over-target boxes (up to ceiling) when all are
at target.
"""

import logging

from allocator.config import MINMAX_MAX_PASSES
from allocator.models import AllocationResult
from allocator.strategies._helpers import assign_item, box_deficit, can_assign

logger = logging.getLogger(__name__)


def run(result: AllocationResult) -> None:
    """Min-max deficit: assign scarce items first to neediest boxes."""
    num_boxes = len(result.boxes)
    if num_boxes == 0:
        return

    for pass_num in range(MINMAX_MAX_PASSES):
        # Items sorted by scarcity (lowest remaining overage first)
        available = [
            item for item in result.items.values()
            if result.remaining_overage(item.id) > 0
        ]
        if not available:
            break

        available.sort(key=lambda i: (result.remaining_overage(i.id), -i.price))

        made_progress = False

        for item in available:
            while result.remaining_overage(item.id) > 0:
                # Find box with largest positive deficit that can accept this item
                best_box = None
                best_deficit = -1

                for box in result.boxes:
                    if not can_assign(item, 1, box, result):
                        continue
                    d = box_deficit(box, result)
                    if d > best_deficit:
                        best_deficit = d
                        best_box = box

                if best_box is None:
                    # No box with positive deficit — try over-target boxes (up to ceiling)
                    for box in result.boxes:
                        if not can_assign(item, 1, box, result):
                            continue
                        # can_assign already checks ceiling, so any passing box works
                        d = box_deficit(box, result)
                        if best_box is None or d > best_deficit:
                            best_deficit = d
                            best_box = box

                if best_box is None:
                    break

                assign_item(item.id, 1, best_box)
                made_progress = True

        if not made_progress:
            break

    logger.info(f"Min-max deficit completed in {pass_num + 1} passes")
