# STATUS: baseline — regression benchmark only; not the production direction (see CLAUDE.md § Project Direction). Do not extend.
"""
Greedy best-fit allocation strategy.

Single pass: for each item unit (scarce items first), assign to the
highest-scoring box using score_topup_candidate().
"""

import logging

from allocator.config import GREEDY_MAX_ROUNDS
from allocator.models import AllocationResult
from allocator.scorer import score_topup_candidate
from allocator.strategies._helpers import assign_item, box_deficit

logger = logging.getLogger(__name__)


def run(result: AllocationResult) -> None:
    """Greedy best-fit: assign each item unit to best-scoring box."""
    num_boxes = len(result.boxes)
    if num_boxes == 0:
        return

    for round_num in range(GREEDY_MAX_ROUNDS):
        # Get items with remaining overage, sorted by scarcity (lowest first)
        available = [
            item for item in result.items.values()
            if result.remaining_overage(item.id) > 0
        ]
        if not available:
            break

        available.sort(key=lambda i: result.remaining_overage(i.id))

        made_progress = False

        for item in available:
            if result.remaining_overage(item.id) <= 0:
                continue

            # Find the best box for this item
            best_score = float("-inf")
            best_box = None

            for box in result.boxes:
                s = score_topup_candidate(item, 1, box, result)
                if s > best_score:
                    best_score = s
                    best_box = box

            if best_box is not None and best_score > float("-inf"):
                assign_item(item.id, 1, best_box)
                made_progress = True

        if not made_progress:
            break

    logger.info(f"Greedy best-fit completed in {round_num + 1} rounds")
