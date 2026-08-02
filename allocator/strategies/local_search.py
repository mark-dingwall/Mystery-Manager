# STATUS: baseline — regression benchmark only; not the production direction (see CLAUDE.md § Project Direction). Do not extend.
# load-bearing: also the ilp-optimal fallback — do not remove.
"""
Local-search allocation strategy.

Bootstrap from discard-worst (greedy draft + penalty-delta trim), then
iteratively relocate/swap items between boxes to minimise the composite
penalty matching compare.py's scoring (value sweet-spot, same-item,
group concentration, 4D diversity, max value share, size floor).

Uses incremental objective evaluation: only the 2 affected boxes are
recomputed per candidate move, not all boxes. When run via compare.py
--all-strategies, pre-computed discard-worst allocations are passed in
to avoid redundant work.
"""

import logging

from allocator.config import (
    BOX_TIERS,
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    GROUP_ALLOWANCES,
    GROUP_CONCENTRATION_MULTIPLIER,
    GROUP_QTY_EXPONENT,
    LOCAL_SEARCH_MAX_ITERATIONS,
    MAX_VALUE_SHARE_MULTIPLIER,
    MAX_VALUE_SHARE_THRESHOLD,
    SAME_ITEM_MULTIPLIER,
    SIZE_FLOOR_MULTIPLIER,
    SIZE_FLOOR_TARGETS,
)
from allocator.models import AllocationResult
from allocator.strategies._helpers import (
    _item_allowance,
    compute_available_tags,
    has_hard_fungible_conflict,
    would_exceed_ceiling,
)
from allocator.strategies._scoring import value_penalty

logger = logging.getLogger(__name__)

MAX_ITERATIONS = LOCAL_SEARCH_MAX_ITERATIONS


class _BoxState:
    """Cached per-box metrics for incremental objective computation."""

    __slots__ = (
        "value_pct", "diversity_score", "same_item_penalty",
        "group_concentration_penalty", "max_value_share", "size_floor",
    )

    def __init__(self):
        self.value_pct = 0.0
        self.diversity_score = 0.0
        self.same_item_penalty = 0.0
        self.group_concentration_penalty = 0.0
        self.max_value_share = 0.0
        self.size_floor = 0.0


class _ObjectiveCache:
    """
    Maintains per-box cached state so the composite objective can be
    recomputed in O(n_boxes) after updating only the affected boxes in O(1).
    """

    def __init__(self, result: AllocationResult, available_tags: dict[str, set[str]]):
        self.result = result
        self.available_tags = available_tags
        # Pre-compute available tag counts (constant for the run)
        self._avail_counts = {
            dim: len(tags) for dim, tags in available_tags.items()
        }
        n = len(result.boxes)
        self.states = [_BoxState() for _ in range(n)]
        for i in range(n):
            self._recompute(i)

    def _recompute(self, box_idx: int) -> None:
        """Recompute cached state for a single box."""
        box = self.result.boxes[box_idx]
        items = self.result.items
        state = self.states[box_idx]

        # Value as % of box price
        value = 0
        for item_id, qty in box.allocations.items():
            if item_id in items:
                value += items[item_id].price * qty
        box_price = BOX_TIERS[box.tier]["price"]
        state.value_pct = value / box_price * 100 if box_price > 0 else 0.0

        # Diversity score using effective species (1/HHI), inlined for speed
        tag_counts: dict[str, dict[str, int]] = {
            "sub_category": {}, "usage": {}, "colour": {}, "shape": {},
        }
        _dim_attrs = (
            ("sub_category", "sub_category"),
            ("usage", "usage_type"),
            ("colour", "colour"),
            ("shape", "shape"),
        )
        for item_id, qty in box.allocations.items():
            if qty > 0 and item_id in items:
                item = items[item_id]
                for dim, attr in _dim_attrs:
                    tag = getattr(item, attr, "")
                    if tag:
                        tc = tag_counts[dim]
                        tc[tag] = tc.get(tag, 0) + qty

        score = 0.0
        ac = self._avail_counts
        for dim, weight in DIVERSITY_WEIGHTS.items():
            n_avail = ac.get(dim, 0)
            dc = tag_counts[dim]
            if n_avail > 0 and dc:
                total = sum(dc.values())
                hhi = sum((q / total) ** 2 for q in dc.values())
                eff = 1.0 / hhi
                score += weight * min(eff / n_avail, 1.0)
            elif n_avail == 0:
                score += weight
        state.diversity_score = score

        # Same-item penalty: per-item excess * price * multiplier
        si_pen = 0.0
        for item_id, qty in box.allocations.items():
            if qty <= 0 or item_id not in items:
                continue
            item = items[item_id]
            allow = _item_allowance(item, box.tier)
            excess = max(0, qty - allow)
            if excess > 0:
                si_pen += excess * item.price * SAME_ITEM_MULTIPLIER / 100.0
        state.same_item_penalty = si_pen

        # Group concentration penalty: min(qty, item_allowance) per group member
        groups: dict[str, tuple[float, float]] = {}
        for item_id, qty in box.allocations.items():
            if qty <= 0 or item_id not in items:
                continue
            item = items[item_id]
            if item.fungible_group:
                key = item.fungible_group
                degree = item.fungible_degree
            else:
                key = f"__item_{item_id}"
                degree = 1.0
            allow = _item_allowance(item, box.tier)
            capped_qty = min(qty, allow)
            if key in groups:
                prev_load, prev_degree = groups[key]
                groups[key] = (prev_load + capped_qty, prev_degree)
            else:
                groups[key] = (capped_qty, degree)

        gc_pen = 0.0
        for key, (group_load, degree) in groups.items():
            if key in GROUP_ALLOWANCES:
                ga = GROUP_ALLOWANCES[key].get(box.tier, group_load)
            else:
                continue
            excess = max(0, group_load - ga)
            if excess > 0:
                gc_pen += (excess ** GROUP_QTY_EXPONENT) * degree * GROUP_CONCENTRATION_MULTIPLIER
        state.group_concentration_penalty = gc_pen

        # Max value share penalty
        total_value = value
        if total_value > 0:
            max_share = 0.0
            for item_id, qty in box.allocations.items():
                if qty <= 0 or item_id not in items:
                    continue
                share = items[item_id].price * qty / total_value
                if share > max_share:
                    max_share = share
            mvs_excess = max(0.0, max_share - MAX_VALUE_SHARE_THRESHOLD)
            state.max_value_share = mvs_excess * MAX_VALUE_SHARE_MULTIPLIER
        else:
            state.max_value_share = 0.0

        # Size floor penalty
        sf_target = SIZE_FLOOR_TARGETS.get(box.tier, 0)
        if sf_target > 0:
            total_size = 0
            for item_id, qty in box.allocations.items():
                if qty <= 0 or item_id not in items:
                    continue
                total_size += items[item_id].size * qty
            deficit = max(0, sf_target - total_size)
            state.size_floor = deficit * SIZE_FLOOR_MULTIPLIER
        else:
            state.size_floor = 0.0

    def save(self, *box_indices: int) -> list[tuple[float, ...]]:
        """Save state of specified boxes for cheap restore on revert."""
        return [
            (self.states[i].value_pct, self.states[i].diversity_score,
             self.states[i].same_item_penalty, self.states[i].group_concentration_penalty,
             self.states[i].max_value_share, self.states[i].size_floor)
            for i in box_indices
        ]

    def restore(self, box_indices: tuple[int, ...], saved: list[tuple[float, ...]]) -> None:
        """Restore previously saved state (O(1) instead of recompute)."""
        for idx, (vp, ds, si, gc, mvs, sf) in zip(box_indices, saved):
            s = self.states[idx]
            s.value_pct = vp
            s.diversity_score = ds
            s.same_item_penalty = si
            s.group_concentration_penalty = gc
            s.max_value_share = mvs
            s.size_floor = sf

    def recompute(self, *box_indices: int) -> None:
        """Recompute only the specified boxes."""
        for i in box_indices:
            self._recompute(i)

    def objective(self) -> float:
        """
        Compute composite objective from cached per-box state.

        Matches compare.py's composite scoring:
        avg(value + same_item + group_concentration + diversity +
            max_value_share + size_floor). No fairness term.
        """
        states = self.states
        n = len(states)
        if n == 0:
            return 0.0

        # Per-box penalties (averaged)
        total_box_pen = 0.0
        for s in states:
            total_box_pen += (
                value_penalty(s.value_pct)
                + s.same_item_penalty
                + s.group_concentration_penalty
                + (1.0 - s.diversity_score) * DIVERSITY_PENALTY_MULTIPLIER
                + s.max_value_share
                + s.size_floor
            )

        return total_box_pen / n


def run(result: AllocationResult) -> None:
    """Local search: bootstrap from deal-topup then improve via moves."""
    # Bootstrap from discard-worst (better starting point than deal-topup).
    # Skip if boxes already have allocations (pre-filled by allocate()).
    if not any(box.allocations for box in result.boxes):
        from allocator.strategies.discard_worst import run as bootstrap_run
        bootstrap_run(result)

    if len(result.boxes) < 2:
        return

    available_tags = compute_available_tags(result)
    cache = _ObjectiveCache(result, available_tags)
    best_obj = cache.objective()
    logger.info(f"Local search starting objective: {best_obj:.4f}")

    improved = True
    iteration = 0

    while improved and iteration < MAX_ITERATIONS:
        improved = False
        iteration += 1

        # Try relocations
        for i, box_from in enumerate(result.boxes):
            for item_id, qty in list(box_from.allocations.items()):
                current_qty = box_from.allocations.get(item_id, 0)
                if current_qty <= 0 or item_id not in result.items:
                    continue

                item = result.items[item_id]

                for j, box_to in enumerate(result.boxes):
                    if i == j:
                        continue

                    # Check constraints for receiving box
                    if box_to.is_excluded(item):
                        continue
                    if has_hard_fungible_conflict(item, box_to, result):
                        continue
                    if would_exceed_ceiling(box_to, item, 1, result):
                        continue

                    # Try relocate: move 1 unit from box_from to box_to
                    cur = box_from.allocations.get(item_id, 0)
                    if cur <= 0:
                        break
                    saved = cache.save(i, j)
                    box_from.allocations[item_id] = cur - 1
                    if box_from.allocations[item_id] == 0:
                        del box_from.allocations[item_id]
                    box_to.allocations[item_id] = box_to.allocations.get(item_id, 0) + 1

                    cache.recompute(i, j)
                    new_obj = cache.objective()
                    if new_obj < best_obj:
                        best_obj = new_obj
                        improved = True
                    else:
                        # Revert
                        box_to.allocations[item_id] -= 1
                        if box_to.allocations[item_id] == 0:
                            del box_to.allocations[item_id]
                        box_from.allocations[item_id] = box_from.allocations.get(item_id, 0) + 1
                        cache.restore((i, j), saved)

                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Try swaps
        for i, box_a in enumerate(result.boxes):
            for item_a_id, qty_a in list(box_a.allocations.items()):
                cur_a = box_a.allocations.get(item_a_id, 0)
                if cur_a <= 0 or item_a_id not in result.items:
                    continue
                item_a = result.items[item_a_id]

                for j, box_b in enumerate(result.boxes):
                    if j <= i:
                        continue

                    for item_b_id, qty_b in list(box_b.allocations.items()):
                        cur_b = box_b.allocations.get(item_b_id, 0)
                        if cur_b <= 0 or item_b_id not in result.items:
                            continue
                        if item_a_id == item_b_id:
                            continue
                        # Re-check box_a still has item_a
                        if box_a.allocations.get(item_a_id, 0) <= 0:
                            break

                        item_b = result.items[item_b_id]

                        # Check constraints
                        if box_b.is_excluded(item_a) or box_a.is_excluded(item_b):
                            continue

                        saved = cache.save(i, j)

                        # Temporarily remove both, check fungible conflicts
                        box_a.allocations[item_a_id] = box_a.allocations.get(item_a_id, 0) - 1
                        if box_a.allocations[item_a_id] == 0:
                            del box_a.allocations[item_a_id]
                        box_b.allocations[item_b_id] = box_b.allocations.get(item_b_id, 0) - 1
                        if box_b.allocations[item_b_id] == 0:
                            del box_b.allocations[item_b_id]

                        ok_a = not has_hard_fungible_conflict(item_b, box_a, result)
                        ok_b = not has_hard_fungible_conflict(item_a, box_b, result)

                        # Check ceiling
                        ok_a = ok_a and not would_exceed_ceiling(box_a, item_b, 1, result)
                        ok_b = ok_b and not would_exceed_ceiling(box_b, item_a, 1, result)

                        if ok_a and ok_b:
                            # Apply swap
                            box_a.allocations[item_b_id] = box_a.allocations.get(item_b_id, 0) + 1
                            box_b.allocations[item_a_id] = box_b.allocations.get(item_a_id, 0) + 1

                            cache.recompute(i, j)
                            new_obj = cache.objective()
                            if new_obj < best_obj:
                                best_obj = new_obj
                                improved = True
                            else:
                                # Revert swap
                                box_a.allocations[item_b_id] -= 1
                                if box_a.allocations[item_b_id] == 0:
                                    del box_a.allocations[item_b_id]
                                box_b.allocations[item_a_id] -= 1
                                if box_b.allocations[item_a_id] == 0:
                                    del box_b.allocations[item_a_id]
                                # Restore originals
                                box_a.allocations[item_a_id] = box_a.allocations.get(item_a_id, 0) + 1
                                box_b.allocations[item_b_id] = box_b.allocations.get(item_b_id, 0) + 1
                                cache.restore((i, j), saved)
                        else:
                            # Revert removal
                            box_a.allocations[item_a_id] = box_a.allocations.get(item_a_id, 0) + 1
                            box_b.allocations[item_b_id] = box_b.allocations.get(item_b_id, 0) + 1
                            cache.restore((i, j), saved)

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    logger.info(
        f"Local search complete: {iteration} iterations, "
        f"final objective: {best_obj:.4f}"
    )
