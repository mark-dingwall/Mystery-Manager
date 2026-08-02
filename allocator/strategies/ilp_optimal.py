"""
ILP-optimal allocation strategy.

Integer Linear Programming via PuLP/CBC: minimise a composite penalty that
matches compare.py's scoring — convex piecewise-linear value penalty,
same-item penalty, group concentration, 4D diversity coverage,
max value share, and size floor.

Falls back to local-search if PuLP is not installed or solver fails.
"""

import logging

from allocator.config import (
    BOX_TIERS,
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    GROUP_ALLOWANCES,
    GROUP_CONCENTRATION_MULTIPLIER,
    GROUP_QTY_EXPONENT,
    ILP_BALANCE_WEIGHT,
    ILP_COVERAGE_WEIGHT,
    ILP_HHI_BREAKPOINTS,
    MAX_VALUE_SHARE_MULTIPLIER,
    MAX_VALUE_SHARE_THRESHOLD,
    SAME_ITEM_MULTIPLIER,
    SIZE_FLOOR_MULTIPLIER,
    SIZE_FLOOR_TARGETS,
    VALUE_CEILING_PCT,
    VALUE_PENALTY_EXPONENT,
    VALUE_SWEET_FROM,
    VALUE_SWEET_TO,
)
from allocator.models import AllocationResult
from allocator.strategies._helpers import _item_allowance, compute_available_tags

logger = logging.getLogger(__name__)

TIME_LIMIT = 30  # seconds


def _compute_value_lines():
    """
    Generate piecewise-linear tangent-line approximation of x^n.

    For convex f(x) = x^n (n > 1), tangent at point x_i:
      slope = n * x_i^(n-1)
      intercept = x_i^n - slope * x_i
    Epigraph constraint: pen >= slope * x + intercept

    Returns list of (slope, intercept) pairs for the full vp domain:
    - Under sweet spot: penalty = (SWEET_FROM - vp)^n
    - Over sweet spot:  penalty = (vp - SWEET_TO)^n
    """
    n = VALUE_PENALTY_EXPONENT
    # Tangent points spanning expected distance range (0-40pp from sweet spot)
    tangent_points = [0.5, 2, 5, 10, 15, 20, 30, 40]

    lines = []
    # Zero line (penalty = 0 inside sweet spot, also valid lower bound elsewhere)
    lines.append((0.0, 0.0))

    for xi in tangent_points:
        # f(xi) = xi^n, f'(xi) = n * xi^(n-1)
        f_xi = xi ** n
        df_xi = n * xi ** (n - 1)

        # Under side: penalty = (SWEET_FROM - vp)^n
        # Let u = SWEET_FROM - vp, then pen >= df_xi * u + (f_xi - df_xi * xi)
        # Substituting u = SWEET_FROM - vp:
        #   pen >= df_xi * (SWEET_FROM - vp) + (f_xi - df_xi * xi)
        #   pen >= -df_xi * vp + (df_xi * SWEET_FROM + f_xi - df_xi * xi)
        under_slope = -df_xi
        under_intercept = df_xi * VALUE_SWEET_FROM + f_xi - df_xi * xi
        lines.append((under_slope, under_intercept))

        # Over side: penalty = (vp - SWEET_TO)^n
        # Let u = vp - SWEET_TO, then pen >= df_xi * u + (f_xi - df_xi * xi)
        # Substituting u = vp - SWEET_TO:
        #   pen >= df_xi * vp + (-df_xi * SWEET_TO + f_xi - df_xi * xi)
        over_slope = df_xi
        over_intercept = -df_xi * VALUE_SWEET_TO + f_xi - df_xi * xi
        lines.append((over_slope, over_intercept))

    return lines


_VALUE_LINES = _compute_value_lines()


def run(result: AllocationResult) -> None:
    """ILP allocation: optimal assignment via mixed-integer programming."""
    try:
        import pulp
    except ImportError:
        from allocator.strategies import FALLBACK_STRATEGY, get_strategy
        logger.warning(f"PuLP not installed, falling back to {FALLBACK_STRATEGY}")
        get_strategy(FALLBACK_STRATEGY)(result)
        return

    if not result.boxes or not result.items:
        return

    try:
        _solve_ilp(result, pulp)
    except Exception as e:
        from allocator.strategies import FALLBACK_STRATEGY, get_strategy
        logger.warning(f"ILP solver failed ({e}), falling back to {FALLBACK_STRATEGY}")
        # Clear any partial allocations
        for box in result.boxes:
            box.allocations.clear()
        get_strategy(FALLBACK_STRATEGY)(result)


def _solve_ilp(result: AllocationResult, pulp) -> None:
    """Build and solve the ILP model."""
    items = list(result.items.values())
    boxes = result.boxes
    item_ids = [i.id for i in items]
    n_items = len(items)
    n_boxes = len(boxes)

    prob = pulp.LpProblem("mystery_box_allocation", pulp.LpMinimize)

    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # x[i][b] = integer qty of item i assigned to box b
    x = {}
    for i, item in enumerate(items):
        x[i] = {}
        for b in range(n_boxes):
            ub = item.overage
            if boxes[b].is_excluded(item):
                ub = 0
            x[i][b] = pulp.LpVariable(
                f"x_{item.id}_{b}", lowBound=0, upBound=ub, cat="Integer"
            )

    # y[i][b] = binary: is item i present in box b?
    y = {}
    for i, item in enumerate(items):
        y[i] = {}
        for b in range(n_boxes):
            y[i][b] = pulp.LpVariable(f"y_{item.id}_{b}", cat="Binary")

    # Link x and y: y[i][b] = 1 iff x[i][b] >= 1
    for i, item in enumerate(items):
        for b in range(n_boxes):
            prob += y[i][b] <= x[i][b]
            prob += x[i][b] <= item.overage * y[i][b]

    # -----------------------------------------------------------------------
    # Structural constraints
    # -----------------------------------------------------------------------

    # Overage: total assigned across all boxes <= overage
    for i, item in enumerate(items):
        prob += (
            pulp.lpSum(x[i][b] for b in range(n_boxes)) <= item.overage,
            f"overage_{item.id}",
        )

    # Ceiling: value in each box <= ceiling (% of box price)
    for b in range(n_boxes):
        box = boxes[b]
        prob += (
            pulp.lpSum(items[i].price * x[i][b] for i in range(n_items))
            <= VALUE_CEILING_PCT * BOX_TIERS[box.tier]["price"],
            f"ceiling_{b}",
        )

    # Ceiling guard per fungible group: total qty <= ceil(2 * group_allowance)
    # Also per-item ceiling: x[i][b] <= 2 * item_allowance
    import math as _math
    fungible_group_members: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if item.fungible_group:
            fungible_group_members.setdefault(item.fungible_group, []).append(i)

    for group_name, member_indices in fungible_group_members.items():
        if len(member_indices) <= 1:
            continue
        for b in range(n_boxes):
            # Per-group cap from GROUP_ALLOWANCES
            if group_name in GROUP_ALLOWANCES:
                ga = GROUP_ALLOWANCES[group_name].get(boxes[b].tier, 99)
                cap = _math.ceil(2 * ga)
                prob += (
                    pulp.lpSum(x[mi][b] for mi in member_indices) <= cap,
                    f"fungible_cap_{group_name}_{b}",
                )

    # Per-item ceiling: x[i][b] <= 2 * item_allowance
    for i, item in enumerate(items):
        for b in range(n_boxes):
            item_allow_tier = _item_allowance(item, boxes[b].tier)
            cap = 2 * item_allow_tier
            if cap < item.overage:
                prob += (
                    x[i][b] <= cap,
                    f"item_cap_{item.id}_{b}",
                )

    # -----------------------------------------------------------------------
    # 4a. Convex piecewise-linear value penalty
    # -----------------------------------------------------------------------
    # vp[b] = value as % of box price. Link: vp[b] * price_b == 100 * value_b
    vp = {}
    pen_val = {}
    for b in range(n_boxes):
        box = boxes[b]
        vp[b] = pulp.LpVariable(f"vp_{b}", lowBound=0)
        pen_val[b] = pulp.LpVariable(f"pen_val_{b}", lowBound=0)

        # Link vp to allocations: vp[b] = value / box_price * 100
        value_expr = pulp.lpSum(items[i].price * x[i][b] for i in range(n_items))
        box_price = BOX_TIERS[box.tier]["price"]
        prob += (
            box_price * vp[b] == 100 * value_expr,
            f"vp_link_{b}",
        )

        # Epigraph constraints: pen_val[b] >= slope * vp[b] + intercept
        for k, (slope, intercept) in enumerate(_VALUE_LINES):
            prob += (
                pen_val[b] >= slope * vp[b] + intercept,
                f"val_line_{b}_{k}",
            )

    # -----------------------------------------------------------------------
    # 4b. Diversity: binary coverage + HHI concentration penalty
    # -----------------------------------------------------------------------
    available_tags = compute_available_tags(result)

    # For each dimension, build tag→item-indices mapping
    dim_attrs = {
        "sub_category": "sub_category",
        "usage": "usage_type",
        "colour": "colour",
        "shape": "shape",
    }

    pen_div = {}
    for b in range(n_boxes):
        pen_div[b] = pulp.LpVariable(f"pen_div_{b}", lowBound=0)

    # --- Part (a): Binary coverage (same as before) ---
    coverage_exprs = {b: [] for b in range(n_boxes)}

    # Estimate total items per box (for HHI share normalisation)
    median_price = sorted(it.price for it in items)[len(items) // 2] if items else 1
    q_est = {b: max(boxes[b].target_value / max(median_price, 1), 1.0) for b in range(n_boxes)}

    # --- Part (b): HHI concentration penalty accumulators ---
    hhi_terms = {b: [] for b in range(n_boxes)}

    # Piecewise-linear tangent lines for f(s) = s^2 at breakpoints
    # At breakpoint p: tangent is f'(p)*(s - p) + p^2 = 2p*s - p^2
    tangent_lines = [(2 * p, -(p ** 2)) for p in ILP_HHI_BREAKPOINTS if p > 0]

    for dim, attr in dim_attrs.items():
        weight = DIVERSITY_WEIGHTS[dim]
        avail_tags = available_tags.get(dim, set())
        n_avail = len(avail_tags)

        if n_avail == 0:
            for b in range(n_boxes):
                coverage_exprs[b].append(weight)
            continue

        # Build tag → item indices
        tag_items: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            tag_val = getattr(item, attr, "")
            if tag_val and tag_val in avail_tags:
                tag_items.setdefault(tag_val, []).append(i)

        for tag in avail_tags:
            members = tag_items.get(tag, [])
            if not members:
                continue

            for b in range(n_boxes):
                # Binary coverage variable (unchanged)
                z_var = pulp.LpVariable(f"z_{dim}_{tag}_{b}", cat="Binary")
                prob += z_var <= pulp.lpSum(y[i][b] for i in members)
                for i in members:
                    prob += z_var >= y[i][b]
                coverage_exprs[b].append(weight / n_avail * z_var)

                # HHI: share = cnt / Q_est, sq >= share^2
                cnt = pulp.lpSum(x[i][b] for i in members)
                share = cnt / q_est[b]
                sq = pulp.LpVariable(f"sq_{dim}_{tag}_{b}", lowBound=0)
                # Tangent-line constraints: sq >= 2p*share - p^2
                for slope, intercept in tangent_lines:
                    prob += sq >= slope * share + intercept
                hhi_terms[b].append(sq)

    # Combined diversity penalty: alpha * DPM * (1 - coverage) + beta * DPM * hhi
    dpm = DIVERSITY_PENALTY_MULTIPLIER
    alpha = ILP_COVERAGE_WEIGHT
    beta = ILP_BALANCE_WEIGHT
    for b in range(n_boxes):
        coverage = pulp.lpSum(coverage_exprs[b])
        hhi_approx = pulp.lpSum(hhi_terms[b]) if hhi_terms[b] else 0
        prob += (
            pen_div[b] >= alpha * dpm * (1.0 - coverage) + beta * dpm * hhi_approx,
            f"div_pen_{b}",
        )

    # -----------------------------------------------------------------------
    # 4c. Same-item penalty (per-item excess * price * multiplier)
    # -----------------------------------------------------------------------
    pen_si = {}
    for b in range(n_boxes):
        pen_si[b] = pulp.LpVariable(f"pen_si_{b}", lowBound=0)

    si_terms = {b: [] for b in range(n_boxes)}

    for i, item in enumerate(items):
        for b in range(n_boxes):
            item_allow_tier = _item_allowance(item, boxes[b].tier)
            if item_allow_tier >= item.overage:
                continue  # can never exceed allowance
            # excess >= x[i][b] - item_allowance, excess >= 0
            si_excess = pulp.LpVariable(f"si_excess_{item.id}_{b}", lowBound=0)
            prob += si_excess >= x[i][b] - item_allow_tier
            si_terms[b].append(si_excess * item.price * SAME_ITEM_MULTIPLIER / 100.0)

    for b in range(n_boxes):
        if si_terms[b]:
            prob += pen_si[b] >= pulp.lpSum(si_terms[b]), f"si_pen_{b}"
        else:
            prob += pen_si[b] == 0, f"si_pen_{b}"

    # -----------------------------------------------------------------------
    # 4d. Group concentration penalty (group load via min-capped qty)
    # -----------------------------------------------------------------------
    # Only model explicit fungible groups with GROUP_ALLOWANCES entries
    all_fungible: dict[str, tuple[float, list[int]]] = {}
    for i, item in enumerate(items):
        if item.fungible_group:
            if item.fungible_group in all_fungible:
                _, members = all_fungible[item.fungible_group]
                members.append(i)
            else:
                all_fungible[item.fungible_group] = (item.fungible_degree, [i])

    pen_gc = {}
    for b in range(n_boxes):
        pen_gc[b] = pulp.LpVariable(f"pen_gc_{b}", lowBound=0)

    gc_terms = {b: [] for b in range(n_boxes)}

    # Tangent-line breakpoints for excess^exponent approximation
    _gq_breakpoints = [0, 1, 2, 4, 6, 8]

    for group_name, (degree, member_indices) in all_fungible.items():
        if group_name not in GROUP_ALLOWANCES:
            continue
        if len(member_indices) <= 1:
            continue

        for b in range(n_boxes):
            ga = GROUP_ALLOWANCES[group_name].get(boxes[b].tier, 99)
            # group_load = sum of min(x[i][b], item_allowance) for members
            # Approximate: use x[i][b] directly (min-capping is hard in ILP;
            # the per-item cap constraint above limits this)
            group_load_expr = pulp.lpSum(x[i][b] for i in member_indices)
            # excess >= group_load - group_allowance, excess >= 0
            gc_excess = pulp.LpVariable(f"gc_excess_{group_name}_{b}", lowBound=0)
            prob += gc_excess >= group_load_expr - ga

            # Epigraph for excess^exponent via tangent lines
            pen_var = pulp.LpVariable(f"gc_pen_{group_name}_{b}", lowBound=0)
            n_exp = GROUP_QTY_EXPONENT
            for xi in _gq_breakpoints:
                if xi == 0:
                    continue
                f_xi = xi ** n_exp
                df_xi = n_exp * xi ** (n_exp - 1)
                prob += pen_var >= df_xi * gc_excess + (f_xi - df_xi * xi)

            gc_terms[b].append(degree * GROUP_CONCENTRATION_MULTIPLIER * pen_var)

    for b in range(n_boxes):
        if gc_terms[b]:
            prob += pen_gc[b] >= pulp.lpSum(gc_terms[b]), f"gc_pen_{b}"
        else:
            prob += pen_gc[b] == 0, f"gc_pen_{b}"

    # -----------------------------------------------------------------------
    # 4e. Max value share penalty
    # -----------------------------------------------------------------------
    pen_mvs = {}
    for b in range(n_boxes):
        pen_mvs[b] = pulp.LpVariable(f"pen_mvs_{b}", lowBound=0)

    for b in range(n_boxes):
        box = boxes[b]
        target_value = BOX_TIERS[box.tier]["target_value"]
        if target_value <= 0:
            prob += pen_mvs[b] == 0, f"mvs_pen_{b}"
            continue
        # For each item, share_i ≈ item.price * x[i][b] / target_value
        # mvs_excess_i >= share_i - threshold
        mvs_item_terms = []
        for i, item in enumerate(items):
            if item.price <= 0:
                continue
            mvs_excess_i = pulp.LpVariable(f"mvs_ex_{item.id}_{b}", lowBound=0)
            share_expr = item.price * x[i][b] / target_value
            prob += mvs_excess_i >= share_expr - MAX_VALUE_SHARE_THRESHOLD
            mvs_item_terms.append(mvs_excess_i)
        if mvs_item_terms:
            # pen_mvs >= max(mvs_excess_i) * multiplier — approximate with sum
            # (only one item usually dominates; sum is a valid upper bound for
            # the minimiser to push down)
            for mvs_t in mvs_item_terms:
                prob += pen_mvs[b] >= mvs_t * MAX_VALUE_SHARE_MULTIPLIER
        else:
            prob += pen_mvs[b] == 0, f"mvs_pen_{b}"

    # -----------------------------------------------------------------------
    # 4f. Size floor penalty
    # -----------------------------------------------------------------------
    pen_sf = {}
    for b in range(n_boxes):
        pen_sf[b] = pulp.LpVariable(f"pen_sf_{b}", lowBound=0)

    for b in range(n_boxes):
        box = boxes[b]
        sf_target = SIZE_FLOOR_TARGETS.get(box.tier, 0)
        if sf_target <= 0:
            prob += pen_sf[b] == 0, f"sf_pen_{b}"
            continue
        total_size_expr = pulp.lpSum(items[i].size * x[i][b] for i in range(n_items))
        # deficit >= sf_target - total_size, deficit >= 0
        sf_deficit = pulp.LpVariable(f"sf_deficit_{b}", lowBound=0)
        prob += sf_deficit >= sf_target - total_size_expr
        prob += pen_sf[b] >= sf_deficit * SIZE_FLOOR_MULTIPLIER, f"sf_pen_{b}"

    # -----------------------------------------------------------------------
    # 4g. Objective: minimise avg(pen_val + pen_si + pen_gc + pen_div +
    #                               pen_mvs + pen_sf)
    # -----------------------------------------------------------------------
    avg_box_pen = pulp.lpSum(
        pen_val[b] + pen_si[b] + pen_gc[b] + pen_div[b] + pen_mvs[b] + pen_sf[b]
        for b in range(n_boxes)
    ) / n_boxes

    prob += avg_box_pen

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    try:
        solver = pulp.HiGHS(msg=0, timeLimit=TIME_LIMIT)
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        if pulp.LpStatus[status] == "Infeasible":
            raise RuntimeError("ILP infeasible")

    if prob.sol_status not in (
        pulp.constants.LpSolutionOptimal,
        pulp.constants.LpSolutionIntegerFeasible,
    ):
        raise RuntimeError(f"No feasible solution found: {pulp.LpStatus[status]}")

    # -----------------------------------------------------------------------
    # Extract solution
    # -----------------------------------------------------------------------
    for i, item in enumerate(items):
        for b in range(n_boxes):
            val = x[i][b].varValue
            if val is not None and val > 0.5:
                qty = int(round(val))
                if qty > 0:
                    boxes[b].allocations[item.id] = (
                        boxes[b].allocations.get(item.id, 0) + qty
                    )

    total_assigned = sum(
        sum(q for q in box.allocations.values())
        for box in boxes
    )
    logger.info(
        f"ILP solved: status={pulp.LpStatus[status]}, "
        f"total items assigned={total_assigned}"
    )
