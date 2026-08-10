"""Tests for allocator/tuning.py, extract_features, and tune_scoring CV splits."""

import math
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from allocator.config import (
    BOX_TIERS,
    CATEGORY_FRUIT,
    CATEGORY_VEGETABLES,
    DIVERSITY_PENALTY_MULTIPLIER,
    DIVERSITY_WEIGHTS,
    GROUP_ALLOWANCES,
    GROUP_CONCENTRATION_MULTIPLIER,
    GROUP_QTY_EXPONENT,
    MAX_COMPOSITE_SCORE,
    MAX_VALUE_SHARE_MULTIPLIER,
    MAX_VALUE_SHARE_THRESHOLD,
    PREF_VIOLATION_PENALTY,
    SAME_ITEM_MULTIPLIER,
    SIZE_FLOOR_MULTIPLIER,
    SIZE_FLOOR_TARGETS,
)
from allocator.strategies._helpers import (
    _box_tag_counts,
    _effective_species,
    compute_available_tags,
    compute_diversity_score,
)
from allocator.strategies._scoring import (
    _resolve_item_allowance,
    group_concentration_penalty_for_box,
    same_item_penalty_for_box,
    max_value_share_penalty_for_box,
    size_floor_penalty_for_box,
    total_penalty,
    value_penalty,
)
from allocator.tuning import (
    _diversity_penalty,
    _group_concentration_penalty,
    _max_value_share_penalty,
    _same_item_penalty,
    _size_floor_penalty,
    _value_penalty,
    compute_marginal_deltas,
    compute_objective,
    default_params,
    rescore_box,
    rescore_offer,
)


# ── default_params ──────────────────────────────────────────────────────────


class TestDefaultParams:
    def test_returns_all_required_keys(self):
        p = default_params()
        required = [
            "value_sweet_from", "value_sweet_to", "value_penalty_exponent",
            "same_item_multiplier", "group_concentration_multiplier",
            "group_qty_exponent", "group_allowances",
            "diversity_penalty_multiplier", "w_subcat", "w_usage", "w_colour",
            "pref_violation_penalty",
            "max_value_share_threshold", "max_value_share_multiplier",
            "size_floor_multiplier", "size_floor_targets",
            "max_composite_score",
        ]
        for key in required:
            assert key in p, f"Missing key: {key}"

    def test_matches_config_values(self):
        p = default_params()
        assert p["same_item_multiplier"] == SAME_ITEM_MULTIPLIER
        assert p["group_concentration_multiplier"] == GROUP_CONCENTRATION_MULTIPLIER
        assert p["group_qty_exponent"] == GROUP_QTY_EXPONENT
        assert p["diversity_penalty_multiplier"] == DIVERSITY_PENALTY_MULTIPLIER
        assert p["pref_violation_penalty"] == PREF_VIOLATION_PENALTY
        assert p["max_composite_score"] == MAX_COMPOSITE_SCORE

    def test_diversity_weights_match(self):
        p = default_params()
        assert p["w_subcat"] == DIVERSITY_WEIGHTS["sub_category"]
        assert p["w_usage"] == DIVERSITY_WEIGHTS["usage"]
        assert p["w_colour"] == DIVERSITY_WEIGHTS["colour"]


# ── _value_penalty parity ──────────────────────────────────────────────────


class TestValuePenaltyParity:
    def test_sweet_spot_zero(self):
        p = default_params()
        assert _value_penalty(115.0, p) == 0.0
        assert value_penalty(115.0) == 0.0

    def test_below_sweet_spot_matches(self):
        p = default_params()
        for vp in [100.0, 105.0, 110.0, 113.5]:
            assert abs(_value_penalty(vp, p) - value_penalty(vp)) < 1e-10

    def test_above_sweet_spot_matches(self):
        p = default_params()
        for vp in [118.0, 120.0, 125.0, 140.0]:
            assert abs(_value_penalty(vp, p) - value_penalty(vp)) < 1e-10


# ── _group_concentration_penalty ──────────────────────────────────────────


class TestGroupConcentrationPenaltyParity:
    def test_below_allowance_zero(self):
        p = default_params()
        # group_totals: [group_load, degree, group_allowance]
        group_totals = [[1, 0.7, 3]]  # load=1, well below allowance=3
        assert _group_concentration_penalty(group_totals, "small", p) == 0.0

    def test_empty_groups_zero(self):
        p = default_params()
        assert _group_concentration_penalty([], "small", p) == 0.0

    def test_above_allowance_penalised(self):
        p = default_params()
        # load=5, degree=0.7, allowance=3 → excess=2
        group_totals = [[5, 0.7, 3]]
        pen = _group_concentration_penalty(group_totals, "small", p)
        expected = (2 ** p["group_qty_exponent"]) * 0.7 * p["group_concentration_multiplier"]
        assert abs(pen - expected) < 0.01

    def test_matches_scoring_module(self, make_item, make_box, make_result):
        """group_totals-based penalty matches object-based penalty from _scoring.py."""
        items = [
            make_item(id=1, name="Apples - Royal Gala", price=400,
                      fungible_group="apple", fungible_degree=0.7, overage=10),
            make_item(id=2, name="Apples - Granny Smith", price=350,
                      fungible_group="apple", fungible_degree=0.7, overage=10),
            make_item(id=3, name="Bananas - Cavendish", price=300,
                      fungible_group="banana", fungible_degree=1.0, overage=10),
        ]
        box = make_box(tier="small", allocations={1: 2, 2: 2, 3: 3})
        result = make_result(items=items, boxes=[box])

        # Build group_totals with [group_load, degree, group_allowance]
        groups: dict[str, dict] = {}
        for item_id, qty in box.allocations.items():
            item = result.items[item_id]
            if item.fungible_group:
                key = item.fungible_group
                degree = item.fungible_degree
            else:
                key = f"__item_{item_id}"
                degree = 1.0
            allowance = _resolve_item_allowance(item, box.tier)
            capped_qty = min(qty, allowance)
            if key not in groups:
                groups[key] = {"load": 0, "degree": degree}
            groups[key]["load"] += capped_qty

        group_totals = []
        for key, gdata in groups.items():
            if key in GROUP_ALLOWANCES:
                ga = GROUP_ALLOWANCES[key].get(box.tier, gdata["load"])
                group_totals.append([gdata["load"], gdata["degree"], ga])

        p = default_params()
        expected = group_concentration_penalty_for_box(box, result)
        actual = _group_concentration_penalty(group_totals, "small", p)
        assert abs(actual - expected) < 1e-10


# ── _diversity_penalty parity ──────────────────────────────────────────────


class TestDiversityPenaltyParity:
    def test_matches_helpers_module(self, make_item, make_box, make_result):
        """Precomputed dim_ratios produce the same penalty as compute_diversity_score."""
        items = [
            make_item(id=1, sub_category="tropical", usage_type="snacking",
                      colour="yellow", shape="long"),
            make_item(id=2, sub_category="root_veg", usage_type="cooking",
                      colour="orange", shape="long"),
            make_item(id=3, sub_category="pome_fruit", usage_type="snacking",
                      colour="red", shape="round"),
        ]
        box = make_box(allocations={1: 1, 2: 1, 3: 1})
        result = make_result(items=items, boxes=[box])
        avail_tags = compute_available_tags(result)

        # Compute via helpers module
        div_score = compute_diversity_score(box, result, avail_tags)

        # Build precomputed features
        tc = _box_tag_counts(box, result)
        dim_ratios = {}
        dim_available = {}
        for dim in ["sub_category", "usage", "colour", "shape"]:
            n_avail = len(avail_tags.get(dim, set()))
            dim_available[dim] = n_avail
            dim_counts = tc.get(dim, {})
            if n_avail > 0 and dim_counts:
                eff = _effective_species(dim_counts)
                dim_ratios[dim] = eff / n_avail
            elif n_avail == 0:
                dim_ratios[dim] = 1.0  # full marks
            else:
                dim_ratios[dim] = 0.0

        p = default_params()
        expected_pen = (1.0 - div_score) * DIVERSITY_PENALTY_MULTIPLIER
        actual_pen = _diversity_penalty(dim_ratios, dim_available, p)
        assert abs(actual_pen - expected_pen) < 1e-10


# ── _same_item_penalty parity ─────────────────────────────────────────────


class TestSameItemPenaltyParity:
    def test_matches_scoring_module(self, make_item, make_box, make_result):
        """item_quantities-based penalty matches object-based penalty."""
        items = [
            make_item(id=1, name="Apples - Royal Gala", price=400,
                      fungible_group="apple", fungible_degree=0.7, overage=10),
            make_item(id=2, name="Bananas - Cavendish", price=300,
                      fungible_group="banana", fungible_degree=1.0, overage=10),
        ]
        box = make_box(tier="small", allocations={1: 4, 2: 3})
        result = make_result(items=items, boxes=[box])

        item_quantities = []
        for item_id, qty in box.allocations.items():
            item = result.items[item_id]
            allowance = _resolve_item_allowance(item, box.tier)
            item_quantities.append([qty, item.price, allowance])

        p = default_params()
        expected = same_item_penalty_for_box(box, result)
        actual = _same_item_penalty(item_quantities, p)
        assert abs(actual - expected) < 1e-10

    def test_within_allowance_zero(self):
        p = default_params()
        assert _same_item_penalty([[1, 400, 2]], p) == 0.0

    def test_empty_items_zero(self):
        p = default_params()
        assert _same_item_penalty([], p) == 0.0


# ── _max_value_share_penalty parity ───────────────────────────────────────


class TestMaxValueSharePenaltyParity:
    def test_below_threshold_zero(self):
        p = default_params()
        assert _max_value_share_penalty(0.10, p) == 0.0

    def test_above_threshold_positive(self):
        p = default_params()
        threshold = p["max_value_share_threshold"]
        pen = _max_value_share_penalty(threshold + 0.1, p)
        expected = 0.1 * p["max_value_share_multiplier"]
        assert abs(pen - expected) < 1e-10


# ── _size_floor_penalty parity ────────────────────────────────────────────


class TestSizeFloorPenaltyParity:
    def test_above_target_zero(self):
        p = default_params()
        target = p["size_floor_targets"]["small"]
        assert _size_floor_penalty(target + 5, "small", p) == 0.0

    def test_below_target_positive(self):
        p = default_params()
        target = p["size_floor_targets"]["small"]
        pen = _size_floor_penalty(target - 3, "small", p)
        expected = 3 * p["size_floor_multiplier"]
        assert abs(pen - expected) < 1e-10


# ── rescore_box ─────────────────────────────────────────────────────────────


class TestRescoreBox:
    def _make_feature(self, **overrides) -> dict:
        feature = {
            "offer_id": 100,
            "box_name": "test@example.com",
            "tier": "small",
            "source": "manual",
            "value_pct": 115.0,
            "group_totals": [],
            "item_quantities": [],
            "max_value_share": 0.15,
            "total_size_points": 12,
            "dim_ratios": {"sub_category": 0.5, "usage": 0.5, "colour": 0.5, "shape": 0.5},
            "dim_available": {"sub_category": 5, "usage": 3, "colour": 4, "shape": 3},
            "pref_violations": 0,
        }
        feature.update(overrides)
        return feature

    def test_sweet_spot_value_zero_penalty(self):
        f = self._make_feature(value_pct=115.0)
        p = default_params()
        r = rescore_box(f, p)
        assert r["value_pen"] == 0.0

    def test_high_value_penalty(self):
        f = self._make_feature(value_pct=70.0)
        p = default_params()
        r = rescore_box(f, p)
        assert r["value_pen"] > 50.0

    def test_pref_violations_penalised(self):
        f = self._make_feature(pref_violations=2)
        p = default_params()
        r = rescore_box(f, p)
        assert r["pref_pen"] == 2 * p["pref_violation_penalty"]

    def test_all_components_sum_to_total(self):
        f = self._make_feature(
            value_pct=110.0, pref_violations=1,
            group_totals=[[4, 0.7, 3]],
            item_quantities=[[3, 400, 2]],
            max_value_share=0.35,
            total_size_points=5,
        )
        p = default_params()
        r = rescore_box(f, p)
        expected = (r["value_pen"] + r["si_pen"] + r["gc_pen"] +
                    r["diversity_pen"] + r["mvs_pen"] + r["sf_pen"] + r["pref_pen"])
        assert abs(r["box_penalty"] - expected) < 1e-10


# ── rescore_offer ───────────────────────────────────────────────────────────


class TestRescoreOffer:
    def _make_feature(self, **overrides) -> dict:
        feature = {
            "offer_id": 100,
            "box_name": "test@example.com",
            "tier": "small",
            "source": "manual",
            "value_pct": 115.0,
            "group_totals": [],
            "item_quantities": [],
            "max_value_share": 0.15,
            "total_size_points": 12,
            "dim_ratios": {"sub_category": 0.8, "usage": 0.8, "colour": 0.8, "shape": 0.8},
            "dim_available": {"sub_category": 5, "usage": 3, "colour": 4, "shape": 3},
            "pref_violations": 0,
        }
        feature.update(overrides)
        return feature

    def test_empty_offer_max_score(self):
        p = default_params()
        r = rescore_offer([], p)
        assert r["score"] == p["max_composite_score"]

    def test_score_decomposition(self):
        """Score = max - sum of all penalties."""
        f1 = self._make_feature(box_name="a@test", value_pct=110.0)
        f2 = self._make_feature(box_name="b@test", value_pct=120.0)
        p = default_params()
        r = rescore_offer([f1, f2], p)
        expected = (p["max_composite_score"] - r["value_pen"] - r["si_pen"] -
                    r["gc_pen"] - r["diversity_pen"] - r["mvs_pen"] -
                    r["sf_pen"] - r["pref_pen"])
        assert abs(r["score"] - expected) < 1e-10


# ── rescore_offer parity with _scoring.total_penalty ────────────────────────


class TestRescoreOfferParity:
    def test_matches_total_penalty(self, sample_items, make_box, make_result, make_charity):
        """rescore_offer should match total_penalty for the same box configuration."""
        box1 = make_box(name="a@test", tier="small", allocations={1: 2, 2: 1, 3: 1})
        box2 = make_box(name="b@test", tier="small", allocations={4: 1, 5: 2, 1: 1})
        charity = [make_charity()]
        result = make_result(items=sample_items, boxes=[box1, box2], charity=charity)
        avail_tags = compute_available_tags(result)

        # Compute via _scoring.total_penalty
        tp = total_penalty(result, avail_tags)

        # Build features and compute via rescore_offer
        features = []
        for box in result.boxes:
            # Build item_quantities
            item_quantities = []
            for item_id, qty in box.allocations.items():
                if qty > 0 and item_id in result.items:
                    item = result.items[item_id]
                    allowance = _resolve_item_allowance(item, box.tier)
                    item_quantities.append([qty, item.price, allowance])

            # Build group_totals with [group_load, degree, group_allowance]
            groups: dict[str, dict] = {}
            for item_id, qty in box.allocations.items():
                if qty > 0 and item_id in result.items:
                    item = result.items[item_id]
                    if item.fungible_group:
                        key = item.fungible_group
                        degree = item.fungible_degree
                    else:
                        key = f"__item_{item_id}"
                        degree = 1.0
                    allowance = _resolve_item_allowance(item, box.tier)
                    capped_qty = min(qty, allowance)
                    if key not in groups:
                        groups[key] = {"load": 0, "degree": degree}
                    groups[key]["load"] += capped_qty

            group_totals = []
            for key, gdata in groups.items():
                if key in GROUP_ALLOWANCES:
                    ga = GROUP_ALLOWANCES[key].get(box.tier, gdata["load"])
                    group_totals.append([gdata["load"], gdata["degree"], ga])

            # Build dim_ratios and dim_available
            tc = _box_tag_counts(box, result)
            dim_ratios = {}
            dim_available = {}
            for dim in ["sub_category", "usage", "colour", "shape"]:
                n_avail = len(avail_tags.get(dim, set()))
                dim_available[dim] = n_avail
                dim_counts = tc.get(dim, {})
                if n_avail > 0 and dim_counts:
                    eff = _effective_species(dim_counts)
                    dim_ratios[dim] = eff / n_avail
                elif n_avail == 0:
                    dim_ratios[dim] = 1.0
                else:
                    dim_ratios[dim] = 0.0

            # Max value share
            box_value = result.box_value(box)
            max_value_share = 0.0
            if box_value > 0:
                for item_id, qty in box.allocations.items():
                    if qty > 0 and item_id in result.items:
                        share = result.items[item_id].price * qty / box_value
                        if share > max_value_share:
                            max_value_share = share

            # Total size points
            total_size_points = 0
            for item_id, qty in box.allocations.items():
                if qty > 0 and item_id in result.items:
                    total_size_points += result.items[item_id].size * qty

            box_price = BOX_TIERS[box.tier]["price"]
            vp = box_value / box_price * 100 if box_price > 0 else 0.0

            features.append({
                "offer_id": result.offer_id,
                "box_name": box.name,
                "tier": box.tier,
                "source": "manual",
                "value_pct": vp,
                "item_quantities": item_quantities,
                "group_totals": group_totals,
                "max_value_share": max_value_share,
                "total_size_points": total_size_points,
                "dim_ratios": dim_ratios,
                "dim_available": dim_available,
                "pref_violations": 0,
            })

        p = default_params()
        r = rescore_offer(features, p)

        # total_penalty = avg(box_penalties)
        # rescore_offer score = 100 - same penalties
        # So: score = 100 - total_penalty
        expected_score = MAX_COMPOSITE_SCORE - tp
        assert abs(r["score"] - expected_score) < 0.01


# ── compute_objective ───────────────────────────────────────────────────────


class TestComputeObjective:
    def _good_feature(self, offer_id=100, **overrides) -> dict:
        """Feature for a well-packed manual box (at sweet spot, good diversity)."""
        f = {
            "offer_id": offer_id,
            "box_name": "good@test",
            "tier": "small",
            "source": "manual",
            "value_pct": 115.0,
            "group_totals": [],
            "item_quantities": [],
            "max_value_share": 0.10,
            "total_size_points": 15,
            "dim_ratios": {"sub_category": 0.9, "usage": 0.9, "colour": 0.9, "shape": 0.9},
            "dim_available": {"sub_category": 5, "usage": 3, "colour": 4, "shape": 3},
            "pref_violations": 0,
        }
        f.update(overrides)
        return f

    def _bad_feature(self, source="synth_monoculture", **overrides) -> dict:
        """Feature for a degenerate bad box."""
        f = {
            "offer_id": 100,
            "box_name": "bad_box",
            "tier": "small",
            "source": source,
            "value_pct": 115.0,
            "group_totals": [[10, 1.0, 3]],
            "item_quantities": [[10, 100, 1]],
            "max_value_share": 0.80,
            "total_size_points": 2,
            "dim_ratios": {"sub_category": 0.1, "usage": 0.1, "colour": 0.1, "shape": 0.1},
            "dim_available": {"sub_category": 5, "usage": 3, "colour": 4, "shape": 3},
            "pref_violations": 0,
        }
        f.update(overrides)
        return f

    def test_good_manual_high_objective(self):
        """Well-packed manual boxes → high objective."""
        manual = [self._good_feature(offer_id=i) for i in range(85, 90)]
        bad = [self._bad_feature()]
        p = default_params()
        obj = compute_objective(manual, bad, p)
        assert obj is not None
        assert obj > 0  # positive because manual scores well

    def test_monoculture_pruning(self):
        """Monoculture box scoring > 50 → objective is None (pruned)."""
        manual = [self._good_feature()]
        # Monoculture that scores very high (no penalties at all)
        mono = self._bad_feature(
            source="synth_monoculture",
            group_totals=[],
            item_quantities=[],
            max_value_share=0.0,
            total_size_points=20,
            dim_ratios={"sub_category": 1.0, "usage": 1.0, "colour": 1.0, "shape": 1.0},
            dim_available={"sub_category": 0, "usage": 0, "colour": 0, "shape": 0},
        )
        p = default_params()
        obj = compute_objective(manual, [mono], p, prune_monoculture_threshold=50.0)
        # Should be pruned since this monoculture box scores ~100
        assert obj is None

    def test_empty_manual_returns_none(self):
        p = default_params()
        assert compute_objective([], [], p) is None

    def test_no_bad_boxes_still_works(self):
        """Objective works with manual only (no synthetic bad boxes)."""
        manual = [self._good_feature(offer_id=i) for i in range(85, 90)]
        p = default_params()
        obj = compute_objective(manual, [], p)
        assert obj is not None


# ── compute_marginal_deltas ─────────────────────────────────────────────────


class TestComputeMarginalDeltas:
    def _base_feature(self, **overrides) -> dict:
        """Feature for a partially-filled box."""
        f = {
            "offer_id": 100,
            "box_name": "test@test",
            "tier": "medium",
            "source": "manual",
            "value_pct": 105.0,
            "group_totals": [],
            "item_quantities": [],
            "max_value_share": 0.15,
            "total_size_points": 12,
            "dim_ratios": {"sub_category": 0.6, "usage": 0.7, "colour": 0.5, "shape": 0.6},
            "dim_available": {"sub_category": 8, "usage": 3, "colour": 4, "shape": 4},
            "pref_violations": 0,
        }
        f.update(overrides)
        return f

    def test_returns_list_matching_candidates(self):
        """Output length matches number of candidates."""
        base = self._base_feature()
        candidates = [self._base_feature(value_pct=107.0), self._base_feature(value_pct=110.0)]
        deltas = compute_marginal_deltas(base, candidates)
        assert len(deltas) == 2

    def test_improving_candidate_positive_delta(self):
        """Adding an item that improves diversity should give positive delta."""
        base = self._base_feature(
            dim_ratios={"sub_category": 0.3, "usage": 0.4, "colour": 0.3, "shape": 0.3},
        )
        better = self._base_feature(
            dim_ratios={"sub_category": 0.7, "usage": 0.7, "colour": 0.6, "shape": 0.6},
        )
        deltas = compute_marginal_deltas(base, [better])
        assert deltas[0] > 0, "Better diversity should improve score"

    def test_worsening_candidate_negative_delta(self):
        """Adding an item that causes group overload should give negative delta."""
        base = self._base_feature()
        worse = self._base_feature(
            group_totals=[[8, 1.0, 3]],  # heavy group overload
        )
        deltas = compute_marginal_deltas(base, [worse])
        assert deltas[0] < 0, "Group overload should worsen score"

    def test_uses_default_params_when_none(self):
        """Should work without explicit params (uses default_params)."""
        base = self._base_feature()
        same = self._base_feature()
        deltas = compute_marginal_deltas(base, [same], params=None)
        assert len(deltas) == 1
        assert deltas[0] == pytest.approx(0.0)

    def test_empty_candidates_returns_empty(self):
        base = self._base_feature()
        deltas = compute_marginal_deltas(base, [])
        assert deltas == []

    def test_ordering_matches_input(self):
        """Delta at index i corresponds to candidate at index i."""
        base = self._base_feature(value_pct=115.0)  # at sweet spot
        # Candidate 0: slightly over sweet spot (small penalty)
        c0 = self._base_feature(value_pct=118.0)
        # Candidate 1: way over sweet spot (larger penalty)
        c1 = self._base_feature(value_pct=130.0)
        deltas = compute_marginal_deltas(base, [c0, c1])
        # c1 is further from sweet spot, so its delta should be worse
        assert deltas[1] < deltas[0]


# ── Feature extraction helpers ──────────────────────────────────────────────


class TestFeatureSchema:
    def test_required_fields_present(self):
        """Verify the expected feature dict schema."""
        required = [
            "offer_id", "box_name", "tier", "source", "value_pct",
            "group_totals", "item_quantities", "max_value_share",
            "total_size_points", "dim_ratios", "dim_available",
            "pref_violations",
        ]
        feature = {
            "offer_id": 100, "box_name": "test", "tier": "small",
            "source": "manual", "value_pct": 115.0, "group_totals": [],
            "item_quantities": [], "max_value_share": 0.1,
            "total_size_points": 10, "dim_ratios": {}, "dim_available": {},
            "pref_violations": 0,
        }
        for key in required:
            assert key in feature

    def test_group_totals_structure(self):
        """group_totals should be list of [group_load, degree, group_allowance] triples."""
        gt = [[3, 0.7, 5], [2, 1.0, 3]]
        for entry in gt:
            assert len(entry) == 3
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], (int, float))
            assert isinstance(entry[2], (int, float))

    def test_item_quantities_structure(self):
        """item_quantities should be list of [qty, price, item_allowance] triples."""
        iq = [[2, 400, 2], [3, 300, 4]]
        for entry in iq:
            assert len(entry) == 3
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], (int, float))
            assert isinstance(entry[2], (int, float))


# ── extract_box_features ────────────────────────────────────────────────────


class TestExtractBoxFeatures:
    """Tests for scripts/extract_features.extract_box_features."""

    def test_returns_all_required_fields(self, sample_items):
        from scripts.extract_features import extract_box_features

        item_lookup = {item.id: {
            "name": item.name, "price": item.price,
            "category_id": item.category_id, "fungible_group": item.fungible_group,
            "fungible_degree": item.fungible_degree,
            "sub_category": item.sub_category, "usage": item.usage_type,
            "colour": item.colour, "shape": item.shape,
            "size": item.size,
        } for item in sample_items}

        avail_tags = {
            "sub_category": {info["sub_category"] for info in item_lookup.values() if info["sub_category"]},
            "usage": {info["usage"] for info in item_lookup.values() if info["usage"]},
            "colour": {info["colour"] for info in item_lookup.values() if info["colour"]},
            "shape": {info["shape"] for info in item_lookup.values() if info["shape"]},
        }

        allocs = {1: 2, 2: 1, 3: 1, 4: 1}
        f = extract_box_features("test@box", allocs, item_lookup, "small",
                                 avail_tags, 100)
        assert f is not None
        required = [
            "offer_id", "box_name", "tier", "source", "value_pct",
            "item_quantities", "group_totals", "max_value_share",
            "total_size_points", "dim_ratios", "dim_available",
            "pref_violations",
        ]
        for key in required:
            assert key in f, f"Missing key: {key}"

    def test_item_quantities_correct(self, sample_items):
        """Each item should have [qty, price, item_allowance] in item_quantities."""
        from scripts.extract_features import extract_box_features

        item_lookup = {item.id: {
            "name": item.name, "price": item.price,
            "category_id": item.category_id, "fungible_group": item.fungible_group,
            "fungible_degree": item.fungible_degree,
            "sub_category": item.sub_category, "usage": item.usage_type,
            "colour": item.colour, "shape": item.shape,
            "size": item.size,
        } for item in sample_items}

        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}

        allocs = {1: 3, 2: 2}
        f = extract_box_features("test@box", allocs, item_lookup, "small",
                                 avail_tags, 100)
        assert f is not None
        assert len(f["item_quantities"]) == 2
        for entry in f["item_quantities"]:
            assert len(entry) == 3

    def test_max_value_share_computed(self, sample_items):
        """max_value_share should be the highest single-item share of total value."""
        from scripts.extract_features import extract_box_features

        item_lookup = {item.id: {
            "name": item.name, "price": item.price,
            "category_id": item.category_id, "fungible_group": item.fungible_group,
            "fungible_degree": item.fungible_degree,
            "sub_category": item.sub_category, "usage": item.usage_type,
            "colour": item.colour, "shape": item.shape,
            "size": item.size,
        } for item in sample_items}

        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}

        # Item 4 (broccoli) = 500, item 5 (kiwi) = 250
        allocs = {4: 1, 5: 1}
        f = extract_box_features("test@box", allocs, item_lookup, "small",
                                 avail_tags, 100)
        assert f is not None
        total = 500 + 250
        expected_share = 500 / total
        assert abs(f["max_value_share"] - expected_share) < 1e-6

    def test_total_size_points_computed(self, sample_items):
        """total_size_points should sum item.size * qty."""
        from scripts.extract_features import extract_box_features

        item_lookup = {item.id: {
            "name": item.name, "price": item.price,
            "category_id": item.category_id, "fungible_group": item.fungible_group,
            "fungible_degree": item.fungible_degree,
            "sub_category": item.sub_category, "usage": item.usage_type,
            "colour": item.colour, "shape": item.shape,
            "size": item.size,
        } for item in sample_items}

        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}

        allocs = {1: 2, 2: 3}
        f = extract_box_features("test@box", allocs, item_lookup, "small",
                                 avail_tags, 100)
        assert f is not None
        assert f["total_size_points"] == 5  # 2*1 + 3*1

    def test_empty_allocations_returns_none(self, sample_items):
        from scripts.extract_features import extract_box_features

        item_lookup = {item.id: {
            "name": item.name, "price": item.price,
            "category_id": item.category_id, "fungible_group": item.fungible_group,
            "fungible_degree": item.fungible_degree,
            "sub_category": item.sub_category, "usage": item.usage_type,
            "colour": item.colour, "shape": item.shape,
            "size": item.size,
        } for item in sample_items}

        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        f = extract_box_features("test@box", {}, item_lookup, "small", avail_tags, 100)
        assert f is None

    def test_pref_violation_counted(self):
        """Fruit item in veg-only box should count as preference violation."""
        from scripts.extract_features import extract_box_features
        from allocator.config import CATEGORY_FRUIT

        item_lookup = {1: {
            "name": "Test Apple", "price": 400, "category_id": CATEGORY_FRUIT,
            "fungible_group": None, "fungible_degree": 0.0,
            "sub_category": "pome_fruit", "usage": "snacking",
            "colour": "red", "shape": "round", "size": 1,
        }}
        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        f = extract_box_features("test@box", {1: 1}, item_lookup, "small",
                                 avail_tags, 100, preference="veg_only")
        assert f is not None
        assert f["pref_violations"] == 1


# ── Synthetic box generation ────────────────────────────────────────────────


class TestSyntheticBoxes:
    """Tests for scripts/extract_features.generate_synthetic_boxes."""

    def _item_lookup(self):
        from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES
        return {
            1: {"name": "Cheap Apple", "price": 100, "category_id": CATEGORY_FRUIT,
                "fungible_group": "apple", "fungible_degree": 0.7,
                "sub_category": "pome_fruit", "usage": "snacking",
                "colour": "red", "shape": "round", "size": 1},
            2: {"name": "Banana", "price": 200, "category_id": CATEGORY_FRUIT,
                "fungible_group": "banana", "fungible_degree": 1.0,
                "sub_category": "tropical", "usage": "snacking",
                "colour": "yellow", "shape": "long", "size": 1},
            3: {"name": "Carrot", "price": 300, "category_id": CATEGORY_VEGETABLES,
                "fungible_group": None, "fungible_degree": 0.0,
                "sub_category": "root_veg", "usage": "cooking",
                "colour": "orange", "shape": "long", "size": 1},
        }

    def test_generates_multiple_types(self):
        from scripts.extract_features import generate_synthetic_boxes

        lookup = self._item_lookup()
        avail_tags = {
            "sub_category": {"pome_fruit", "tropical", "root_veg"},
            "usage": {"snacking", "cooking"},
            "colour": {"red", "yellow", "orange"},
            "shape": {"round", "long"},
        }
        synths = generate_synthetic_boxes(100, lookup, avail_tags)
        assert len(synths) > 0
        sources = {s["source"] for s in synths}
        assert "synth_monoculture" in sources
        assert "synth_random" in sources

    def test_monoculture_has_single_item(self):
        from scripts.extract_features import generate_synthetic_boxes

        lookup = self._item_lookup()
        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        synths = generate_synthetic_boxes(100, lookup, avail_tags)
        monos = [s for s in synths if s["source"] == "synth_monoculture"]
        for m in monos:
            assert len(m["item_quantities"]) == 1

    def test_value_low_below_target(self):
        from scripts.extract_features import generate_synthetic_boxes

        lookup = self._item_lookup()
        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        synths = generate_synthetic_boxes(100, lookup, avail_tags)
        lows = [s for s in synths if s["source"] == "synth_value_low"]
        for low in lows:
            assert low["value_pct"] < 100.0

    def test_synthetic_features_have_new_fields(self):
        """All synthetic boxes should have the new feature fields."""
        from scripts.extract_features import generate_synthetic_boxes

        lookup = self._item_lookup()
        avail_tags = {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()}
        synths = generate_synthetic_boxes(100, lookup, avail_tags)
        for s in synths:
            assert "item_quantities" in s
            assert "max_value_share" in s
            assert "total_size_points" in s
            assert isinstance(s["item_quantities"], list)
            assert isinstance(s["max_value_share"], float)
            assert isinstance(s["total_size_points"], int)


def test_legacy_synthetic_sequence_remains_five_per_tier(monkeypatch):
    import hashlib
    import json

    import scripts.extract_features as extractor

    # Earlier box-feature tests import this module while temporarily replacing
    # allocator.config.BOX_TIERS. Restore the suite's normal tier fixture so
    # the legacy compatibility digest is independent of test collection order.
    monkeypatch.setattr(extractor, "BOX_TIERS", BOX_TIERS)

    records = extractor.generate_synthetic_boxes(
        100, TestSyntheticBoxes()._item_lookup(),
        {"sub_category": set(), "usage": set(), "colour": set(), "shape": set()},
    )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    assert len(records) == 15
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "b1bb474118ce9f873d07e11493d59dc254c4e25af4a4aee362a838de6c8896e8"
    )


# ── CV split helpers ────────────────────────────────────────────────────────


class TestCVSplitProperties:
    """Test properties that any CV split implementation should satisfy."""

    def _make_features_multi_offer(self):
        """Create features from multiple offers with different tiers."""
        features = []
        for oid in range(85, 95):
            for i in range(3):
                tier = ["small", "medium", "large"][i % 3]
                features.append({
                    "offer_id": oid,
                    "box_name": f"box{i}@offer{oid}",
                    "tier": tier,
                    "source": "manual",
                    "value_pct": 115.0,
                    "group_totals": [],
                    "item_quantities": [],
                    "max_value_share": 0.1,
                    "total_size_points": 10,
                    "dim_ratios": {},
                    "dim_available": {},
                    "pref_violations": 0,
                })
        return features

    def test_no_offer_leakage(self):
        """All boxes from one offer must be in the same fold."""
        features = self._make_features_multi_offer()
        offer_ids = sorted(set(f["offer_id"] for f in features))

        # Simple K-fold at offer level
        k = 3
        folds = [[] for _ in range(k)]
        for i, oid in enumerate(offer_ids):
            folds[i % k].append(oid)

        # Verify: each offer appears in exactly one fold
        all_assigned = []
        for fold in folds:
            all_assigned.extend(fold)
        assert sorted(all_assigned) == sorted(offer_ids)

        # Verify: no overlap between folds
        for i in range(k):
            for j in range(i + 1, k):
                assert not set(folds[i]) & set(folds[j])

    def test_all_boxes_covered(self):
        """Every box should appear in exactly one test fold."""
        features = self._make_features_multi_offer()
        offer_ids = sorted(set(f["offer_id"] for f in features))

        k = 3
        folds = [[] for _ in range(k)]
        for i, oid in enumerate(offer_ids):
            folds[i % k].append(oid)

        # Count boxes in test splits
        test_box_count = 0
        for fold_offers in folds:
            fold_set = set(fold_offers)
            test_box_count += sum(1 for f in features if f["offer_id"] in fold_set)

        assert test_box_count == len(features)


# ── CV splits (actual implementation) ──────────────────────────────────────


class TestMakeCVFolds:
    """Tests for scripts/tune_scoring.make_cv_folds."""

    def _features(self):
        features = []
        for oid in range(85, 100):  # 15 offers
            n_boxes = 2 + (oid % 3)  # 2-4 boxes per offer
            for i in range(n_boxes):
                tier = ["small", "medium", "large"][i % 3]
                features.append({
                    "offer_id": oid,
                    "box_name": f"box{i}@offer{oid}",
                    "tier": tier,
                    "source": "manual",
                    "value_pct": 115.0,
                    "group_totals": [],
                    "item_quantities": [],
                    "max_value_share": 0.1,
                    "total_size_points": 10,
                    "dim_ratios": {},
                    "dim_available": {},
                    "pref_violations": 0,
                })
        return features

    def test_correct_number_of_folds(self):
        from scripts.tune_scoring import make_cv_folds
        features = self._features()
        folds = make_cv_folds(features, k=3)
        assert len(folds) == 3

    def test_no_offer_leakage(self):
        from scripts.tune_scoring import make_cv_folds
        features = self._features()
        folds = make_cv_folds(features, k=5)
        for train, test in folds:
            train_offers = {f["offer_id"] for f in train}
            test_offers = {f["offer_id"] for f in test}
            assert not train_offers & test_offers, "Offer leakage between train and test"

    def test_all_boxes_appear_once_in_test(self):
        from scripts.tune_scoring import make_cv_folds
        features = self._features()
        folds = make_cv_folds(features, k=5)
        test_box_names = []
        for _, test in folds:
            test_box_names.extend(f["box_name"] for f in test)
        assert len(test_box_names) == len(features)
        assert len(set(test_box_names)) == len(features)

    def test_train_test_sizes_reasonable(self):
        from scripts.tune_scoring import make_cv_folds
        features = self._features()
        folds = make_cv_folds(features, k=5)
        for train, test in folds:
            assert len(train) > 0
            assert len(test) > 0
            assert len(train) + len(test) == len(features)
