"""Tests for web/comparison.py — box matching and item diff logic."""

from web.comparison import build_box_pairs, compute_item_diff


class TestComputeItemDiff:
    def test_identical_items(self):
        items = {1: 2, 2: 1, 3: 3}
        diff = compute_item_diff(items, items)
        assert diff["added"] == {}
        assert diff["removed"] == {}
        assert diff["changed"] == {}
        assert diff["unchanged"] == {1: 2, 2: 1, 3: 3}

    def test_all_added(self):
        diff = compute_item_diff({}, {1: 2, 2: 1})
        assert diff["added"] == {1: 2, 2: 1}
        assert diff["removed"] == {}
        assert diff["changed"] == {}
        assert diff["unchanged"] == {}

    def test_all_removed(self):
        diff = compute_item_diff({1: 2, 2: 1}, {})
        assert diff["added"] == {}
        assert diff["removed"] == {1: 2, 2: 1}
        assert diff["changed"] == {}
        assert diff["unchanged"] == {}

    def test_mixed_diff(self):
        manual = {1: 2, 2: 3, 3: 1}
        algo = {1: 2, 2: 1, 4: 5}
        diff = compute_item_diff(manual, algo)
        assert diff["unchanged"] == {1: 2}
        assert diff["changed"] == {2: (3, 1)}
        assert diff["removed"] == {3: 1}
        assert diff["added"] == {4: 5}

    def test_empty_both(self):
        diff = compute_item_diff({}, {})
        assert diff == {"added": {}, "removed": {}, "changed": {}, "unchanged": {}}

    def test_qty_change_not_unchanged(self):
        diff = compute_item_diff({1: 3}, {1: 5})
        assert diff["changed"] == {1: (3, 5)}
        assert diff["unchanged"] == {}


class TestBuildBoxPairs:
    def _make_metrics(self, box_name, tier="small", score=85.0, **kwargs):
        m = {
            "box_name": box_name,
            "tier": tier,
            "target_value": 2300,
            "total_value": 2500,
            "value_pct": 115.0,
            "unique_items": 8,
            "fruit_value": 1200,
            "veg_value": 1300,
            "fruit_pct": 48.0,
            "diversity_score": 0.8,
            "same_item_penalty": 0.0,
            "group_concentration_penalty": 0.0,
            "max_value_share": 0.125,
            "max_value_share_penalty": 0.0,
            "total_size_points": 12,
            "size_floor_penalty": 0.0,
            "fungible_dupes": 0,
            "slot_dupes": 0,
            "bad_dupes": 0,
            "pref_violations": 0,
            "score": score,
        }
        m.update(kwargs)
        return m

    def test_exact_match(self):
        manual_m = [self._make_metrics("alice@test.com")]
        algo_m = [self._make_metrics("alice@test.com", score=92.0)]
        manual_items = {"alice@test.com": {1: 2, 2: 1}}
        algo_items = {"alice@test.com": {1: 2, 3: 3}}

        pairs = build_box_pairs(manual_m, algo_m, manual_items, algo_items)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["box_name"] == "alice@test.com"
        assert p["manual"]["metrics"]["score"] == 85.0
        assert p["algo"]["metrics"]["score"] == 92.0
        assert p["diff"]["unchanged"] == {1: 2}
        assert p["diff"]["removed"] == {2: 1}
        assert p["diff"]["added"] == {3: 3}

    def test_unmatched_manual_only(self):
        manual_m = [self._make_metrics("solo@test.com")]
        pairs = build_box_pairs(manual_m, [], {"solo@test.com": {1: 1}}, {})
        assert len(pairs) == 1
        assert pairs[0]["algo"]["metrics"] is None
        assert pairs[0]["diff"]["removed"] == {1: 1}
        assert pairs[0]["diff"]["added"] == {}

    def test_unmatched_algo_only(self):
        algo_m = [self._make_metrics("new@test.com")]
        pairs = build_box_pairs([], algo_m, {}, {"new@test.com": {5: 2}})
        assert len(pairs) == 1
        assert pairs[0]["manual"]["metrics"] is None
        assert pairs[0]["diff"]["added"] == {5: 2}
        assert pairs[0]["diff"]["removed"] == {}

    def test_multiple_boxes_ordering(self):
        m1 = self._make_metrics("a@test.com")
        m2 = self._make_metrics("b@test.com")
        a1 = self._make_metrics("a@test.com")
        a2 = self._make_metrics("b@test.com")
        pairs = build_box_pairs(
            [m1, m2], [a1, a2],
            {"a@test.com": {1: 1}, "b@test.com": {2: 2}},
            {"a@test.com": {1: 1}, "b@test.com": {2: 2}},
        )
        assert len(pairs) == 2
        assert pairs[0]["box_name"] == "a@test.com"
        assert pairs[1]["box_name"] == "b@test.com"

    def test_tier_from_manual_when_both_present(self):
        manual_m = [self._make_metrics("x@test.com", tier="medium")]
        algo_m = [self._make_metrics("x@test.com", tier="medium")]
        pairs = build_box_pairs(manual_m, algo_m, {}, {})
        assert pairs[0]["tier"] == "medium"

    def test_tier_from_algo_when_manual_missing(self):
        algo_m = [self._make_metrics("x@test.com", tier="large")]
        pairs = build_box_pairs([], algo_m, {}, {})
        assert pairs[0]["tier"] == "large"

    def test_empty_inputs(self):
        pairs = build_box_pairs([], [], {}, {})
        assert pairs == []


def test_get_available_algorithms_includes_baselines():
    from allocator.strategies import BASELINE_STRATEGIES
    from web.comparison import get_available_algorithms

    algs = get_available_algorithms()
    assert "ilp-optimal" in algs
    assert set(BASELINE_STRATEGIES).issubset(set(algs))
