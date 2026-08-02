import inspect

from allocator.strategies import DEFAULT_STRATEGY


def test_effective_algorithm_defaults_to_canonical():
    import compare

    assert compare._effective_algorithm(None) == "ilp-optimal"
    assert compare._effective_algorithm(None) == DEFAULT_STRATEGY
    assert compare._effective_algorithm("round-robin") == "round-robin"


def test_allocate_default_strategy_is_canonical():
    from allocator.allocator import allocate

    default = inspect.signature(allocate).parameters["strategy"].default
    assert default == "ilp-optimal"
