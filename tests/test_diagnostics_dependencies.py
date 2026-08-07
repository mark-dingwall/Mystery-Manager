"""Collection gate for the isolated diagnostics dependency stack."""

import pytest

from tests.conftest import require_dep


pytestmark = pytest.mark.diagnostics

# These calls intentionally happen at module scope: diagnostic modules must not
# collect under a partial or below-floor analysis stack. Bootstrap dependencies
# (pytest and packaging) are validated earlier by pytest_configure().
for _dependency in (
    "interpret",
    "statsmodels",
    "sklearn",
    "numpy",
    "pandas",
    "numexpr",
    "bottleneck",
):
    require_dep(_dependency)


def test_diagnostics_dependency_stack_is_available():
    """The strict diagnostics selector always contains a real checked-in test."""
