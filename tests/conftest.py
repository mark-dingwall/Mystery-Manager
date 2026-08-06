"""
Test configuration bootstrap and shared fixtures.

Config bootstrap strategy:
- Set env vars at module level BEFORE any allocator import
- load_dotenv() in config.py won't override existing env vars
- Copy fixture JSON files to project root only if they don't exist (CI)
- This means `from allocator.config import VALUE_SWEET_FROM` gets test values
"""

import importlib
from importlib import metadata as importlib_metadata
import json
import os
import shutil
from pathlib import Path

from packaging.version import Version

# ---------------------------------------------------------------------------
# Module-level env var setup — MUST happen before any allocator import
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Box pricing (synthetic, obviously not real)
os.environ.setdefault("BOX_PRICE_SMALL", "2000")
os.environ.setdefault("BOX_PRICE_MEDIUM", "3500")
os.environ.setdefault("BOX_PRICE_LARGE", "5000")
os.environ.setdefault("BOX_TARGET_PCT", "115")

# Scoring params
os.environ.setdefault("VALUE_SWEET_FROM", "114")
os.environ.setdefault("VALUE_SWEET_TO", "117")
os.environ.setdefault("VALUE_PENALTY_EXPONENT", "1.25")

# Charity
os.environ.setdefault("CHARITY_NAME", "Test Charity")
os.environ.setdefault("CHARITY_BASE_PERCENT", "0.0")
os.environ.setdefault("CHARITY_GIVING_MULTIPLIER", "2.0")

# DB disabled
os.environ.setdefault("SSH_ENABLED", "false")

# ---------------------------------------------------------------------------
# Provision fixture JSON files to project root if absent (CI portability)
# ---------------------------------------------------------------------------

_PROVISIONED: list[Path] = []


def _provision_fixture(filename: str) -> None:
    """Copy fixture file to project root if it doesn't already exist."""
    target = _PROJECT_ROOT / filename
    if not target.exists():
        source = _FIXTURES / filename
        if source.exists():
            shutil.copy2(source, target)
            _PROVISIONED.append(target)


_provision_fixture("scoring_config.json")
_provision_fixture("identifiers.json")

# ---------------------------------------------------------------------------
# NOW safe to import allocator modules
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from allocator.config import BOX_TIERS, CATEGORY_FRUIT, CATEGORY_VEGETABLES  # noqa: E402
from allocator.models import (  # noqa: E402
    AllocationResult,
    CharityBox,
    ExclusionRule,
    Item,
    MysteryBox,
)


# ---------------------------------------------------------------------------
# Session-scoped cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _cleanup_provisioned_files():
    """Remove fixture files we provisioned to project root (CI only)."""
    yield
    for path in _PROVISIONED:
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_item():
    """Factory for Item objects with sensible defaults."""

    def _make(
        id: int = 1,
        name: str = "Test Item",
        price: int = 500,
        category_id: int = CATEGORY_FRUIT,
        category_name: str = "fruit",
        size: int = 1,
        pack_order: int = 0,
        overage: int = 5,
        fungible_group: str | None = None,
        fungible_degree: float = 0.0,
        sub_category: str = "pome_fruit",
        usage_type: str = "snacking",
        colour: str = "red",
        shape: str = "round",
        **kwargs,
    ) -> Item:
        return Item(
            id=id,
            name=name,
            price=price,
            category_id=category_id,
            category_name=category_name,
            size=size,
            pack_order=pack_order,
            overage=overage,
            fungible_group=fungible_group,
            fungible_degree=fungible_degree,
            sub_category=sub_category,
            usage_type=usage_type,
            colour=colour,
            shape=shape,
            **kwargs,
        )

    return _make


@pytest.fixture
def make_box():
    """Factory for MysteryBox objects, auto-deriving target_value from BOX_TIERS."""

    def _make(
        name: str = "test@example.com",
        tier: str = "small",
        merged: bool = True,
        target_value: int | None = None,
        allocations: dict | None = None,
        exclusions: list | None = None,
        preference: str | None = None,
        notes: str | None = None,
    ) -> MysteryBox:
        if target_value is None:
            target_value = BOX_TIERS[tier]["target_value"]
        return MysteryBox(
            name=name,
            tier=tier,
            merged=merged,
            target_value=target_value,
            allocations=allocations if allocations is not None else {},
            exclusions=exclusions if exclusions is not None else [],
            preference=preference,
            notes=notes,
        )

    return _make


@pytest.fixture
def make_charity():
    """Factory for CharityBox objects."""

    def _make(
        name: str = "Test Charity",
        allocations: dict | None = None,
        target_value: int = 0,
    ) -> CharityBox:
        return CharityBox(
            name=name,
            allocations=allocations if allocations is not None else {},
            target_value=target_value,
        )

    return _make


@pytest.fixture
def make_result():
    """Factory for AllocationResult objects.

    items can be a list of Items (auto-keyed by id) or a dict.
    """

    def _make(
        offer_id: int = 1,
        boxes: list | None = None,
        charity: list | None = None,
        stock: dict | None = None,
        items: list | dict | None = None,
    ) -> AllocationResult:
        if boxes is None:
            boxes = []
        if charity is None:
            charity = []
        if stock is None:
            stock = {}
        if items is None:
            items_dict = {}
        elif isinstance(items, list):
            items_dict = {item.id: item for item in items}
        else:
            items_dict = items
        return AllocationResult(
            offer_id=offer_id,
            boxes=boxes,
            charity=charity,
            stock=stock,
            items=items_dict,
        )

    return _make


@pytest.fixture
def sample_items(make_item):
    """5 diverse items: 2 fruit (1 fungible apple), 2 veg, 1 non-fungible fruit."""
    return [
        make_item(id=1, name="Apples - Royal Gala", price=400, category_id=CATEGORY_FRUIT,
                  fungible_group="apple", fungible_degree=0.7,
                  sub_category="pome_fruit", usage_type="snacking", colour="red", shape="round"),
        make_item(id=2, name="Bananas - Cavendish", price=300, category_id=CATEGORY_FRUIT,
                  fungible_group="banana", fungible_degree=1.0,
                  sub_category="tropical", usage_type="snacking", colour="yellow", shape="long"),
        make_item(id=3, name="Carrots - Organic", price=350, category_id=CATEGORY_VEGETABLES,
                  sub_category="root_veg", usage_type="cooking", colour="orange", shape="long"),
        make_item(id=4, name="Broccoli", price=500, category_id=CATEGORY_VEGETABLES,
                  sub_category="brassica", usage_type="cooking", colour="green", shape="chunky"),
        make_item(id=5, name="Kiwifruit", price=250, category_id=CATEGORY_FRUIT,
                  sub_category="tropical", usage_type="snacking", colour="green", shape="round"),
    ]


@pytest.fixture
def two_box_result(sample_items, make_box, make_result, make_charity):
    """Ready-to-use result with 5 items + 2 empty small boxes + 1 charity."""
    boxes = [
        make_box(name="box1@test.example", tier="small"),
        make_box(name="box2@test.example", tier="small"),
    ]
    charity = [make_charity()]
    return make_result(items=sample_items, boxes=boxes, charity=charity)


# ---------------------------------------------------------------------------
# Diagnostics dependency gating
# ---------------------------------------------------------------------------
#
# Under ``--strict-diagnostics-deps`` a missing dependency must FAIL the run; a
# skip would let the whole diagnostic suite vanish silently. Outside explicit
# strict mode it skips normally, so plain pytest still passes on a checkout that
# does not have the isolated diagnostics stack installed.

_STRICT = False

_DIAGNOSTIC_DEPENDENCIES = {
    # module name: (distribution name, minimum version)
    "interpret": ("interpret", "0.6.0"),
    "statsmodels": ("statsmodels", "0.14.0"),
    "sklearn": ("scikit-learn", "1.3.0"),
    "numpy": ("numpy", "1.24.0"),
    "pandas": ("pandas", "2.0.0"),
    "numexpr": ("numexpr", "2.8.4"),
    "bottleneck": ("bottleneck", "1.3.6"),
}


def pytest_addoption(parser):
    diagnostics = parser.getgroup("diagnostics")
    diagnostics.addoption(
        "--strict-diagnostics-deps",
        action="store_true",
        default=False,
        help="fail collection when a diagnostic dependency is absent or too old",
    )


def pytest_configure(config):
    global _STRICT
    _STRICT = config.getoption("--strict-diagnostics-deps")


def require_dep(name: str):
    """Import `name` and enforce known floors; skip unless strict mode is armed."""
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        message = f"required diagnostic dependency {name!r} is not importable"
        if _STRICT:
            raise ImportError(message) from exc
        pytest.skip(message, allow_module_level=True)

    requirement = _DIAGNOSTIC_DEPENDENCIES.get(name)
    if requirement is None:
        return module
    distribution, minimum = requirement
    try:
        installed = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        message = f"required diagnostic distribution {distribution!r} is absent"
        if _STRICT:
            raise ImportError(message) from exc
        pytest.skip(message, allow_module_level=True)
    if Version(installed) < Version(minimum):
        message = (
            f"required diagnostic dependency {distribution}>={minimum}; "
            f"found {installed}"
        )
        if _STRICT:
            raise ImportError(message)
        pytest.skip(message, allow_module_level=True)
    return module
