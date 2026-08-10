"""Data models for mystery box allocation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    """An item available for allocation (from DB + XLSX overage)."""

    id: int
    name: str
    price: int  # cents per unit
    category_id: int  # part_category_id (2=fruit, 3=veg)
    category_name: str  # "fruit" or "vegetables"
    size: int  # pack size weight/volume indicator
    pack_order: int
    overage: int  # available qty from shopping list
    fungible_group: str | None = None  # e.g. "apple", "tomato"
    fungible_degree: float = 0.0  # 0.0 = not fungible, 1.0 = near-identical
    sub_category: str = ""  # e.g. "pome_fruit", "citrus", "root_veg"
    usage_type: str = ""  # "snacking", "cooking", "salad", "garnish"
    colour: str = ""  # "green", "red", "orange_yellow", "purple", "white_brown"
    shape: str = ""  # "round", "long", "small", "leafy", "chunky"
    buy_qty: int | None = None
    buy_price: int | None = None  # cost in cents
    components: list | None = None  # bundle components

    @property
    def value(self) -> int:
        """Retail value of one unit in cents."""
        return self.price


@dataclass
class ExclusionRule:
    """A rule that excludes items from a box."""

    pattern: str  # text pattern to match against item name (case-insensitive)
    source: str  # "preference", "note", or "manual"

    def matches(self, item: Item) -> bool:
        return self.pattern.lower() in item.name.lower()


@dataclass
class MysteryBox:
    """A mystery box to be filled with items."""

    name: str  # display name or email
    tier: str  # "small", "medium", "large"
    merged: bool  # True if merged with customer's regular order
    target_value: int  # target total value in cents
    allocations: dict[int, int] = field(default_factory=dict)  # item_id -> qty
    exclusions: list[ExclusionRule] = field(default_factory=list)
    preference: str | None = None  # structured preference from buy_options
    notes: str | None = None  # free-text note_to_seller

    # Categories the customer already has in their regular order (for merged boxes)
    existing_categories: set[int] = field(default_factory=set)

    @property
    def allocated_value(self) -> int:
        """Total value of allocated items in cents. Requires items dict."""
        # This is computed externally since we need access to item prices
        return 0

    def is_excluded(self, item: Item) -> bool:
        """Check if an item is excluded from this box by any rule."""
        # Structured preference exclusions
        from allocator.config import CATEGORY_FRUIT, CATEGORY_VEGETABLES

        if self.preference == "fruit_only" and item.category_id != CATEGORY_FRUIT:
            return True
        if self.preference == "veg_only" and item.category_id != CATEGORY_VEGETABLES:
            return True

        # Manual/note exclusion rules
        return any(rule.matches(item) for rule in self.exclusions)


@dataclass
class CharityBox:
    """A charity recipient box."""

    name: str
    allocations: dict[int, int] = field(default_factory=dict)  # item_id -> qty
    target_value: int = 0  # computed from charity formula


@dataclass
class AllocationResult:
    """Complete allocation result for an offer."""

    offer_id: int
    boxes: list[MysteryBox]
    charity: list[CharityBox]
    stock: dict[int, int]  # item_id -> qty (leftover for next week)
    items: dict[int, Item]  # item_id -> Item (reference)
    # None means ilp-optimal was never invoked for this result.
    solver_status: str | None = None

    def box_value(self, box: MysteryBox) -> int:
        """Compute total allocated value for a box."""
        return sum(
            self.items[item_id].price * qty
            for item_id, qty in box.allocations.items()
            if item_id in self.items
        )

    def charity_value(self, charity: CharityBox) -> int:
        """Compute total allocated value for a charity box."""
        return sum(
            self.items[item_id].price * qty
            for item_id, qty in charity.allocations.items()
            if item_id in self.items
        )

    def total_allocated_qty(self, item_id: int) -> int:
        """Total qty allocated across all boxes + charity for an item."""
        total = 0
        for box in self.boxes:
            total += box.allocations.get(item_id, 0)
        for c in self.charity:
            total += c.allocations.get(item_id, 0)
        total += self.stock.get(item_id, 0)
        return total

    def remaining_overage(self, item_id: int) -> int:
        """Unallocated overage for an item."""
        if item_id not in self.items:
            return 0
        return self.items[item_id].overage - self.total_allocated_qty(item_id)
