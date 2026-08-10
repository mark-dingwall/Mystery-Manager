"""DB-free roster reconciliation for hard-negative generation."""

import copy
from dataclasses import dataclass
from typing import Iterable, Sequence

from allocator.box_features import stable_hash
from allocator.config import BOX_TIERS, PER_OFFER_BOX_SIZE_OVERRIDES
from allocator.models import MysteryBox


@dataclass(frozen=True)
class RosterMatch:
    """The preserved CSV and DB spellings for one shared roster identity."""

    csv_name: str
    db_name: str


@dataclass(frozen=True)
class RosterIntersection:
    """The deterministic overlap and differences between CSV and DB rosters."""

    matches: tuple[RosterMatch, ...]
    csv_only: tuple[str, ...]
    db_only: tuple[str, ...]


class AmbiguousRosterIdentityError(ValueError):
    """A source contains names that collide after case normalization."""


def _casefold_unique(names: Iterable[str], source: str) -> dict[str, str]:
    """Return normalised identity -> original spelling, rejecting ambiguity."""
    result = {}
    for name in names:
        identity = name.casefold()
        if identity in result:
            raise AmbiguousRosterIdentityError(
                f"{source} has case-normalised collision for {identity!r}: "
                f"{result[identity]!r}, {name!r}"
            )
        result[identity] = name
    return result


def correct_box_tiers(offer_id: int, boxes: Sequence[MysteryBox]) -> list[MysteryBox]:
    """Copy boxes, apply case-normalised email overrides, and sort deterministically."""
    corrected_boxes = copy.deepcopy(list(boxes))
    overrides = PER_OFFER_BOX_SIZE_OVERRIDES.get(str(offer_id), {})
    override_names = _casefold_unique(overrides, "tier override")
    for box in corrected_boxes:
        original_override_name = override_names.get(box.name.casefold())
        tier = overrides.get(original_override_name) if original_override_name else None
        if original_override_name:
            if tier not in BOX_TIERS:
                raise ValueError(
                    f"invalid tier override for offer {offer_id}, "
                    f"{original_override_name!r}: {tier!r}"
                )
            box.tier = tier
            box.target_value = BOX_TIERS[tier]["target_value"]
    return sorted(
        corrected_boxes, key=lambda box: (box.target_value, box.name.casefold())
    )


def intersect_roster(
    csv_box_names: Iterable[str], db_box_names: Iterable[str]
) -> RosterIntersection:
    """Reconcile one-to-one roster names while preserving each source's spelling."""
    csv_by_identity = _casefold_unique(csv_box_names, "CSV roster")
    db_by_identity = _casefold_unique(db_box_names, "DB roster")

    matches = tuple(
        RosterMatch(
            csv_name=csv_by_identity[identity],
            db_name=db_by_identity[identity],
        )
        for identity in sorted(csv_by_identity.keys() & db_by_identity.keys())
    )
    csv_only = tuple(
        csv_by_identity[identity]
        for identity in sorted(csv_by_identity.keys() - db_by_identity.keys())
    )
    db_only = tuple(
        db_by_identity[identity]
        for identity in sorted(db_by_identity.keys() - csv_by_identity.keys())
    )
    return RosterIntersection(matches=matches, csv_only=csv_only, db_only=db_only)


def roster_config_hash() -> str:
    """Stamp the complete shared override mapping."""
    return stable_hash(PER_OFFER_BOX_SIZE_OVERRIDES)
