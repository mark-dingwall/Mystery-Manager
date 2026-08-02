#!/usr/bin/env python3
# STATUS: dev-tool — infrequent maintenance utility; not part of the canonical pipeline.
"""
Standardize historical XLSX filenames to canonical format.

Renames files to `offer_{N}_shopping_list.xlsx`, stripping copy suffixes
like "(1)", "(2)", "copy - 2", "charity check", etc.

For multi-file offers (e.g. three offer_34 variants), prefers "Final" in name,
then largest file size. Does NOT move files between directories.

Usage:
    python3 scripts/standardize_filenames.py          # dry-run (default)
    python3 scripts/standardize_filenames.py --apply  # actually rename
"""

import argparse
import re
from pathlib import Path

HISTORICAL_DIR = Path(__file__).parent.parent / "historical"
OLDER_DIR = HISTORICAL_DIR / "older"

# Manual overrides for filenames that can't be auto-mapped
FILENAME_OVERRIDES = {
    "Week 6 Shopping List.xlsx": None,  # unclear offer mapping, skip
}

# Regex to extract offer ID from various filename patterns
_OFFER_ID_RE = re.compile(r"offer_(\d+)")


def extract_offer_id(filename: str) -> int | None:
    """Extract offer ID from filename."""
    m = _OFFER_ID_RE.search(filename)
    return int(m.group(1)) if m else None


def canonical_name(offer_id: int) -> str:
    """Return the canonical filename for an offer."""
    return f"offer_{offer_id}_shopping_list.xlsx"


def _preference_score(path: Path) -> tuple[int, int]:
    """
    Score a file for preference when multiple files exist for the same offer.
    Higher is better. Returns (has_final, file_size).
    """
    name_lower = path.name.lower()
    has_final = 1 if "final" in name_lower else 0
    # Deprioritize sub-offer variants (not the main allocation)
    if "sub_offer" in name_lower or "sub-offer" in name_lower:
        has_final = -1
    return (has_final, path.stat().st_size)


def scan_directory(directory: Path) -> list[tuple[Path, int | None, str]]:
    """
    Scan a directory for XLSX files.

    Returns [(path, offer_id, canonical_name_or_skip)].
    """
    results = []
    if not directory.exists():
        return results

    for path in sorted(directory.glob("*.xlsx")):
        name = path.name

        # Check manual overrides
        if name in FILENAME_OVERRIDES:
            override = FILENAME_OVERRIDES[name]
            if override is None:
                results.append((path, None, "SKIP"))
                continue
            results.append((path, override, canonical_name(override)))
            continue

        offer_id = extract_offer_id(name)
        if offer_id is None:
            results.append((path, None, "SKIP"))
            continue

        target = canonical_name(offer_id)
        results.append((path, offer_id, target))

    return results


def plan_renames(directory: Path) -> list[tuple[Path, Path, str]]:
    """
    Plan renames for a single directory.

    Returns [(old_path, new_path, reason)] where reason explains the action.
    """
    entries = scan_directory(directory)
    actions = []

    # Group by offer_id
    by_offer: dict[int, list[Path]] = {}
    for path, offer_id, target in entries:
        if offer_id is None or target == "SKIP":
            actions.append((path, path, "SKIP: can't determine offer ID"))
            continue
        by_offer.setdefault(offer_id, []).append(path)

    for offer_id, paths in sorted(by_offer.items()):
        target = canonical_name(offer_id)
        target_path = directory / target

        if len(paths) == 1:
            path = paths[0]
            if path.name == target:
                actions.append((path, target_path, "OK: already canonical"))
            else:
                actions.append((path, target_path, f"RENAME: {path.name} → {target}"))
        else:
            # Multiple files for same offer — pick best one
            ranked = sorted(paths, key=_preference_score, reverse=True)
            winner = ranked[0]
            actions.append((winner, target_path,
                           f"RENAME (best of {len(paths)}): {winner.name} → {target}"))
            for loser in ranked[1:]:
                actions.append((loser, loser,
                               f"KEEP (duplicate, lower priority): {loser.name}"))

    return actions


def main():
    parser = argparse.ArgumentParser(description="Standardize historical XLSX filenames")
    parser.add_argument("--apply", action="store_true",
                       help="Actually rename files (default is dry-run)")
    args = parser.parse_args()

    for directory in [HISTORICAL_DIR, OLDER_DIR]:
        if not directory.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Directory: {directory}")
        print(f"{'='*60}")

        actions = plan_renames(directory)
        rename_count = 0

        for old_path, new_path, reason in actions:
            if reason.startswith("OK:"):
                continue  # don't clutter output with already-canonical files

            print(f"  {reason}")

            if args.apply and reason.startswith("RENAME"):
                if new_path.exists() and old_path != new_path:
                    print(f"    WARNING: {new_path.name} already exists, skipping")
                    continue
                old_path.rename(new_path)
                rename_count += 1
                print(f"    DONE")

        if args.apply:
            print(f"\n  {rename_count} files renamed")
        else:
            renames = sum(1 for _, _, r in actions if r.startswith("RENAME"))
            print(f"\n  {renames} files would be renamed (use --apply to execute)")


if __name__ == "__main__":
    main()
