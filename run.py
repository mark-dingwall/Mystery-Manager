#!/usr/bin/env python3
"""
CLI entry point for mystery box allocation.

Usage:
    python run.py <offer_id> <shopping_list.xlsx> [options]

Options:
    --no-tui        Skip interactive TUI (use auto-detected boxes as-is)
    --no-llm        Skip LLM review pass
    --parse-notes   Use LLM to parse buyer notes into exclusion rules
    --charity NAME  Add charity recipient (default: CHARITY_NAME env var). Can be repeated.
    --output FILE   Write tab-delimited output to file instead of stdout
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main():
    # No-args: launch the Textual TUI
    if len(sys.argv) == 1:
        log_path = Path(__file__).resolve().parent / "mystery_manager.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            filename=str(log_path),
            filemode="w",
        )
        # Quiet down noisy libraries
        logging.getLogger("paramiko").setLevel(logging.WARNING)
        logging.getLogger("textual").setLevel(logging.WARNING)

        from allocator.app import MysteryManagerApp
        MysteryManagerApp().run()
        return

    parser = argparse.ArgumentParser(
        description="Mystery box allocation tool",
    )
    parser.add_argument("offer_id", type=int, help="Offer ID to allocate for")
    parser.add_argument("xlsx", type=Path, help="Path to tweaked shopping list XLSX")
    parser.add_argument("--no-tui", action="store_true", help="Skip interactive TUI")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM review")
    parser.add_argument("--parse-notes", action="store_true", help="Use LLM to parse buyer notes")
    parser.add_argument(
        "--charity", action="append", default=None,
        help="Charity recipient name (can be repeated)",
    )
    parser.add_argument(
        "--charity-target", type=float, default=None, metavar="DOLLARS",
        help="Override charity target in dollars (bypasses DB calculation)",
    )
    parser.add_argument("--output", type=Path, help="Write output to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--algorithm", default=None,
        help="Allocation algorithm (default: ilp-optimal)",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Quiet down paramiko unless verbose
    if not args.verbose:
        logging.getLogger("paramiko").setLevel(logging.WARNING)

    # Validate input
    if not args.xlsx.exists():
        console.print(f"[red]File not found: {args.xlsx}[/]")
        sys.exit(1)

    from allocator.config import CHARITY_NAME
    charity_names = args.charity or [CHARITY_NAME]

    # Import here to avoid slow startup for --help
    from allocator.allocator import allocate, build_boxes_from_db, print_summary
    from allocator.excel_io import format_output
    from allocator.llm_review import parse_buyer_notes, review_allocation
    from allocator.tui import review_boxes

    # Step 1: Auto-detect boxes from DB
    console.print(f"[bold]Loading boxes for offer {args.offer_id}...[/]")
    boxes = build_boxes_from_db(args.offer_id)
    console.print(f"Found {len(boxes)} mystery boxes")

    # Step 2: Optional LLM note parsing
    if args.parse_notes and not args.no_llm:
        console.print("[bold]Parsing buyer notes with LLM...[/]")
        note_rules = parse_buyer_notes(boxes)
        if note_rules:
            for box_idx, rules in note_rules:
                for rule in rules:
                    boxes[box_idx].exclusions.append(rule)
                    console.print(
                        f"  Added exclusion '{rule.pattern}' to box "
                        f"{boxes[box_idx].name} (from note)"
                    )
        else:
            console.print("  No exclusion rules parsed from notes")

    # Step 3: Interactive TUI review
    if not args.no_tui:
        boxes = review_boxes(boxes, args.offer_id, args.xlsx, charity_names)

    # Step 4: Run allocation
    console.print(f"\n[bold]Running allocation...[/]")
    kwargs = {}
    if args.algorithm:
        kwargs["strategy"] = args.algorithm
    if args.charity_target is not None:
        kwargs["charity_target_cents"] = int(args.charity_target * 100)
    result = allocate(
        offer_id=args.offer_id,
        xlsx_path=args.xlsx,
        boxes=boxes,
        charity_names=charity_names,
        **kwargs,
    )

    # Step 5: Print summary
    print_summary(result)

    # Step 6: Optional LLM review
    if not args.no_llm:
        console.print(f"\n[bold]Running LLM review...[/]")
        review = review_allocation(result)
        if review:
            console.print(f"\n[bold yellow]LLM Review:[/]")
            console.print(review)

    # Step 7: Output
    output_text = format_output(result)

    if args.output:
        args.output.write_text(output_text)
        console.print(f"\n[green]Output written to {args.output}[/]")
    else:
        console.print(f"\n[bold]Tab-delimited output (paste into admin tool):[/]")
        console.print("─" * 60)
        console.print(output_text)
        console.print("─" * 60)

    # Copy to clipboard if possible
    try:
        import subprocess
        process = subprocess.run(
            ["clip.exe"],
            input=output_text,
            text=True,
            timeout=5,
        )
        if process.returncode == 0:
            console.print("[green]Copied to clipboard![/]")
    except Exception:
        pass


if __name__ == "__main__":
    main()
