"""Command-line entry point for the MARKTSIGNAL economy scoring engine."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="marktsignal",
        description=(
            "Economy impact scoring: translate unstructured information into "
            "a context-aware score in [-1, +1]."
        ),
    )
    subcommands = parser.add_subparsers(dest="command")

    info = subcommands.add_parser(
        "info",
        help="show project status and available modules",
    )

    arguments = parser.parse_args(argv)
    if arguments.command == "info":
        print("MARKTSIGNAL — Economy Impact Scoring")
        print()
        print("Pipeline:")
        print("  sources      — News / Social → SourceDocument")
        print("  extraction   — SourceDocument → CanonicalEvent")
        print("  memory       — Event Memory, cutoff-sichere Dedup")
        print("  graph        — World Graph (RAG-Kontext)")
        print("  retrieval    — Context Retrieval (case-gebunden)")
        print("  economy      — EconomyModel → ImpactAssessment.score [-1,+1]")
        print("  output       — auditierbarer Score-Export")
        print()
        print("Training ist separat: python -m training train")
        print("Die KI liefert einen Wert. Was danach damit passiert,")
        print("ist nicht Teil dieses Systems.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
