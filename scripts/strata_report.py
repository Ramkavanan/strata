"""Thin wrapper: `uv run scripts/strata_report.py ...` is the same as `strata report ...`."""
import sys

from strata.cli import main

if __name__ == "__main__":
    sys.exit(main(["report", *sys.argv[1:]]))
