"""Thin wrapper: `uv run scripts/strata_evaluate.py ...` is the same as `strata evaluate ...`."""
import sys

from strata.cli import main

if __name__ == "__main__":
    sys.exit(main(["evaluate", *sys.argv[1:]]))
