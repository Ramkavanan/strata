"""Thin wrapper: `uv run scripts/strata_ingest.py ...` is the same as `strata ingest ...`."""
import sys

from strata.cli import main

if __name__ == "__main__":
    sys.exit(main(["ingest", *sys.argv[1:]]))
