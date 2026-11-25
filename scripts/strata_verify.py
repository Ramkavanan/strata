"""Thin wrapper: `uv run scripts/strata_verify.py ...` is the same as `strata verify ...`."""
import sys

from strata.cli import main

if __name__ == "__main__":
    sys.exit(main(["verify", *sys.argv[1:]]))
