"""Thin wrapper: `uv run scripts/strata_generate.py ...` is the same as `strata generate ...`."""
import sys

from strata.cli import main

if __name__ == "__main__":
    sys.exit(main(["generate", *sys.argv[1:]]))
