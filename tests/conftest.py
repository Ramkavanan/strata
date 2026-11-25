import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strata.templates import build_scenario, load_template, template_paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES = template_paths(os.path.join(ROOT, "templates"))


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def scenarios():
    return {load_template(p)["id"]: build_scenario(load_template(p), 0) for p in TEMPLATES}


@pytest.fixture(scope="session")
def northwind(scenarios):
    return scenarios["b2b-northwind-01"]
