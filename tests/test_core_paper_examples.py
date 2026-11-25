"""The paper's Appendix A derivations. These are golden: they must never change without an erratum."""

from datetime import date

from strata.core import Assertion, contradict_cross, contradict_low, incomplete, stale
from strata.examples import example_1, example_2

NOW = date(2026, 7, 31)


def test_example1_clean():
    w = example_1()
    tgt, pri = w.valid_state("sf.target", NOW), w.valid_state("sf.priority", NOW)
    assert tgt["status"] == "valid" and tgt["values"] == "Q4" and tgt["establishing"] == ["a2", "a3"]
    assert pri["status"] == "contested" and pri["establishing"] == ["a4", "a5"]


def test_example1_stale():
    w = stale(example_1(), date(2026, 4, 1))
    assert w.valid_state("sf.priority", NOW)["status"] == "undetermined"
    assert w.valid_state("sf.target", NOW)["values"] == "Q4"


def test_example1_contradict_low_is_ignored():
    w = contradict_low(
        example_1(), [Assertion("inj1", date(2026, 6, 10), "eng", "product_scope", "decide", "sf.priority", "dropped")]
    )
    assert w.valid_state("sf.priority", NOW)["establishing"] == ["a4", "a5"]


def test_example1_contradict_cross_is_contested():
    w = contradict_cross(
        example_1(),
        "sf.target",
        "commercial",
        [Assertion("inj2", date(2026, 6, 10), "sales_head", "commercial", "decide", "sf.target", "Q3")],
    )
    assert w.valid_state("sf.target", NOW)["status"] == "contested"


def test_example1_incomplete_is_undetermined():
    w = incomplete(
        example_1(),
        {"a4", "a5"},
        [
            Assertion(
                "ind1",
                date(2026, 6, 2),
                "pm",
                "product_scope",
                "report",
                "sf.priority",
                "per the priority call last month",
                ("a4",),
            )
        ],
    )
    assert w.valid_state("sf.priority", NOW)["status"] == "undetermined"


def test_example2():
    now = date(2026, 6, 30)
    w = example_2()
    spec, rule = w.valid_state("export.spec", now), w.valid_state("export.arch_rule", now)
    assert spec["status"] == "valid" and spec["establishing"] == ["b1", "b2"]
    assert rule["status"] == "valid" and rule["values"].startswith("exports run only")
    assert w.decision_status("b4", now) == "superseded"
    assert w.decision_status("b1", now) == "superseded"
    assert w.drifted("export.spec", now) is False
    assert w.open_questions(now) == ["b6"]


def test_lineage_and_reversals():
    now = date(2026, 6, 30)
    w = example_2()
    assert [c["value"] for c in w.fact_lineage("export.spec", now)] == [
        "bulk export of all records, CSV",
        "bulk export, CSV, max 50k rows per job",
    ]
    # b4 never took force (an engineer cannot override the principal), so only b1 is "no longer in force"
    assert w.not_in_force(now) == ["b1"]
    lin = w.decision_lineage("b2", now)
    assert lin["reasons"] == ["b1"] and lin["decider"] == "tech_lead"


def test_engine_matches_the_published_reference_implementation(root):
    """strata.core only adds to the Zenodo reference implementation: valid_state agrees on every variable and day."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "paper_core", f"{root}/tests/fixtures/paper_original/strata_core.py.txt", submodule_search_locations=None
    )
    import importlib.machinery

    loader = importlib.machinery.SourceFileLoader(
        "paper_core", f"{root}/tests/fixtures/paper_original/strata_core.py.txt"
    )
    spec = importlib.util.spec_from_loader("paper_core", loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_core"] = mod
    loader.exec_module(mod)
    from datetime import timedelta

    for ours in (example_1(), example_2()):
        theirs = mod.WorldModel(
            roles=ours.roles,
            authority=ours.authority,
            governance=ours.governance,
            assertions=[
                mod.Assertion(a.id, a.t, a.speaker, a.domain, a.kind, a.var, a.value, a.ref, a.complete)
                for a in ours.assertions
            ],
        )
        day = min(a.t for a in ours.assertions)
        while day <= date(2026, 8, 1):
            for var in ours.variables():
                assert ours.valid_state(var, day) == theirs.valid_state(var, day)
            assert ours.open_questions(day) == theirs.open_questions(day)
            day += timedelta(days=9)
