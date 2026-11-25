"""Invariants of the engine, checked with hypothesis."""

from dataclasses import replace
from datetime import date, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from strata.core import Assertion, WorldModel

AUTH = {"d": {"low": 1, "mid": 2, "high": 3}}
ROLES = {"p_low": "low", "p_mid": "mid", "p_high": "high"}


def world(decisions):
    return WorldModel(
        roles=ROLES,
        authority=AUTH,
        governance={"x": {"d"}},
        assertions=[
            Assertion(f"a{i}", date(2026, 1, 1) + timedelta(days=d), p, "d", "decide", "x", v)
            for i, (p, d, v) in enumerate(decisions)
        ],
    )


dec = st.lists(
    st.tuples(st.sampled_from(list(ROLES)), st.integers(0, 60), st.sampled_from("ABCD")), min_size=1, max_size=8
)


@settings(max_examples=150, deadline=None)
@given(dec, st.integers(1, 30), st.sampled_from("EFGH"))
def test_lower_authority_later_never_changes_valid_state(decisions, gap, value):
    w = world(decisions)
    t = date(2026, 6, 1)
    base = w.valid_state("x", t)
    top = max(w.undominated("x", t), key=lambda a: w.auth(a))
    lower = [p for p, r in ROLES.items() if AUTH["d"][r] < w.auth(top)]
    if not lower:
        return
    inj = Assertion("inj", min(t, top.t + timedelta(days=gap)), lower[0], "d", "decide", "x", value)
    after = replace(w, assertions=w.assertions + [inj]).valid_state("x", t)
    assert after["status"] == base["status"] and after["values"] == base["values"]


@settings(max_examples=100, deadline=None)
@given(dec)
def test_valid_state_is_independent_of_assertion_order(decisions):
    w = world(decisions)
    t = date(2026, 6, 1)
    rev = replace(w, assertions=list(reversed(w.assertions)))
    assert w.valid_state("x", t)["values"] == rev.valid_state("x", t)["values"]
    assert w.valid_state("x", t)["status"] == rev.valid_state("x", t)["status"]


@settings(max_examples=100, deadline=None)
@given(dec)
def test_revocation_by_lower_authority_is_a_no_op(decisions):
    w = world(decisions)
    t = date(2026, 6, 1)
    top = max(w.undominated("x", t), key=lambda a: w.auth(a))
    lower = [p for p, r in ROLES.items() if AUTH["d"][r] < w.auth(top)]
    if not lower:
        return
    rev = Assertion("rev", t, lower[0], "d", "revoke", "x", None, (top.id,))
    assert replace(w, assertions=w.assertions + [rev]).valid_state("x", t) == w.valid_state("x", t)


@settings(max_examples=100, deadline=None)
@given(dec)
def test_stale_cut_equals_state_as_of_the_cut(decisions):
    w = world(decisions)
    tau = date(2026, 2, 15)
    cut = replace(w, assertions=[a for a in w.assertions if a.t <= tau])
    assert cut.valid_state("x", date(2026, 6, 1)) == w.valid_state("x", tau)
