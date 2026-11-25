from datetime import date

import pytest

from strata import evaluate
from strata.adapters.base import get_adapter
from strata.conformance import run_conformance
from strata.schema import Task


@pytest.fixture(scope="module")
def att():
    return {
        rd: evaluate(
            "bm25", reader=rd, limit=3, block="perturbation,attribution", operators=["clean", "contradict_cross"]
        )
        for rd in ("reference", "recency")
    }


def test_attribution_identity_holds_exactly(att):
    for res in att.values():
        for v in res["blocks"]["attribution"].values():
            assert 1 - v["y_none"] == pytest.approx(
                v["context_value"] + v["context_loss"] + v["consumer_gap"], abs=1e-12
            )
            assert v["y_oracle"] >= v["y_system"] >= v["y_none"] - 1e-12


def test_attribution_separates_a_context_problem_from_a_consumer_problem(att):
    good, weak = att["reference"]["blocks"]["attribution"]["clean"], att["recency"]["blocks"]["attribution"]["clean"]
    assert (
        good["consumer_gap"] == 0 and good["context_loss"] > 0.05
    )  # authority-aware reader: retrieval is the bottleneck
    assert weak["consumer_gap"] > 0.1 and weak["consumer_gap"] > good["consumer_gap"]  # recency reader: the consumer is
    cross = att["reference"]["blocks"]["attribution"]["contradict_cross"]
    assert cross["context_loss"] > good["context_loss"]  # the operator moves value into loss


def test_attribution_is_skipped_for_systems_that_answer_directly():
    res = evaluate("oracle", limit=1, block="attribution", operators=["clean"])
    assert "skipped" in res["blocks"]["attribution"]


def test_costs_follow_the_tokens_each_architecture_uses():
    c = {
        n: evaluate(n, reader="reference", limit=2, operators=["clean"])["blocks"]["perturbation"]["cost"]["clean"]
        for n in ("long_context", "bm25", "vector", "agentic", "reference")
    }
    assert c["long_context"]["mean_context_tokens"] > 20 * c["bm25"]["mean_context_tokens"]
    assert c["agentic"]["mean_tool_calls"] >= 2 and c["agentic"]["mean_system_input_tokens"] > 0
    assert c["vector"]["mean_ingest_cost_usd"] > 0 and c["long_context"]["mean_ingest_cost_usd"] == 0
    assert c["reference"]["mean_ingest_cost_usd"] > c["vector"]["mean_ingest_cost_usd"]
    assert all(
        v["cpco_usd"] is not None and v["cpco_curve_usd"]["100"] >= v["cpco_curve_usd"]["10000"] for v in c.values()
    )


def test_staleness_flag_is_scored_across_paired_renders():
    ref = evaluate("reference", limit=3, operators=["clean", "stale"])["blocks"]["perturbation"]
    assert ref["staleness_flag_balanced_accuracy"] == 1.0
    weak = evaluate("bm25", reader="recency", limit=3, operators=["clean", "stale"])["blocks"]["perturbation"]
    assert weak["staleness_flag_balanced_accuracy"] == 0.5  # never raises the flag: right on clean, wrong on stale


@pytest.mark.parametrize("name", ["vector", "vector_rerank", "graph", "agentic", "oracle_context"])
def test_new_baselines_pass_conformance(name, capsys):
    assert run_conformance(name, {}) == 0


def test_vector_baseline_accepts_an_external_embedder(scenarios):
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return [[float(len(t) % 7), float(t.count("Q")), 1.0] for t in texts]

    ad = get_adapter("vector", embedder=embed, top_k=5)
    scn = scenarios["paper-a1"]
    ad.ingest(scn.public_artifacts())
    ctx = ad.context(
        Task(
            item_id="x",
            family="T3",
            question='What is "Salesforce integration release target"?',
            query_time=date(2026, 6, 1),
            budget=1000,
        ),
        1000,
    )
    assert calls and 0 < len(ctx) <= 5 and all(c.source_ids for c in ctx)


def test_graph_baseline_follows_references_one_hop(scenarios):
    ad = get_adapter("graph")
    scn = scenarios["paper-a1"]
    ad.ingest(scn.public_artifacts())
    ctx = ad.context(
        Task(
            item_id="x",
            family="T4",
            question='Why did "Salesforce integration release target" change?',
            query_time=date(2026, 6, 1),
            budget=1000,
        ),
        1000,
    )
    text = " ".join(c.text for c in ctx)
    assert "Salesforce integration release target" in text and "Salesforce API rate limit" in text


def test_reference_system_keeps_small_talk_out_of_the_context():
    cost = evaluate("reference", limit=2, operators=["clean"])["blocks"]["perturbation"]["cost"]["clean"]
    assert cost["mean_context_tokens"] < 800
