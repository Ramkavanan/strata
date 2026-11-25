from dataclasses import replace
from datetime import date

from strata.grid import as_of_ingestion, fact_lineage, load_grid

GRID = "data/golden/table11_grid.csv"


def test_table11_truth(root):
    w, facts, meta = load_grid(f"{root}/{GRID}")
    now = max(m["ingested"] for m in meta)
    expected = {
        "Primary datastore": "MySQL",
        "Service architecture": "Modular monolith",
        "Release target": "15 Oct",
        "Payment gateway": "Stripe",
        "On-call owner": "Payments team",
        "Data residency": "EU only",
    }
    for f, v in expected.items():
        st = w.valid_state(f, now)
        assert st["status"] == "valid" and st["values"] == v


def test_table11_traps(root):
    w, _, meta = load_grid(f"{root}/{GRID}")
    now = max(m["ingested"] for m in meta)
    assert [c["value"] for c in fact_lineage(w, "Data residency", now)] == [
        "US and EU",
        "EU only",
    ]  # D2 is older than S4
    assert [c["value"] for c in fact_lineage(w, "Release target", now)] == [
        "30 Sep",
        "15 Oct",
        "31 Oct",
        "15 Nov",
        "15 Oct",
    ]


def test_authority_variant_changes_truth(root):
    w2, _, meta = load_grid(f"{root}/{GRID}", authority={"team": {"member": 2, "cto": 3, "engineer": 1}})
    roles = dict(w2.roles)
    roles["spk_4"] = "cto"
    roles["spk_6"] = "engineer"
    w2 = replace(w2, roles=roles)
    assert w2.valid_state("Primary datastore", max(m["ingested"] for m in meta))["values"] == "PostgreSQL"


def test_growth_by_ingestion_order(root):
    w, _, meta = load_grid(f"{root}/{GRID}")
    now = max(m["ingested"] for m in meta)
    assert as_of_ingestion(w, meta, date(2026, 8, 5)).valid_state("Data residency", now)["status"] == "undetermined"
