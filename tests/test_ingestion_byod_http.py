import csv
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from strata.adapters.base import get_adapter
from strata.byod import grid_to_template
from strata.conformance import run_conformance
from strata.datasets import export_deepeval, load_dataset
from strata.operators import OPERATORS, apply
from strata.questions import generate_items
from strata.runner import evaluate_render
from strata.schema import Artifact
from strata.templates import build_scenario, validate_template


# --------------------------------------------------------------------------- artifact formats
def test_every_artifact_type_has_a_native_file_form(northwind):
    types = {a.type for a in northwind.artifacts}
    assert {"meeting", "chat", "ticket", "org_chart", "source_code", "release_note", "customer_call"} <= types
    for a in northwind.artifacts:
        text = a.to_file_text()
        assert a.filename and a.mime and text
        if a.type == "chat":
            recs = [json.loads(line) for line in text.splitlines()]
            assert [r["id"] for r in recs] == [s.id for s in a.spans]
        if a.type in ("ticket", "org_chart"):
            rows = list(csv.reader(io.StringIO(text)))
            assert len(rows) == len(a.spans) + 1
        if a.type == "source_code":
            assert a.filename.endswith(".py") and " = " in text


def test_source_code_realises_the_implementation_and_maps_to_it(northwind):
    code = [a for a in northwind.artifacts if a.type == "source_code"]
    assert code
    for a in code:
        aid = northwind.span_map[a.spans[0].id]
        impl = northwind.world._by_id(aid)
        assert impl.kind == "implement" and str(impl.value) in a.text


def test_file_export_writes_native_files_with_a_manifest(tmp_path):
    ds = load_dataset()
    export_deepeval(ds, str(tmp_path), ["clean"], corpus=True)
    scn = ds.scenarios()[0]
    d = tmp_path / "corpus" / scn.id / "clean"
    files = {p.name for p in d.rglob("*") if p.is_file()}
    assert (
        any(f.endswith(".jsonl") for f in files)
        and any(f.endswith(".csv") for f in files)
        and any(f.endswith(".py") for f in files)
    )
    man = [json.loads(line) for line in (d / "MANIFEST.jsonl").read_text().splitlines()]
    assert len(man) == len(ds.corpus(scn.id, "clean")) and all((d / m["file"]).exists() for m in man)


# --------------------------------------------------------------------------- HTTP adapter
class _Store:
    def __init__(self):
        self.docs = []


def _server(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            store.docs.clear()
            self._send({"ok": True})

        def do_POST(self):
            body = self._json()
            if self.path == "/ingest":
                assert body["headers_ok"] == "Bearer secret-123"
                store.docs.append(body)
                return self._send({"ok": True})
            if self.path == "/query":
                words = set(body["question"].lower().replace('"', " ").split())
                hits = []
                for d in store.docs:
                    score = len(words & set(d["text"].lower().split()))
                    if score:
                        hits.append(
                            {
                                "text": d["text"],
                                "meta": {"unit_ids": d["units"], "occurred_at": d["occurred_at"]},
                                "score": score,
                            }
                        )
                hits.sort(key=lambda h: -h["score"])
                return self._send({"results": hits[: body["limit"]]})
            self._send({"error": "no such route"}, 404)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(scope="module")
def http_cfg(monkeypatch_module=None):
    store = _Store()
    srv = _server(store)
    base = f"http://127.0.0.1:{srv.server_port}"
    cfg = {
        "name": "toy-engine",
        "headers": {"X-Ignored": "x"},
        "reset": {"url": base + "/reset", "method": "DELETE"},
        "ingest": {
            "url": base + "/ingest",
            "per": "unit",
            "body": {
                "headers_ok": "Bearer ${TOY_KEY}",
                "text": "{text}",
                "units": "{unit_ids}",
                "occurred_at": "{event_date}",
            },
        },
        "query": {"url": base + "/query", "body": {"question": "{question}", "limit": 12}},
        "results": {
            "items": "results",
            "text": "text",
            "ids": "meta.unit_ids",
            "score": "score",
            "when": "meta.occurred_at",
            "prefix_when": False,
        },
        "reader": "reference",
    }
    yield cfg, store
    srv.shutdown()


def test_http_adapter_from_config_is_a_full_system(http_cfg, monkeypatch, scenarios):
    cfg, store = http_cfg
    monkeypatch.setenv("TOY_KEY", "secret-123")
    ad = get_adapter("http", config=cfg)
    assert ad.describe()["adapter"] == "toy-engine" and ad.describe()["provenance"] == "native"
    assert "Authorization" not in json.dumps(ad.describe())  # secrets never reach the manifest
    scn = scenarios["paper-a1"]
    ev = evaluate_render(ad, scn, 128_000)
    assert len(store.docs) == sum(len(a.spans) for a in scn.artifacts)
    acc = sum(r["correct"] for r in ev["rows"]) / len(ev["rows"])
    assert acc > 0.5 and all(r["attribution"] == "native" for r in ev["rows"])
    assert ev["ingest_usage"]["tool_calls"] > 0


def test_http_adapter_passes_conformance(http_cfg, monkeypatch, tmp_path, capsys):
    cfg, _ = http_cfg
    monkeypatch.setenv("TOY_KEY", "secret-123")
    p = tmp_path / "toy.json"
    p.write_text(json.dumps(cfg))
    assert run_conformance("http", {"config_path": str(p)}) == 0


def test_http_adapter_marks_text_only_results_as_aligned(http_cfg, monkeypatch):
    cfg, _ = http_cfg
    monkeypatch.setenv("TOY_KEY", "secret-123")
    cfg2 = {**cfg, "results": {**cfg["results"], "ids": None}}
    ad = get_adapter("http", config=cfg2)
    assert ad.describe()["provenance"] == "aligned"


# --------------------------------------------------------------------------- bring your own data
def test_grid_import_reproduces_table_11_and_carries_ingestion_metadata(root):
    tpl = grid_to_template(f"{root}/data/golden/table11_grid.csv", template_id="t11", authority={"member": 1})
    assert validate_template(tpl) == []
    scn = build_scenario(tpl, 0)
    w, t = scn.world, scn.query_time
    truth = {w.valid_state(v, t)["values"] for v in w.variables()}
    assert {"MySQL", "Modular monolith", "15 Oct", "Stripe", "Payments team", "EU only"} <= truth
    late = [a for a in scn.artifacts if a.ingest_date > a.event_date]
    assert late and all(a.event_date.isoformat() == "2026-06-30" for a in late)  # D2 arrived late
    assert any(a.status == "partial" for a in scn.artifacts)  # S4 and D2 partial
    assert len(generate_items(scn)) > 20


def test_grid_import_with_roles_changes_the_truth(root):
    tpl = grid_to_template(f"{root}/data/golden/table11_grid.csv", authority={"member": 2, "cto": 3, "engineer": 1})
    tpl["roles"]["spk_4"], tpl["roles"]["spk_6"] = "cto", "engineer"
    scn = build_scenario(tpl, 0)
    assert scn.world.valid_state("primary_datastore", scn.query_time)["values"] == "PostgreSQL"


def test_byod_scenario_runs_every_operator_and_the_reference_is_perfect(root):
    tpl = grid_to_template(f"{root}/data/golden/table11_grid.csv", authority={"member": 1})
    scn = build_scenario(tpl, 0)
    for op in OPERATORS:
        ev = evaluate_render(get_adapter("reference"), apply(scn, op))
        assert all(r["correct"] for r in ev["rows"]), (op, [r["item_id"] for r in ev["rows"] if not r["correct"]])


def test_artifact_schema_round_trips_through_json(northwind):
    for a in northwind.artifacts[:5]:
        assert Artifact.model_validate_json(a.model_dump_json()) == a
