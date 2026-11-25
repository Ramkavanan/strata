"""`strata` command line: generate, ingest, evaluate, report, verify, golden, adapter check, grid, schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, operators
from .adapters.base import get_adapter
from .golden import check_golden, write_golden
from .questions import generate_items
from .report import check_reporting, render
from .results import summarize
from .runner import BLOCKS, BUDGETS, make_reader, run_attribution, run_block
from .runs import dump_json, new_run, read_manifest, resolve_run, write_manifest
from .scenario import Scenario
from .templates import build_scenario, load_template, template_paths, validate_template
from .verify import verify_evaluation

ROOT = Path(__file__).resolve().parent.parent


def _load_scenarios(scn_run: str) -> list[Scenario]:
    out = []
    for sid in sorted(os.listdir(os.path.join(scn_run, "renders"))):
        with open(os.path.join(scn_run, "renders", sid, "clean.json")) as f:
            out.append(Scenario.from_dict(json.load(f)))
    return out


def cmd_generate(a) -> int:
    dataset = None
    if a.dataset:
        from .datasets import load_dataset

        ds = load_dataset(a.dataset, a.split)
        a.template = [str(p) for p in sorted((ds.path / a.split / "templates").glob("*.yaml"))]
        a.seed, a.instances = ds.manifest["spec"]["instantiation_seed"], 1
        dataset = {"name": ds.name, "version": ds.version, "split": a.split}
    paths = a.template or template_paths(str(ROOT / "templates"))
    hashes, snap = {}, []
    scns = []
    for p in paths:
        tpl = load_template(p)
        errs = validate_template(tpl)
        if errs:
            print(f"{p}: invalid template: {errs}", file=sys.stderr)
            return 2
        if getattr(a, "no_panel", False):
            tpl.pop("panel_size", None)
        for seed in range(a.seed, a.seed + a.instances):
            scns.append(build_scenario(tpl, seed))
        snap.append(p)
    run_dir, man = new_run(
        a.root,
        "scenarios",
        {
            "templates": [os.path.basename(p) for p in paths],
            "seed": a.seed,
            "instances": a.instances,
            "dataset": dataset,
        },
        snapshot=snap,
    )
    for base in scns:
        d = os.path.join(run_dir, "renders", base.id)
        os.makedirs(d)
        for name in operators.OPERATORS:
            r = operators.apply(base, name)
            dump_json(os.path.join(d, f"{name}.json"), r.to_dict())
            hashes[f"{base.id}/{name}"] = {
                "world": r.world_hash(),
                "render": r.render_hash(),
                "n_items": len(generate_items(r)),
            }
    man.hashes = hashes
    write_manifest(run_dir, man)
    print(f"scenario run {man.run_number}: {run_dir}  ({len(scns)} scenarios x {len(operators.OPERATORS)} operators)")
    return 0


def cmd_ingest(a) -> int:
    scn_run = resolve_run(a.root, "scenarios", a.scenario)
    adapter = get_adapter(a.adapter, **_cfg(a))
    scns = _load_scenarios(scn_run)
    handles = {}
    import time

    for s in scns:
        adapter.reset()
        if getattr(adapter, "needs_ground_truth", False):
            adapter.attach(s, generate_items(s))
        t0 = time.perf_counter()
        adapter.ingest(s.public_artifacts(), checkpoint=len(s.checkpoints) - 1)
        handles[s.id] = {"seconds": time.perf_counter() - t0, "handle": getattr(adapter, "handle", lambda: None)()}
    run_dir, man = new_run(
        a.root,
        "ingests",
        {"adapter": adapter.describe(), "ingest": handles},
        parent={"scenarios": os.path.basename(scn_run)},
    )
    print(f"ingest run {man.run_number}: {run_dir}")
    return 0


def run_evaluation(
    root: str,
    scn_run: str,
    adapter_name: str,
    config: dict,
    block: str = "perturbation",
    repeats: int = 1,
    budgets=None,
    reader_kind=None,
    price_table=None,
    operators_=None,
) -> tuple[str, dict]:
    """Run blocks for one (context system, reader) pair over a scenario run and write an evaluation run."""
    scns = _load_scenarios(scn_run)
    adapter = get_adapter(adapter_name, **config)
    reader = make_reader(adapter, reader_kind)
    price_table = price_table or str(ROOT / "config" / "prices" / "2026-11.json")
    prices = json.loads(Path(price_table).read_text())
    blocks_to_run = BLOCKS if block == "all" else tuple(block.split(","))
    out_blocks: dict = {}
    all_resp, all_rows = [], []
    for b in blocks_to_run:
        if b == "attribution":
            sys_evals = out_blocks.get("perturbation", {}).get("evaluations")
            out_blocks[b] = run_attribution(adapter, scns, reader, operators_, system_evals=sys_evals)
        else:
            out_blocks[b] = run_block(
                adapter,
                scns,
                b,
                repeats=repeats if b == "perturbation" else 1,
                budgets=budgets,
                reader=reader,
                operator_names=operators_,
            )
        for ev in out_blocks[b]["evaluations"]:
            op = ev["render_id"].split("(")[0].split("@")[0]
            cond = ev.get("condition")
            for row in ev["rows"]:
                all_rows.append({**row, "block": b, "op": op, "condition": cond})
            if b == "growth":
                recs = [(rec["checkpoint"], 0, rec["response"]) for rec in ev["responses"]]
            else:
                recs = [(None, row["run"], resp) for row, resp in zip(ev["rows"], ev["responses"])]
            for checkpoint, run, resp in recs:
                all_resp.append(
                    {
                        "block": b,
                        "condition": cond,
                        "scenario_id": ev["scenario_id"],
                        "op": op,
                        "render_id": ev["render_id"],
                        "budget": ev.get("budget", 128_000),
                        "run": run,
                        "checkpoint": checkpoint,
                        "response": resp,
                    }
                )
    sman = read_manifest(scn_run)
    reader_desc = getattr(reader, "describe", lambda: {})()
    extra = {
        "harness_version": __version__,
        "git_sha": sman.git_sha,
        "scenario_run": os.path.basename(scn_run),
        "dataset": sman.config.get("dataset"),
        "reader": reader_desc,
        "system": f"{adapter_name}+{reader_desc.get('preset') or reader_desc.get('reader')}",
        "price_table": os.path.basename(price_table),
        "prices": prices,
        "repeats": repeats,
        "scenario_hashes": sman.hashes,
        "blocks_run": list(blocks_to_run),
        "budgets": budgets or BUDGETS,
        "n_templates": len({s.template_id for s in scns}),
    }
    results = summarize(adapter.describe(), out_blocks, extra, prices)
    run_dir, man = new_run(
        root,
        "evaluations",
        {"adapter": adapter_name, "block": block, "repeats": repeats, "reader": reader_kind, "adapter_config": config},
        parent={"scenarios": os.path.basename(scn_run)},
        snapshot=[price_table],
    )
    dump_json(os.path.join(run_dir, "results.json"), results)
    with open(os.path.join(run_dir, "responses.jsonl"), "w") as f:
        for r in all_resp:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    with open(os.path.join(run_dir, "rows.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps({**r, "system": extra["system"]}, sort_keys=True, default=str) + "\n")
    return run_dir, results


def cmd_evaluate(a) -> int:
    scn_run = resolve_run(a.root, "scenarios", a.scenario)
    budgets = [int(x) for x in a.budgets.split(",")] if a.budgets else None
    run_dir, results = run_evaluation(
        a.root,
        scn_run,
        a.adapter,
        _cfg(a),
        a.block,
        a.repeats,
        budgets,
        a.reader,
        a.price_table,
        a.operators.split(",") if a.operators else None,
    )
    print(f"evaluation run {os.path.basename(run_dir).split('_')[0]}: {run_dir}")
    if not a.quiet:
        print(render(results))
    return 0


def cmd_report(a) -> int:
    d = resolve_run(a.root, "evaluations", a.evaluation)
    results = json.loads((Path(d) / "results.json").read_text())
    other = None
    if a.compare:
        other = json.loads((Path(resolve_run(a.root, "evaluations", a.compare)) / "results.json").read_text())
    text = render(results, other)
    if a.check:
        miss = check_reporting(results)
        if miss:
            print("reporting checklist incomplete: " + ", ".join(miss), file=sys.stderr)
            return 3
    print(text)
    return 0


def cmd_verify(a) -> int:
    ev = resolve_run(a.root, "evaluations", a.evaluation)
    parent = read_manifest(ev).parent.get("scenarios")
    scn_run = os.path.join(a.root, "scenarios", parent)
    problems = verify_evaluation(ev, scn_run)
    if problems:
        print("VERIFY FAILED")
        [print(" -", p) for p in problems]
        return 1
    print(f"verified: {ev}")
    return 0


def cmd_golden(a) -> int:
    t, g = str(ROOT / "templates"), str(ROOT / "data" / "golden")
    if a.write:
        write_golden(t, g)
        print("golden data written")
        return 0
    diffs = check_golden(t, g)
    if diffs:
        print("golden data differs:")
        [print(" -", d) for d in diffs]
        return 1
    print("golden data matches")
    return 0


def cmd_adapter_check(a) -> int:
    from .conformance import run_conformance

    return run_conformance(a.name, _cfg(a))


def cmd_dataset(a) -> int:
    from . import datasets as D

    if a.action == "build":
        man = D.build_dataset(a.out, a.private)
        dev = man["splits"]["dev"]
        print(
            f"built {man['name']} {man['version']}: dev {len(dev['templates'])} templates, "
            f"{dev['counts']['clean']['goldens']} clean goldens"
            + (f"; validation {man['splits']['validation']['counts']}" if "validation" in man["splits"] else "")
        )
        return 0
    ds = D.load_dataset(a.name, a.split, a.dataset_root)
    if a.action == "info":
        print((ds.path / "DATASET_CARD.md").read_text())
        return 0
    if a.action == "verify":
        problems = ds.verify()
        if problems:
            print("DATASET VERIFY FAILED")
            [print(" -", p) for p in problems]
            return 1
        print(f"dataset verified: {ds.name} {ds.version} [{a.split}]")
        return 0
    if a.action == "export":
        if a.format != "deepeval":
            print("only --format deepeval is implemented", file=sys.stderr)
            return 2
        w = D.export_deepeval(
            ds,
            a.out or f"export/{ds.name}-{a.split}",
            a.operators.split(",") if a.operators else None,
            corpus=not a.no_corpus,
        )
        print(json.dumps(w, indent=1))
        return 0
    return 2


def cmd_answer(a) -> int:
    from .datasets import load_dataset
    from .submission import answer_split

    ds = load_dataset(a.dataset, a.split)
    adapter = get_adapter(a.adapter, **_cfg(a))
    reader = make_reader(adapter, a.reader)
    recs = answer_split(adapter, ds, reader, limit=a.limit)
    desc = getattr(reader, "describe", lambda: {})()
    run_dir, man = new_run(
        a.root,
        "submissions",
        {
            "adapter": adapter.describe(),
            "reader": desc,
            "dataset": ds.name,
            "version": ds.version,
            "split": a.split,
            "system": f"{a.adapter}+{desc.get('preset') or desc.get('reader')}",
        },
    )
    with open(os.path.join(run_dir, "responses.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    print(
        f"submission {man.run_number}: {run_dir} ({len(recs)} responses). Send responses.jsonl and manifest.json to the maintainers."
    )
    return 0


def cmd_score(a) -> int:
    from .submission import score_submission

    d = resolve_run(a.root, "submissions", a.submission)
    man = read_manifest(d)
    recs = [json.loads(line) for line in open(os.path.join(d, "responses.jsonl")) if line.strip()]
    prices = json.loads(Path(a.price_table).read_text())
    res = score_submission(
        recs,
        a.private,
        man.config["dataset"],
        man.config["adapter"],
        {
            "harness_version": __version__,
            "git_sha": man.git_sha,
            "scenario_run": None,
            "dataset": man.config["dataset"],
            "reader": man.config["reader"],
            "system": man.config["system"],
            "price_table": os.path.basename(a.price_table),
            "prices": prices,
            "repeats": 1,
            "blocks_run": ["perturbation"],
        },
        prices,
    )
    rows = res.pop("rows")
    dump_json(os.path.join(d, "results.json"), res)
    with open(os.path.join(d, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        **r,
                        "block": "perturbation",
                        "op": r["render_id"].split("(")[0],
                        "condition": None,
                        "system": man.config["system"],
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
    print(
        f"scored submission {man.run_number}: coverage {res['coverage']:.1%}, unknown records {res['unknown_records']}"
    )
    print(render(res))
    return 0


def _track_scenarios(a) -> list[Scenario]:
    if getattr(a, "scenario", None):
        scns = _load_scenarios(resolve_run(a.root, "scenarios", a.scenario))
    else:
        from .datasets import load_dataset

        scns = load_dataset(a.dataset, "dev").scenarios()
    return scns[: a.limit] if getattr(a, "limit", None) else scns


def cmd_track(a) -> int:
    from . import tracks as T

    scns = _track_scenarios(a)
    prices = json.loads((ROOT / "config" / "prices" / "2026-11.json").read_text())
    adapter = get_adapter(a.adapter, **_cfg(a))
    reader = make_reader(adapter, a.reader)
    if a.track == "agents":
        agent = {"careful": T.CarefulAgent(), "naive": T.NaiveAgent()}[a.agent]
        res = T.run_agent_track(adapter, scns, agent, reader)
    elif a.track == "business":
        res = T.run_business_track(adapter, scns, reader)
    else:
        res = T.fleet_comparison(lambda: get_adapter(a.adapter, **_cfg(a)), scns, a.reader, prices=prices)
    run_dir, man = new_run(
        a.root, "tracks", {"track": a.track, "adapter": a.adapter, "reader": a.reader, "scenarios": len(scns)}
    )
    dump_json(os.path.join(run_dir, "results.json"), res)
    print(f"track run {man.run_number}: {run_dir}")
    print(json.dumps(res.get("summary", {k: v for k, v in res.items() if k != "rows"}), indent=1, default=str))
    return 0


def cmd_power(a) -> int:
    from .study import pilot_power

    out = pilot_power(a.system, a.reader, replicates=a.replicates)
    run_dir, man = new_run(a.root, "power", {"system": a.system, "reader": a.reader})
    dump_json(os.path.join(run_dir, "power.json"), out)
    print(out["text"])
    print(f"power run {man.run_number}: {run_dir}")
    return 0


def cmd_hypotheses(a) -> int:
    from .hypotheses import hypothesis_report, load_run, render_hypotheses

    runs = [load_run(resolve_run(a.root, "evaluations", int(n))) for n in a.evaluations.split(",")]
    abl = json.loads(Path(a.ablation).read_text()) if a.ablation else None
    fleet = json.loads(Path(a.fleet).read_text()) if a.fleet else None
    print(render_hypotheses(hypothesis_report(runs, abl, fleet)))
    return 0


def cmd_study(a) -> int:
    from .study import run_study

    out_dir = run_study(
        a.root,
        a.systems.split(","),
        a.readers.split(","),
        a.dataset,
        [int(x) for x in a.budgets.split(",")] if a.budgets else None,
        a.limit,
        a.repeats,
    )
    print((Path(out_dir) / "REPORT.md").read_text())
    print(f"study written to {out_dir}")
    return 0


def cmd_ablate(a) -> int:
    from .ablation import run_ablation

    scns = [build_scenario(load_template(p), a.seed) for p in (a.template or template_paths(str(ROOT / "templates")))]
    res = run_ablation(scns)
    print(json.dumps(res, indent=2, sort_keys=True, default=str))
    return 0


def cmd_import(a) -> int:
    """Bring your own data: a grid CSV becomes a template with computed ground truth."""
    from .byod import grid_to_template, write_template

    auth = json.loads(a.authority) if a.authority else None
    tpl = grid_to_template(
        a.grid,
        template_id=a.id,
        authority=auth,
        domain_label=a.domain_label,
        render={"chatter": a.chatter, "filler_sessions": a.filler_sessions} if a.chatter or a.filler_sessions else None,
        panel_size=a.panel_size,
    )
    errs = validate_template(tpl)
    if errs:
        print("the grid does not make a valid template:", file=sys.stderr)
        for e in errs:
            print(" -", e, file=sys.stderr)
        return 2
    out = write_template(tpl, a.out or f"templates/{tpl['id']}.yaml")
    scn = build_scenario(tpl, 0)
    print(
        f"wrote {out}: {len(tpl['variables'])} facts, {len(tpl['assertions'])} assertions, {len(scn.artifacts)} artifacts, "
        f"{len(generate_items(scn))} questions. Next: strata generate --template {out} && strata evaluate --adapter <yours>"
    )
    return 0


def cmd_grid(a) -> int:
    from .grid import render_report

    print(render_report(a.file))
    return 0


def cmd_schema(a) -> int:
    from .schema import export_json_schemas

    for p in export_json_schemas(a.out):
        print(p)
    return 0


def cmd_validate(a) -> int:
    bad = 0
    for p in a.templates or template_paths(str(ROOT / "templates")):
        errs = validate_template(load_template(p))
        print(f"{os.path.basename(p)}: {'ok' if not errs else errs}")
        bad += bool(errs)
    return 1 if bad else 0


def _cfg(a) -> dict:
    return json.loads(a.config) if getattr(a, "config", None) else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="strata", description="STRATA benchmark harness")
    ap.add_argument("--root", default="runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--template", action="append")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--instances", type=int, default=1)
    g.add_argument("--dataset", help="build the scenario run from a golden dataset, for example strata-core-v0")
    g.add_argument("--split", default="dev")
    g.add_argument("--no-panel", action="store_true", help="ask every item, not the panel of about 30 per scenario")
    g.set_defaults(fn=cmd_generate)
    i = sub.add_parser("ingest")
    i.add_argument("--adapter", required=True)
    i.add_argument("--scenario", type=int)
    i.add_argument("--config")
    i.set_defaults(fn=cmd_ingest)
    e = sub.add_parser("evaluate")
    e.add_argument("--adapter", required=True)
    e.add_argument("--scenario", type=int)
    e.add_argument("--config")
    e.add_argument(
        "--block", default="perturbation", help="perturbation | budget | growth | attribution | all, or a comma list"
    )
    e.add_argument(
        "--reader", choices=["recency", "reference", "llm"], help="the consumer; default is the adapter's own"
    )
    e.add_argument("--operators", help="comma list of operators; default all seven")
    e.add_argument("--repeats", type=int, default=1)
    e.add_argument(
        "--budgets",
        help="comma list of token budgets; default 2000,8000,32000,128000 (golden scenarios are smaller: try 256,512,1024,2048)",
    )
    e.add_argument("--price-table", default=str(ROOT / "config" / "prices" / "2026-11.json"))
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("report")
    r.add_argument("--evaluation", type=int)
    r.add_argument("--compare", type=int)
    r.add_argument("--check", action="store_true")
    r.set_defaults(fn=cmd_report)
    v = sub.add_parser("verify")
    v.add_argument("--evaluation", type=int)
    v.set_defaults(fn=cmd_verify)
    gd = sub.add_parser("golden")
    gd.add_argument("--write", action="store_true")
    gd.set_defaults(fn=cmd_golden)
    ac = sub.add_parser("adapter-check")
    ac.add_argument("name")
    ac.add_argument("--config")
    ac.set_defaults(fn=cmd_adapter_check)
    dsp = sub.add_parser("dataset")
    dsp.add_argument("action", choices=["build", "info", "verify", "export"])
    dsp.add_argument("--name", default="strata-core-v0")
    dsp.add_argument("--split", default="dev")
    dsp.add_argument("--dataset-root")
    dsp.add_argument("--out")
    dsp.add_argument("--private", help="directory for hidden validation answers (build only); keep it out of the repo")
    dsp.add_argument("--format", default="deepeval")
    dsp.add_argument("--operators")
    dsp.add_argument("--no-corpus", action="store_true")
    dsp.set_defaults(fn=cmd_dataset)
    an = sub.add_parser("answer")
    an.add_argument("--adapter", required=True)
    an.add_argument("--dataset", default="strata-core-v0")
    an.add_argument("--split", default="validation")
    an.add_argument("--reader", choices=["recency", "reference", "llm"])
    an.add_argument("--config")
    an.add_argument("--limit", type=int)
    an.set_defaults(fn=cmd_answer)
    sco = sub.add_parser("score")
    sco.add_argument("--submission", type=int)
    sco.add_argument("--private", required=True)
    sco.add_argument("--price-table", default=str(ROOT / "config" / "prices" / "2026-11.json"))
    sco.set_defaults(fn=cmd_score)
    tr = sub.add_parser("track")
    tr.add_argument("track", choices=["agents", "business", "fleet"])
    tr.add_argument("--adapter", required=True)
    tr.add_argument("--reader", choices=["recency", "reference", "llm"])
    tr.add_argument("--agent", default="careful", choices=["careful", "naive"])
    tr.add_argument("--dataset", default="strata-core-v0")
    tr.add_argument("--scenario", type=int)
    tr.add_argument("--limit", type=int)
    tr.add_argument("--config")
    tr.set_defaults(fn=cmd_track)
    pw = sub.add_parser("power")
    pw.add_argument("--system", default="vector")
    pw.add_argument("--reader", default="reference", choices=["recency", "reference"])
    pw.add_argument("--replicates", type=int, default=300)
    pw.set_defaults(fn=cmd_power)
    hy = sub.add_parser("hypotheses")
    hy.add_argument("--evaluations", required=True, help="comma list of evaluation run numbers")
    hy.add_argument("--ablation")
    hy.add_argument("--fleet")
    hy.set_defaults(fn=cmd_hypotheses)
    st = sub.add_parser("study")
    st.add_argument("--systems", default="no_context,long_context,bm25,vector,vector_rerank,graph,agentic,reference")
    st.add_argument("--readers", default="recency,reference")
    st.add_argument("--dataset", default="strata-core-v0")
    st.add_argument("--budgets")
    st.add_argument("--limit", type=int, help="use only the first N scenarios")
    st.add_argument("--repeats", type=int, default=1)
    st.set_defaults(fn=cmd_study)
    ab = sub.add_parser("ablate")
    ab.add_argument("--template", action="append")
    ab.add_argument("--seed", type=int, default=0)
    ab.set_defaults(fn=cmd_ablate)
    im = sub.add_parser("import", help="bring your own data: grid CSV -> template with computed ground truth")
    im.add_argument("grid")
    im.add_argument("--out")
    im.add_argument("--id")
    im.add_argument("--authority", help='JSON role -> level, e.g. \'{"engineer": 1, "cto": 3}\'')
    im.add_argument("--domain-label", default="the team")
    im.add_argument("--chatter", type=int, default=0)
    im.add_argument("--filler-sessions", type=int, default=0)
    im.add_argument("--panel-size", type=int)
    im.set_defaults(fn=cmd_import)
    gr = sub.add_parser("grid")
    gr.add_argument("file")
    gr.set_defaults(fn=cmd_grid)
    sc = sub.add_parser("schema")
    sc.add_argument("--out", default="schema")
    sc.set_defaults(fn=cmd_schema)
    va = sub.add_parser("validate")
    va.add_argument("templates", nargs="*")
    va.set_defaults(fn=cmd_validate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
