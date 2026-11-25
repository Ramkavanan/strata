# strata

STRATA benchmarks AI reasoning over evolving organizational knowledge, authority, time, and attribution. It evaluates state awareness, decision lineage, requirement drift, and uncertainty across models and agents using deterministic world models, controlled perturbations, evidence attribution, and cost-per-correct-outcome metrics.

An evaluation harness for **organizational context engines**: systems that hold what a company decided, who had the authority to decide it, and what is still in force. It implements the benchmark specified in *STRATA: Stress-Testing AI Systems on Evolving Organizational Context* (Ramachandran K, Mithun Chandar V; DOI 10.5281/zenodo.23469830).

Anyone can point it at their own system by writing one adapter file, next to the offline reference systems the paper specifies. Ground truth is computed from a hidden world model by deterministic replay. No model is involved in deciding what is correct.

## Quick start (no API keys)

```bash
uv sync --extra dev
uv run strata validate                                  # templates satisfy the world-model invariants
uv run strata generate                                  # scenarios x 7 operators -> runs/scenarios/1_<ts>/
uv run strata evaluate --adapter reference --block perturbation
uv run strata evaluate --adapter long_context --block perturbation
uv run strata report --evaluation 2 --compare 1         # tables per operator, family, budget, checkpoint
uv run strata verify --evaluation 2                     # re-derive ground truth, re-score every response
```

The three blocks of the paper (Section 4.4) are `--block perturbation|budget|growth|all`. The golden scenarios are small, so the budget block needs small budgets to bind: `--budgets 256,512,1024,2048`.

## Evaluate your own system

```bash
uv run strata adapter-check my_package.my_module:MySystem
uv run strata evaluate --adapter my_package.my_module:MySystem --block all --repeats 3
```

See [docs/adapter-guide.md](docs/adapter-guide.md). The interface is `ingest(artifacts)` and `context(task, budget)`; you return context items that cite the span ids they came from.

## Layout

```
templates/          world-model templates (YAML): roles, authority per domain, governance, decisions, planted traps
data/golden/        golden scenarios, items and content hashes, diffed in CI
strata/             engine (core, grid), operators, renderer, questions, scoring, runner, adapters
runs/               numbered, timestamped runs with config snapshots (gitignored)
  scenarios/N_ts/   renders for every operator + hashes
  evaluations/N_ts/ results.json, responses.jsonl
docs/               adapter guide, operators, metrics, statistics, FAQ, status
.claude/commands/   SKILL.md so Claude Code can drive the harness
```

## Adapters

| name | what it is | needs |
|---|---|---|
| `oracle` | the establishing set; upper bound, reads the hidden items | nothing |
| `reference` | modular reference system (Figure 4) with switches for temporal state, authority, lineage, validation | nothing |
| `long_context`, `bm25`, `no_context` | paper baselines (BM25 stands in for vector retrieval) | nothing |

## What is and is not done


The template renderer is deliberately easy to parse. It validates the pipeline; it is not evidence about real systems.

## Citing

Cite the paper (see `CITATION.cff`). Code: Apache-2.0. Paper and datasets: CC BY 4.0.
