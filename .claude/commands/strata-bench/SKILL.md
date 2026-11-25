---
name: strata-bench
---

# STRATA bench

Evaluates organizational context engines on authority, time and attribution. Ground truth is computed from a hidden world model, never from a model. All commands run from the repo root with `uv run strata ...` (or `scripts/strata_*.py`).

## Datasets
`strata dataset info|verify|export|build`. The shipped golden dataset is strata-core-v0 (development split: 12 templates with
answers; validation split: 18 scenarios, answers held by the maintainers). `strata answer --split validation` produces a
submission; `strata score --private DIR` scores it. From Python: `from strata import load_dataset, evaluate`.

## Pipeline
1. `strata validate` checks templates against the world-model invariants.
2. `strata generate [--template T] [--seed N] [--instances K]` writes `runs/scenarios/N_ts/` (renders for all 7 operators, hashes).
3. `strata evaluate --adapter NAME --block perturbation|budget|growth|all [--repeats 3] [--budgets 256,512,1024,2048]` writes `runs/evaluations/N_ts/{results.json,responses.jsonl}`.
4. `strata report [--evaluation N] [--compare M] [--check]` prints tables per operator, family, budget and checkpoint. There is no composite score.
5. `strata verify [--evaluation N]` re-derives ground truth and re-scores every response. Run it before quoting any number.
6. `strata ablate` runs the H2 mechanism ablation on the reference system.
7. `strata track agents|business|fleet --adapter NAME` runs the other consumer tracks with scripted consumers.
8. `strata study` runs everything for a list of systems and readers and writes REPORT.md with the hypothesis verdicts;
   `strata hypotheses --evaluations 1,2,3` classifies from existing runs; `strata power` sizes the study from the pilot.

## Adapters
`no_context`, `long_context`, `bm25`, `vector`, `vector_rerank`, `graph`, `agentic`, `reference`, `oracle`, `oracle_context` run offline with no keys.

## Reading results
- `--block attribution` runs the same consumer with no context, the system, and oracle context; 1 - Y_none = V + L + G.
- Compare each operator to clean on the same items (delta on affected items with a cluster-bootstrap interval); the class is degraded, improved, equivalent or inconclusive against a 5-point smallest effect.
- A system that returns nothing can look good on `incomplete` because the correct answer there is undetermined. Read sensitivity next to clean accuracy.
- `attribution: aligned` means evidence ids were matched from text, not returned natively.
- The golden scenarios are small. The budget axis only binds at small budgets here; say so when reporting.

## Rules
- Never change a golden hash without understanding why (`strata golden`). Update with `strata golden --write` only for an intended change.
- Do not publish a vendor number before the vendor has reviewed its adapter and configuration.
