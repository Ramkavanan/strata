# Golden datasets

A STRATA dataset is a set of world-model templates and everything derived from them: the corpus a context system
ingests under each operator and the goldens it is scored against. The derivation is deterministic, so the
development split ships templates plus goldens and regenerates corpora on demand against pinned hashes.

## strata-core-v0

| | development split | validation split |
|---|---|---|
| templates | 12 (4 per archetype) | 18 (6 per archetype) |
| what ships | templates, goldens, hashes | corpora and questions only |
| answers | public | held by the maintainers |
| operators | 7 renders per scenario | 7 conditions per scenario, named by opaque codes |
| questions | about 30 per scenario, all 14 families | same, with opaque ids |

Archetypes: B2B product team, engineering incident, customer success team. The dataset card in
`data/datasets/strata-core-v0/DATASET_CARD.md` reports the realised shape of every scenario against the paper's
Appendix A.3 targets, and the limitations.

## A golden

```json
{"id": "b2b-00-s0/T3/e1.priority", "input": "What is the current state of \"Salesforce integration priority\", and is it settled?",
 "expected_output": "Contested between P0 and P2 since 2026-05-30: two authorities disagree and neither can overrule the other.",
 "context": ["On 2026-04-22, Dev Rao (cto) decided in product scope: ...", "On 2026-05-30, Maya Iyer (sales_head) noted in commercial terms: ..."],
 "family": "T3", "capability": "state", "operator": "clean", "query_time": "2026-07-31",
 "expected": {"status": "contested", "values": ["P0", "P2"], "since": "2026-05-30", "establishing": ["a014", "a019"], "support": [...]},
 "evidence_ids": ["sp-3f2a...", "sp-9c1d..."], "affected": false, "difficulty": 4, "explicitness": "diffuse"}
```

`input`, `expected_output` and `context` are the three fields DeepEval's goldens carry. `expected` is the structured
ground truth STRATA scores against; `expected_output` is a sentence derived from it for human readers and judges.

## Commands

```bash
strata dataset info                                     # the dataset card
strata dataset verify [--split validation]              # re-derive and compare every pinned hash
strata dataset export --format deepeval --out export/   # goldens as DeepEval JSON, corpus as text files
strata generate --dataset strata-core-v0                # a scenario run from the development split
strata answer --adapter mysystem --split validation     # answer the hidden split; send responses.jsonl to the maintainers
strata score --submission N --private <dir>             # maintainers: score against the private answers
strata dataset build --private <dir>                    # rebuild (maintainers); the private dir holds the validation seed
```

From Python:

```python
from strata import load_dataset, evaluate
ds = load_dataset()                                 # development split
for g in ds.goldens("contradict_low")[:5]:
    print(g.input, "->", g.expected_output)
results = evaluate("bm25", reader="reference")      # every block, in memory
```

## How the hidden split stays hidden

The validation templates come from a secret seed whose SHA-256 is published in the manifest. Scenario ids,
condition codes and task ids are opaque hashes, so a file of questions says nothing about the operator in force
or the statements an operator injected. Anyone holding the corpora can still compare conditions of one scenario,
which is why a private test split of fresh instantiations is what protects a leaderboard.

## Why the questions are a panel

Every scenario supports 80 to 110 questions. A panel of about 30 keeps a run affordable and matches the paper
(Appendix A.3). The panel is chosen on the clean render and kept across operators, so comparisons are paired. State
questions about the variables the operators target are chosen first, because they are the affected items the primary
test rests on; the rest fill one item per family in rounds. An operator's own probes, such as a question about a
statement it injected, are added on top. `strata generate --no-panel` asks everything.

## What a generated template cannot give you

The templates are produced by a seeded generator from a fixed set of motifs, entities and values. They are
consistent, planted traps are verified, and nothing passes through a model. They are not the templates the paper
asks for in version 1, which are written by at least two independent domain experts per archetype. A system that
learns the generator's regularities will look better here than on real histories. Use the dataset for comparing
systems and for regression testing; treat any absolute number as provisional.
