# strata-core-v0 (version 0.1.0)

Golden dataset for the STRATA benchmark (DOI 10.5281/zenodo.23469830): organisational histories with a hidden world model, the corpus a context system ingests, and the questions it is scored on.

## Composition

- 10 templates for each of 3 archetypes (b2b_product_team, engineering_incident, customer_success_team), 30 in total.
- Development split: 12 templates, with world models and answers.
- Validation split: 18 scenarios, corpus and questions only. World models, answers and the operator behind each condition code are held by the maintainers.
- Each scenario is rendered under 7 operators (clean, stale, contradict_low, contradict_cross, incomplete, late, noisy).
- A panel of about 30 questions per scenario covers the 14 task families (T1 to T13 and T5b).

| operator | development goldens | T1 | T10 | T11 | T12 | T13 | T2 | T3 | T4 | T5 | T5b | T6 | T7 | T8 | T9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 360 | 14 | 14 | 12 | 12 | 12 | 78 | 140 | 12 | 6 | 12 | 12 | 12 | 12 | 12 |
| stale | 188 | 6 | 13 | 12 | 12 | 12 | 29 | 65 | 4 | 1 | 7 | 5 | 12 | 6 | 4 |
| contradict_low | 420 | 25 | 14 | 12 | 12 | 12 | 78 | 140 | 12 | 55 | 12 | 12 | 12 | 12 | 12 |
| contradict_cross | 415 | 30 | 14 | 12 | 12 | 12 | 99 | 140 | 9 | 6 | 33 | 12 | 12 | 12 | 12 |
| incomplete | 317 | 13 | 14 | 12 | 12 | 12 | 53 | 140 | 7 | 3 | 8 | 9 | 12 | 10 | 12 |
| late | 360 | 14 | 14 | 12 | 12 | 12 | 78 | 140 | 12 | 6 | 12 | 12 | 12 | 12 | 12 |
| noisy | 360 | 14 | 14 | 12 | 12 | 12 | 78 | 140 | 12 | 6 | 12 | 12 | 12 | 12 | 12 |

## Realised scenario shape against the paper's targets (Appendix A.3)

| statistic | target | development min / median / max | validation min / median / max |
|---|---|---|---|
| history days | 180 to 275 | 191 / 210.0 / 258 | 180 / 225.5 / 266 |
| sessions | 40 to 120 | 75 / 85.0 / 92 | 75 / 86.5 / 101 |
| people | 8 to 15 | 10 / 12.0 / 13 | 10 / 12.5 / 14 |
| roles | 5 to 8 | 8 / 8.0 / 8 | 8 / 8.0 / 8 |
| entities | 3 to 5 | 3 / 4.0 / 5 | 3 / 4.5 / 5 |
| variables per entity | 2 to 6 | 4 / 5.0 / 5 | 4 / 5.0 / 5 |
| decisions | 15 to 40 | 17 / 27.0 / 31 | 18 / 27.0 / 32 |
| share superseded or revoked | 0.2 to 0.3 | 0.207 / 0.25 / 0.296 | 0.2 / 0.23 / 0.281 |
| requirements | 5 to 10 | 5 / 6.0 / 8 | 5 / 7.0 / 9 |
| drifted requirements | 1 to 3 | 1 / 3.0 / 3 | 1 / 2.0 / 3 |
| questions | 5 to 10 | 5 / 6.5 / 10 | 5 / 7.5 / 10 |
| open share | 0.4 to 0.6 | 0.429 / 0.5 / 0.6 | 0.4 / 0.5 / 0.571 |
| authority conflicts | 2 to 4 | 2 / 2.5 / 4 | 2 / 3.0 / 4 |
| cross domain conflicts | 1 to 4 | 1 / 2.0 / 2 | 1 / 1.0 / 2 |
| corpus tokens per scenario | 150k to 600k in the paper | 28815 / 30872 / 34477 | 27406 / 31812 / 37390 |

## Files

```
manifest.json                       hashes of every template, render and item set
dev/templates/<id>.yaml             world-model templates (the hidden truth, public for this split)
dev/goldens/<operator>.jsonl        goldens: input, expected_output, context, structured expected answer
validation/corpus/<scenario>/<condition>.jsonl.gz   artifacts to ingest
validation/tasks.jsonl              questions without answers
```

Corpora for the development split are regenerated from the templates and checked against the manifest (`strata dataset verify`). `strata dataset export --format deepeval` writes them as text files with goldens.

## How it was made

World models are produced by a seeded procedural generator (`strata/worldgen.py`) that plants reversals, later statements by lower authorities, cross-domain conflicts, revocations, values that return, drifted requirements and open questions, then rejects any template outside the target shape above. A validator re-derives every planted property from the world model. Artifacts are rendered from templates of text with 10 filler lines per artifact on average and 25 sessions that carry no decision. Questions and answers are computed by deterministic replay. No model is involved at any step.

## Limitations you must keep in mind

- The templates are generated, not written by independent domain experts as the paper requires for version 1. They share the generator's regularities, which a system could learn.
- Artifacts come from a template renderer whose sentences are regular enough to parse. A system that parses them is not thereby shown to handle real organisational text. Model-rendered corpora with a human audit are the missing step.
- Corpora are about 30k tokens, well below the paper's estimate of 150k to 600k. The 2k and 8k budgets bind; 128k does not.
- The validation split hides answers and operator names, but anyone holding the corpora can compare conditions of one scenario. A private test split of fresh instantiations is what protects a leaderboard.
- Authority is modelled as levels per domain with one role per person. Real authority is messier.

## Intended use

Evaluating context and memory systems on authority, time and attribution; regression testing a system between releases; teaching. Not intended as evidence about any vendor until that vendor has reviewed its adapter and configuration.

## Maintenance

Built with harness 0.2.0, generator version 1, public seed 20261019. Validation seed commitment (SHA-256): `f3b9b524ef92bd65f10bfa164a365c7fb167d02f96e129205b8d848c3defe846`.
All scenarios are synthetic and contain no personal data; names are drawn from a fixed list.
Licence: CC BY 4.0.
