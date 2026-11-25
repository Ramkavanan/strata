# Status against the development todo (docs/TODO.md)

Legend: done = implemented and tested; partial = implemented with a stated gap; open = not started.

| Phase | Status | Notes |
|---|---|---|
| 0 Repo, governance | partial | Skeleton, CI, citation, Apache notice done. Open: full Apache text, GOVERNANCE.md, Zenodo 1.0.1 upload of the source zip, public repo. mypy strict is not met: 60 Optional-narrowing errors remain, CI runs it as informational. |
| 1 Engine | done | Paper code moved in; a test proves `valid_state` agrees with the published reference implementation. Fact and decision lineage added. Hypothesis property tests. Open: the `delegate` kind. |
| 2 Schema, golden data | done | Pydantic models, JSON Schema export, seeded generator for all 14 families, difficulty L1 to L5, five golden templates (both paper examples, three archetypes), YAML validator, hash regression in CI. Open: 10 templates per archetype for v1 (needs domain experts). |
| 3 Perturbation, budget, growth | done | Seven operators, budgets with a pinned tokenizer, five-checkpoint growth with censoring, three blocks. |
| 4 LLM rendering | partial | Renderer, deterministic leakage check, LLM entailment check, family assignment, audit sampling, calibration, kappa, realism math are implemented and tested with fake models. Open and needs people and keys: a frozen LLM-rendered corpus, the human audit, the realism study, calibration table. |
| 5 Adapters | partial | Protocol, conformance suite, oracle, reference system with switches, long context, BM25, no context, LLM reader. Open: dense-vector and graph baselines, the agentic baseline. |
| 6 Harness | done | generate, evaluate, report, verify, ablate, golden, adapter-check, grid, schema, SKILL.md. `strata ingest` records a manifest; persisted handles for service-backed adapters are specified but not implemented. |
| 7 Scoring, statistics | partial | All Section 8 metrics, cluster bootstrap, equivalence classes, Holm, tau-b. Open: mixed models, hazard model, power simulation, H3 and H5 computation. |
| 9 Enterprise gates | partial | `verify`, determinism, golden hashes, schema versioning, reporting checklist. Open: private split and refresh process, PII denylist check, price table with real prices, reader-model canary. |
| 10 Docs, release | partial | README, guides, FAQ, SKILL. Open: results notebook, OSF pre-registration, v0.1 tag and DOI, the stranger-adds-an-adapter-in-an-hour test. |
| 11 Pilot, v1 | open | Needs templates, LLM-rendered corpus, and real systems. |

## Findings from building it

- The reference system is 100% on every template renderer and operator, which validates the renderer, parser and scorer by round trip. It also shows the template renderer is too easy to say anything about real systems.
- On the five golden templates the ablation finds temporal state and authority improve robustness under cross-domain contradiction and are equivalent under late and noisy. Most other cells are inconclusive, as expected with five templates.
- Operators that remove information (incomplete) can make a weak system look better, because the correct answer becomes undetermined. Sensitivity must be read next to clean accuracy.
- Item selection had to be hash-based per assertion, otherwise items drifted between operators and broke pairing. Found by a test.
