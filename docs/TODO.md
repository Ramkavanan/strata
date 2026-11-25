# strata-bench: development todo

Goal: anyone can run STRATA against their own context engine with one adapter file,
and get the paper's metrics per operator, budget, checkpoint and family. Reference baselines ship as adapters; product adapters live outside the repository.

Done means: tests pass, the item is reproducible from a fresh clone, and the README
covers it. Order is the build order. Each phase has a gate.

## Phase 0: repo, governance, licence

- [ ] Create `strata-bench` (public later; local until reviewed). Python 3.11, `uv`, `pyproject.toml`, `ruff`, `pytest`, `mypy --strict` on `strata/`.
- [ ] Licence: Apache-2.0 for code; paper stays CC BY 4.0; dataset CC BY 4.0 with a datasheet.
- [ ] `CITATION.cff` pointing at DOI 10.5281/zenodo.23469830.
- [ ] GitHub Actions: lint, typecheck, unit tests, golden-data regression on every PR. No API keys in CI.
- [ ] `GOVERNANCE.md`: who maintains the private split, how a vendor submits an adapter, disclosure rule for author-affiliated systems (paper Section 10).
- [ ] Zenodo: upload `STRATA_arxiv_source.zip` as version 1.0.1 so the reference implementation the paper cites is on the record.
- Gate: `uv sync && uv run pytest` green on a clean machine.

## Phase 1: engine (the golden semantics)

- [ ] Move `strata_core.py` and `strata_grid.py` into `strata/core.py` and `strata/grid.py` unchanged in behaviour. Fix the `examples/` default path.
- [ ] Convert the `__main__` asserts (Appendix A examples 1 and 2, Table 11 grid) into pytest tests. These are the first golden tests and must never change without a paper erratum.
- [ ] Add fact lineage `F(x,t)` and decision lineage `Λ(d)` as first-class functions with tests (today lineage exists only in the grid tool).
- [ ] Add `delegate` assertion kind behind a flag (Section 11 extension), off by default.
- [ ] Property tests (hypothesis): a perfect system's answer is invariant under `late`; adding a lower-authority later decide never changes valid state; revocation by lower authority is a no-op.
- Gate: 100% branch coverage on `core.py`; the three paper examples reproduce byte for byte.

## Phase 2: schema and golden data

- [ ] Pydantic models for Appendix C: `Scenario`, `Item`, `Response`, `RunManifest`, `Results`. JSON Schema exported to `schema/` and versioned.
- [ ] Question generator: deterministic replay of `Wπ` emits every T1 to T13 item (and T5b) with expected answer, status, `since`, establishing set, difficulty vector, affected flag per operator. Seeded; same seed, same bytes.
- [ ] Difficulty levels L1 to L5 from fixed published thresholds (Section 3.7), computed, never hand-set.
- [ ] Golden scenarios checked into `data/golden/`: Appendix A example 1, example 2, Table 11 grid, plus one hand-authored scenario per archetype (B2B product, incident, customer success) of 20 to 40 sessions.
- [ ] Golden regression test: regenerate all golden items and diff against the checked-in files. Any diff fails CI.
- [ ] Template format for domain experts: YAML world-model template (roles, authority per domain, governance, decision and requirement graph, planted traps) and a validator that checks invariants (every planted drift has a valid spec that differs from the implementation; every cross-domain conflict is incomparable; every open question has no complete answer).
- Gate: `strata generate --golden` is idempotent and CI-checked.

## Phase 3: perturbation, budget and growth

- [ ] Operators as pure functions `W -> Wπ` with a declared affected variable set: `clean`, `stale(τ)`, `contradict_low(k)`, `contradict_cross(k)`, `incomplete(ρ)` (fraction, seeded, two realisations: omitted artifact and truncated artifact marked partial), `late(λ)` (ingestion order only), `noisy(η)` (decoys about similarly named entities).
- [ ] Test per operator: oracle adapter scores identically to clean; `late` ground truth equals clean; `stale` flag target is 1 on every item.
- [ ] Budget enforcement in the harness with a pinned reference tokenizer; budgets 2k, 8k, 32k, 128k; truncation logged.
- [ ] Growth protocol: K=5 checkpoints by ingestion time; fixed question panel; stale-answer and update-latency bookkeeping with censoring at tK.
- [ ] Fractional design (Section 4.4) encoded as three named blocks selectable from the CLI.
- Gate: every operator has a golden fixture and a perfect-system invariance test.

## Phase 4: synthetic rendering with LLMs (where model-generated data is allowed)

Rule: models render prose; they never produce ground truth.

- [ ] Renderer interface: assertion + explicitness level + artifact type -> text span with hidden span id. Artifact types: meeting transcript (speaker turns), chat thread, document with revisions, ticket with status history, release note, customer call, org chart.
- [ ] Explicitness distribution published in config: explicit, hedged, diffuse. Diffuse assertions rendered across two artifacts (later reference plus changed plan).
- [ ] Two renderer model families, assigned at random per scenario, recorded in the manifest. Results stratified by renderer family.
- [ ] Entailment and leakage checker from a third model family: every span entails its assertion; removed or withheld information does not leak into other artifacts. Failing artifacts regenerated, with retry budget.
- [ ] Checker calibration table: precision and recall against a human audit sample, per artifact type and explicitness level. Diffuse spans below the published recall threshold go to human audit.
- [ ] Realism check: annotators judge a mix of rendered and real anonymised transcripts; target near-chance discrimination. Report Cohen's κ.
- [ ] Renders frozen and content-hashed; the hash is in every manifest and results file.
- [ ] Noise corpus generator for `noisy`: unrelated artifacts and near-duplicate decoys, same renderer pipeline.
- [ ] Human review pass: two annotators on a stratified 20% sample for realism, answerability and ambiguity, before any system is run. Disagreements revised or excluded; exclusions logged.
- Gate: one full B2B scenario rendered, checked, hashed and frozen, with the calibration table filled in.

## Phase 5: adapter interface and reference systems

- [ ] `strata/adapters/base.py`: `ContextSystem` Protocol with `ingest(artifacts, checkpoint)`, `context(task, budget) -> list[ContextItem]` (each item carries source span ids), `reset()`, `describe() -> config dict` for the manifest, `usage()` for token and call counts.
- [ ] Adapter conformance test suite any vendor can run: ingests the golden scenario, answers T1 items, returns provenance-tagged items within budget, reports usage. `strata adapter check <name>`.
- [ ] Reference baselines: `no_context`, `long_context` (recency-first, truncated to B), `vector` (chunked, embedded, with and without rerank), `graph` (entities and relations), `agentic` (tool-using search over raw artifacts), `oracle` (establishing set).
- [ ] Modular reference system (Figure 4) with five switchable mechanisms: evidence store, temporal state, authority from org chart, lineage links, validation. Each switch is a CLI flag and appears in the manifest.
- [ ] Response alignment helper: free-text answer -> structured `Response` via one model call; results flagged `attribution: aligned` not `native`.
- [ ] Two fixed reader models from different families in config; a `--own-reader` mode for systems that answer directly (excluded from the budget block and reported as such).
- Gate: oracle scores 100% on every golden scenario under every operator; no_context scores at chance; conformance suite passes for all baselines.

## Phase 6: harness scripts and run tracking

- [ ] `strata_generate.py` scenarios + operators -> `runs/scenarios/{N}_{ts}/` with config snapshot and content hashes.
- [ ] `strata_ingest.py --adapter X --scenario N [--checkpoint k]` -> `runs/ingests/{N}_{ts}/manifest.json` (adapter config, usage, timings, event-time support flag).
- [ ] `strata_evaluate.py --ingest N --block perturbation|budget|growth --runs 3` -> `runs/evaluations/{N}_{ts}/results.json` plus per-item `responses.jsonl`.
- [ ] `strata_report.py --evaluation N [--compare M]` -> tables and plots per operator, family, budget, checkpoint, affected vs unaffected; paired differences with intervals.
- [ ] Retry with exponential backoff and jitter; bounded concurrency; checkpoint and resume on ingestion; infrastructure failures retried three times then recorded as failures, never dropped.
- [ ] `.env.example`, `--dry-run`, structured logs (JSONL), deterministic seeds everywhere, `--seed` on every script.
- [ ] `.claude/commands/strata-bench/SKILL.md` so Claude Code can run and compare runs.
- Gate: a full three-block run on the golden B2B scenario with the oracle and vector baselines completes from a clean clone with one command sequence documented in the README.

## Phase 7: scoring and statistics

- [ ] Metrics module, each with a unit test against a hand-computed fixture: state accuracy (value, status, since within one session), fact lineage ordered recall and exact chain, decision lineage identifier recall and exact chain, drift F1, question-resolution F1, open-inventory set F1, evidence precision and recall, evidence recency, abstention balanced accuracy, staleness-flag balanced accuracy, confident error rate, sensitivity Δπ, AUBC and B90, update latency with censoring, stale-answer rate, fleet consistency and propagation lag (schema only until Phase 10), CPCO with H=1,000 and as a curve.
- [ ] Cost: token and call counts logged per model call; dated price table in `config/prices/`; self-hosted reference rate; unverifiable counts marked self-reported.
- [ ] Statistics: binomial GLMM as in Section 9.2 (statsmodels or a documented R script), discrete-time hazard for update latency, cluster bootstrap over templates with 2,000 replicates, Holm–Bonferroni per hypothesis family, equivalence classification (degraded, improved, equivalent, inconclusive) with the 5-point SESOI.
- [ ] Hypothesis report: H1 to H5 each classified from the results file; H6 to H8 labelled exploratory.
- [ ] No composite score anywhere in the output.
- Gate: every metric has a fixture test; the hypothesis report runs on the golden run and prints "inconclusive" where power is insufficient rather than a verdict.

## Phase 8: vendor adapters

- [ ] Each adapter records in its manifest: event-time support, provenance mode (native or aligned), SDK version, configuration used.
- [ ] Vendor review step before any published number: publish the adapter and config, invite the vendor to review and tune, record their response in the results.
- Gate: conformance suite passes for all three; a full perturbation-block run per adapter on the golden scenario.

## Phase 9: enterprise-grade quality gates

- [ ] Reproducibility: every result file carries dataset version, render hashes, adapter config, reader and judge models, tokenizer, price table, seeds, harness git SHA. `strata verify results.json` re-derives ground truth and checks it.
- [ ] Determinism: regenerating a scenario from its seed reproduces its hashes.
- [ ] Contamination: public dev split with answers, public validation split with hidden answers, private test split of fresh instantiations refreshed per release; generator regularities documented as a known limitation.
- [ ] Privacy: all scenarios synthetic; CI check that no file in `data/` matches an email, phone or real-company denylist; calibration set never in the repo.
- [ ] Security: no secrets in repo (gitleaks in CI); adapters run with keys from env only; harness never executes vendor code outside the adapter module.
- [ ] Versioning: SemVer for harness; dataset versions pinned in results; CHANGELOG.
- [ ] Datasheet for the dataset (Gebru et al.) and a reporting checklist (Appendix D) enforced by `strata report --check`.
- [ ] Reader-model drift: fixed model ids per release; a canary run on each release to detect silent provider changes.
- [ ] Judge calibration (business track, later): extraction agreement with two human raters on a stratified sample, reported per release.
- [ ] Performance: a full three-block run on one scenario finishes within a documented budget of time and cost; concurrency limits configurable.
- Gate: `strata verify` passes on every result file in `runs/`; CI green including the secret and PII scans.

## Phase 10: documentation and release

- [ ] README: quick start (clone, `uv sync`, generate, ingest oracle, evaluate, report) with no API key; then "add your system in one file"; then "read the results" with what each operator measures and how to interpret affected vs unaffected.
- [ ] `docs/adapter-guide.md`, `docs/metrics.md` (formulas with paper section refs), `docs/operators.md`, `docs/statistics.md`, `docs/faq.md` (why no composite score; what contradict-low tests; aligned vs native attribution).
- [ ] Pre-register hypotheses, metrics, models and exclusion rules on OSF before any private-split run; link from README.
- [ ] Release v0.1: engine, operators, generator, template renderer, baselines, QA track, golden scenarios. Tag, Zenodo DOI for the code, datasheet.
- Gate: a person who has not read the paper runs the quick start and adds a toy adapter in under one hour, observed.

## Phase 11: pilot and version 1

- [ ] Pilot: five B2B templates excluded from release, QA consumer, three baselines. Used only to check the power simulation's assumptions.
- [ ] Power analysis by simulation from the fitted model across intra-template correlation 0.05 to 0.30; set template count for 80% power on a 5-point degradation for the vector baseline.
- [ ] Version 1 composition per Table 6 (30 templates, 900 core items, 7 operators, 4 budgets, 5 checkpoints). Business outputs, software engineering, agent and fleet tracks follow as separate releases with their own datasheets.
- [ ] Public leaderboard page generated from results files, with per-operator tables, no composite, and vendor review status shown per row.
- Gate: first results published with the pre-registration link, per-system disclosure, and every number reproducible by `strata verify`.
