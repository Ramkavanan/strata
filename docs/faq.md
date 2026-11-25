# FAQ

**Why no composite score?** A single number conceals the differences the benchmark exists to expose.

**Why does the template renderer parse so easily?** It exists to run with no model calls and to validate the pipeline by round trip. A reader that parses template text is not evidence of anything about real systems. Realism claims require the LLM renderers, the entailment checker and the human audit in strata/llm_render.py.

**Can a system see the answers?** No. Artifacts carry opaque span ids. The span-to-assertion map, expected answers and the operator in force stay in the harness. The oracle adapter reads them on purpose and is a diagnostic.

**Is the vendor adapter score a fair verdict on the vendor?** Not until the vendor has reviewed the adapter and configuration. The adapters are unverified against live services.
