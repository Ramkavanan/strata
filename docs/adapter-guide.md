# Adding your system

Implement one class with five methods and run `strata adapter-check your.module:YourAdapter`.

```python
from strata.adapters.base import BaseAdapter
from strata.schema import ContextItem, Task

class MySystem(BaseAdapter):
    name = "mysystem"
    reader = "llm"            # "recency" | "reference" | "llm" (your context is prose) ; or own_reader = True and define answer()
    event_time_support = True # you pass the artifact's event_date to your system on ingest
    provenance = "native"     # you return span ids; use "aligned" and the harness matches text to spans (flagged in results)

    def ingest(self, artifacts, checkpoint=0):
        for a in artifacts:                       # a.event_date is when it happened; a.ingest_date when it arrived
            for s in a.spans:                     # s.id is the id your returned items must cite
                my_store.add(s.text, when=a.event_date, meta={"span_id": s.id})

    def context(self, task: Task, budget: int):   # return at most `budget` tokens (reference tokenizer); the harness truncates beyond
        return [ContextItem(text=h.text, source_ids=[h.meta["span_id"]]) for h in my_store.search(task.question)]
```

## What you receive
Artifacts (org chart, meetings, chats, tickets, documents, release notes, calls) with event and ingest dates and partial-processing flags. You never receive the world model, the expected answers or the operator in force.

## Event time and provenance matter
`stale` and `late` only test what your system does with event time. If your API cannot accept it, say so (`event_time_support = False`): a low score on `late` is then a finding, not a harness bug. Likewise, if you cannot return native span ids, the evidence scores are marked aligned.

## Own reader
If your product answers directly, set `own_reader = True` and implement `answer(task) -> Response`. Such systems are excluded from the budget block and reported separately.

## Statefulness
In-process adapters are re-ingested on every evaluation. For a service-backed system implement `handle()` and `resume(handle)` so `strata ingest` can persist the ingested state once and `evaluate` can reuse it.

## Before publishing numbers
Publish the adapter and its configuration and let the vendor review it. Results from a default-configuration adapter the vendor has never seen are easy to dispute.

## Readers

A context system is evaluated through a consumer, the reader. `--reader recency` resolves conflicts by the latest
statement (what most memory systems do); `--reader reference` applies authority and time from the organisation
chart; `--reader llm` uses a model (needs a key and `pip install strata-bench[llm]`). Results name the pair, for
example `bm25+reference`. Run the same system under two readers to see how much of its score is the consumer's.
The attribution block (`--block attribution`) makes this exact: it runs the reader with no context, your context and
oracle context, so every lost point is assigned to the context layer or to the consumer.

## Cost

Set `ingest_price_key` to the price-table entry for tokens your system runs through a model at ingest
(`"embedding-reference"` for embedders, `"reader-reference"` for extraction passes), and count tokens into
`self._usage.input_tokens` as you go. Query cost is derived from the context the harness measured plus whatever your
`usage()` reports. The shipped price table holds reference rates, not vendor prices.
