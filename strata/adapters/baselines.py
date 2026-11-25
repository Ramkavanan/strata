"""Reference baselines (Section 7): no context, long context, BM25 retrieval over spans."""

from __future__ import annotations

from ..schema import ContextItem, Task
from .base import BM25, BaseAdapter, words


class NoContext(BaseAdapter):
    """The reader receives only the question: measures what is guessable."""

    name = "no_context"

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        pass

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        return []


class LongContext(BaseAdapter):
    """The corpus in the prompt, most recently ingested artifacts first, truncated to the budget by the harness."""

    name = "long_context"

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        ordered = sorted(self.spans, key=lambda s: (s["ingest"], s["event"], s["id"]), reverse=True)
        return [self._item(s) for s in ordered]


class BM25Retrieval(BaseAdapter):
    """Lexical retrieval over spans (a dependency-free stand-in for vector retrieval).

    Dense vector retrieval is a pluggable embedder away; see docs/adapter-guide.md."""

    name = "bm25"

    def __init__(self, top_k: int = 12, **config):
        super().__init__(top_k=top_k, **config)
        self.top_k = top_k
        self._index = None

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        super().ingest(artifacts, checkpoint)
        self._index = BM25([s["text"] for s in self.spans])

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        if not self.spans:
            return []
        sc = self._index.scores(task.question)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[: self.top_k]
        return [self._item(self.spans[i], sc[i]) for i in order if sc[i] > 0]


# --------------------------------------------------------------------------- vector retrieval
class HashEmbedder:
    """A dependency-free embedding: hashed word unigrams and bigrams with idf weights, L2-normalised.

    It is a real vector space (so the retrieval code path is the one a neural embedder would use), not a neural
    model. Pass `embedder=` to VectorRetrieval to plug a neural one: a callable from a list of texts to a list of
    equal-length float vectors."""

    def __init__(self, dim: int = 2048):
        self.dim = dim
        self.idf: dict[int, float] = {}

    def _features(self, text: str) -> list[int]:
        import zlib

        ws = words(text)
        feats = ws + [f"{a} {b}" for a, b in zip(ws, ws[1:])]
        return [zlib.crc32(f.encode()) % self.dim for f in feats]

    def fit(self, texts: list[str]) -> None:
        import math
        from collections import Counter

        df: Counter = Counter()
        for t in texts:
            df.update(set(self._features(t)))
        n = max(1, len(texts))
        self.idf = {k: math.log(1 + n / (1 + c)) for k, c in df.items()}

    def embed(self, text: str) -> dict[int, float]:
        import math
        from collections import Counter

        tf_ = Counter(self._features(text))
        vec = {k: (1 + math.log(c)) * self.idf.get(k, 1.0) for k, c in tf_.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}


def _cos_sparse(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _cos_dense(a, b) -> float:
    import math

    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


_QUOTED = __import__("re").compile(r'"([^"]+)"')


class VectorRetrieval(BaseAdapter):
    """Vector retrieval over chunked artifacts (one chunk per span). With `rerank=True` the top candidates are
    re-scored lexically (BM25 plus a bonus for phrases the question quotes), as a cross-scoring stage would."""

    name = "vector"
    ingest_price_key = "embedding-reference"

    def __init__(self, top_k: int = 12, rerank: bool = False, embedder=None, **config):
        super().__init__(top_k=top_k, rerank=rerank, **config)
        self.top_k, self.rerank, self.external = top_k, rerank, embedder
        self.hash = HashEmbedder()
        self.vecs: list = []
        self._bm25 = None

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        from ..tokenizer import count_tokens

        before = len(self.spans)
        super().ingest(artifacts, checkpoint)
        texts = [s["text"] for s in self.spans]
        self._usage.input_tokens += sum(count_tokens(t) for t in texts[before:])
        if self.external:
            self.vecs = list(self.external(texts))
        else:
            self.hash.fit(texts)
            self.vecs = [self.hash.embed(t) for t in texts]
        self._bm25 = BM25(texts)

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        if not self.spans:
            return []
        if self.external:
            q = list(self.external([task.question]))[0]
            sc = [_cos_dense(q, v) for v in self.vecs]
        else:
            q = self.hash.embed(task.question)
            sc = [_cos_sparse(q, v) for v in self.vecs]
        n = self.top_k * 4 if self.rerank else self.top_k
        order = sorted(range(len(sc)), key=lambda i: -sc[i])[:n]
        if self.rerank:
            lex = self._bm25.scores(task.question)
            phrases = _QUOTED.findall(task.question)
            score = {i: lex[i] + 2.0 * sum(p in self.spans[i]["text"] for p in phrases) for i in order}
            order = sorted(order, key=lambda i: -score[i])[: self.top_k]
        return [self._item(self.spans[i], sc[i]) for i in order if sc[i] > 0]


class VectorRerank(VectorRetrieval):
    name = "vector_rerank"

    def __init__(self, **config):
        config.setdefault("rerank", True)
        super().__init__(**config)


# --------------------------------------------------------------------------- graph retrieval
class GraphRetrieval(BaseAdapter):
    """Graph retrieval over extracted entities and relations.

    Nodes are the topics and people extracted from spans; an edge joins a topic to every span that states
    something about it, and topics that cite each other are neighbours. A question retrieves the spans of the
    topics it names plus their one-hop neighbours. On template renders the extractor is the span parser; on prose
    renders it would be a model, which is why ingestion is priced as an extraction pass."""

    name = "graph"
    ingest_price_key = "reader-reference"

    def __init__(self, top_k: int = 12, hops: int = 1, **config):
        super().__init__(top_k=top_k, hops=hops, **config)
        self.top_k, self.hops = top_k, hops
        self.topic_spans: dict[str, list[int]] = {}
        self.neighbours: dict[str, set[str]] = {}
        self._bm25 = None

    def reset(self) -> None:
        super().reset()
        self.topic_spans, self.neighbours = {}, {}

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        from .. import textformat as tf
        from ..tokenizer import count_tokens

        before = len(self.spans)
        super().ingest(artifacts, checkpoint)
        for i in range(before, len(self.spans)):
            text = self.spans[i]["text"]
            self._usage.input_tokens += count_tokens(text)
            p = tf.parse_span(text)
            if not p:
                continue
            self.topic_spans.setdefault(p["label"], []).append(i)
            for _tag, _d, ref_label, _name in p["refs"]:
                self.neighbours.setdefault(p["label"], set()).add(ref_label)
                self.neighbours.setdefault(ref_label, set()).add(p["label"])
        self._bm25 = BM25([s["text"] for s in self.spans])

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        if not self.spans:
            return []
        hit = {t for t in self.topic_spans if t in task.question}
        # a longer topic name that contains a shorter one is the more specific match
        hit = {t for t in hit if not any(t != u and t in u for u in hit)}
        frontier = set(hit)
        for _ in range(self.hops):
            frontier |= {n for t in list(frontier) for n in self.neighbours.get(t, ())}
        idx = sorted(
            {i for t in frontier for i in self.topic_spans.get(t, [])},
            key=lambda i: (self.spans[i]["event"], self.spans[i]["id"]),
            reverse=True,
        )
        if not idx:
            sc = self._bm25.scores(task.question)
            idx = [i for i in sorted(range(len(sc)), key=lambda i: -sc[i])[: self.top_k] if sc[i] > 0]
        return [self._item(self.spans[i]) for i in idx]


# --------------------------------------------------------------------------- agentic retrieval
class AgenticRetrieval(BaseAdapter):
    """A tool-using agent that searches the raw artifacts on demand (the pattern LongMemEval-V2 found most accurate
    and slowest). Scripted, so it is deterministic: search the question, search every phrase the question quotes,
    follow the references in what it found, then look up the organisation chart. Each step is a tool call, and the
    text it reads at each step is counted as model input, which is what makes it expensive."""

    name = "agentic"

    def __init__(self, k: int = 8, max_steps: int = 4, **config):
        super().__init__(k=k, max_steps=max_steps, **config)
        self.k, self.max_steps = k, max_steps
        self._bm25 = None

    def ingest(self, artifacts, checkpoint: int = 0) -> None:
        super().ingest(artifacts, checkpoint)
        self._bm25 = BM25([s["text"] for s in self.spans])

    def _read(self, idx: list[int]) -> None:
        from ..tokenizer import count_tokens

        self._usage.tool_calls += 1
        self._usage.model_calls += 1
        self._usage.input_tokens += sum(count_tokens(self.spans[i]["text"]) for i in idx)

    def context(self, task: Task, budget: int) -> list[ContextItem]:
        from .. import textformat as tf

        if not self.spans:
            return []
        body = [i for i, s in enumerate(self.spans) if s["type"] != "org_chart"]
        found: list[int] = []
        steps = 0

        def take(idx):
            nonlocal steps
            if steps >= self.max_steps:
                return
            steps += 1
            new = [i for i in idx if i not in found]
            self._read(new)
            found.extend(new)

        sc = self._bm25.scores(task.question)
        take([i for i in sorted(body, key=lambda i: -sc[i])[: self.k] if sc[i] > 0])
        phrases = _QUOTED.findall(task.question)
        if phrases:
            take([i for i in body if any(p in self.spans[i]["text"] for p in phrases)][:60])
        else:  # a question about the whole history: read the most recent material
            take(sorted(body, key=lambda i: (self.spans[i]["event"], self.spans[i]["id"]), reverse=True)[: self.k * 5])
        refs = {(d, lab) for i in found for _t, d, lab, _n in tf.REF_RE.findall(self.spans[i]["text"])}
        if refs:
            take(
                [i for i in body if any(lab in self.spans[i]["text"] and d in self.spans[i]["text"] for d, lab in refs)]
            )
        org = [i for i, s in enumerate(self.spans) if s["type"] == "org_chart"]
        take(org)
        got_org = [i for i in found if self.spans[i]["type"] == "org_chart"]
        rest = sorted(
            (i for i in found if self.spans[i]["type"] != "org_chart"),
            key=lambda i: (self.spans[i]["event"], self.spans[i]["id"]),
            reverse=True,
        )
        return [self._item(self.spans[i]) for i in got_org + rest]
