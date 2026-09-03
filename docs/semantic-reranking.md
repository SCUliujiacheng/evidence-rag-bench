# Optional local semantic re-ranking

The default benchmark remains deterministic and dependency-light. This optional
stage re-ranks a fixed lexical candidate set with a local CrossEncoder; it does
not invent an answer, alter corpus content, or relax citation validation.

## Model choice

The initial experiment target is
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2): a 22.7M-parameter,
Apache-2.0 licensed passage-ranking CrossEncoder. The model card documents its
MS MARCO training data and `CrossEncoder` inference interface. It is a
semantic relevance re-ranker, **not** a factual-entailment verifier; the
project must not claim that it proves an answer is supported.

## Run locally

```bash
uv sync --extra semantic
```

`SentenceTransformersCrossEncoder` loads the named model lazily, so CI and the
baseline demo do not download model weights. Reports record model identity and
candidate depth alongside the existing lexical configuration. A run may use
development cases to choose a threshold, but it must not change the frozen
held-out labels or tune on them.

## Acceptance gate

The initial CPU experiment (15-document corpus, 25 frozen held-out cases,
candidate depth 10) improved Hybrid MRR@3 from 0.667 to 0.738 and nDCG@3 from
0.728 to 0.769, while Recall@3 fell from 0.905 to 0.857. With a
development-selected threshold, false-answer rate fell from 0.75 to 0.00 and
abstention recall rose from 0.25 to 1.00; p50 latency rose to 394ms. Full
measurements and caveats are in [benchmark results](benchmark-results.md).

Hybrid remains the default deterministic retriever because BM25 still wins
retrieval coverage, the CrossEncoder adds CPU latency, and a relevance model is
not yet an explicit answer-entailment verifier.
