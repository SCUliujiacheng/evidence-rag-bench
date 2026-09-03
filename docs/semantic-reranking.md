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
baseline demo do not download model weights. A future benchmark run must report
the exact model revision, device, candidate depth, development-selected
threshold, held-out metrics, and latency alongside the existing lexical
baselines. It may use development cases to choose a threshold, but it must not
change the frozen held-out labels or tune on them.

## Acceptance gate

Before this stage becomes a demo default, it needs a committed held-out report
that improves a predeclared metric without degrading answer/abstain safety, a
failure analysis, and an explicit evaluation of semantic support. Until then,
Hybrid remains the default deterministic retriever.
