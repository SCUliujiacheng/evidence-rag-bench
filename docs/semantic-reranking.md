# Optional local semantic re-ranking

The default path stays deterministic and light on dependencies. This optional
stage re-ranks a fixed lexical candidate set with a local CrossEncoder. It does
not write an answer, change corpus content, or loosen citation validation.

## Model choice

The initial experiment target is
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2): a 22.7M-parameter,
Apache-2.0 licensed passage-ranking CrossEncoder. The model card documents its
MS MARCO training data and `CrossEncoder` inference interface. It is a
semantic relevance re-ranker. It does not verify entailment or establish that
an answer is supported.

This checkpoint is used only with `en-v1`. The runner rejects `zh-v1` because
an English MS MARCO model is not a defensible Chinese baseline. The Chinese
profile currently supports BM25, TF-IDF, and RRF Hybrid.

## Run locally

```bash
uv sync --extra semantic
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner \
  --profile en-v1 --split test --retriever semantic-rerank --k 3
```

`SentenceTransformersCrossEncoder` loads the named model lazily, so CI and the
baseline demo do not download model weights. Reports record model identity and
candidate depth alongside the existing lexical configuration. A run may use
development cases to choose a threshold, but it must not change the fixed test
labels or tune on them.

## Acceptance gate

The initial CPU experiment (15-document corpus, 25 fixed test cases,
candidate depth 10) improved Hybrid MRR@3 from 0.667 to 0.738 and nDCG@3 from
0.728 to 0.769, while Recall@3 fell from 0.905 to 0.857. With a
development-selected threshold, false-answer rate fell from 0.75 to 0.00 and
abstention recall rose from 0.25 to 1.00; p50 latency was about 410ms in the
current CPU run. Full
measurements and caveats are in [benchmark results](benchmark-results.md).

Hybrid remains the default because BM25 still wins retrieval coverage, the
CrossEncoder adds CPU latency, and relevance is not answer entailment.
