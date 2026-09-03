# Open-source corpus benchmark results

## Protocol

The benchmark uses six hash-locked open-source technical documents from the
FAISS (MIT), scikit-learn (BSD-3), and LangChain (MIT) repositories. The
development and held-out test sets contain twelve cases each; nine cases per
split have evidence labels and three exercise ambiguity or out-of-corpus
behavior. Metrics below were generated on 2026-09-04 with the default 80-word
chunker and `k=3`.

## Held-out retrieval results

| Retriever | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | **0.89** | 0.54 | 0.63 |
| TF-IDF (word + bigram) | 0.78 | **0.61** | **0.65** |
| RRF Hybrid | 0.78 | 0.56 | 0.61 |

No retriever dominates every metric after the corpus expansion. BM25 has the
best held-out coverage; TF-IDF has the best first-rank and graded ranking
metrics. The demo retains Hybrid because it exposes a positive TF-IDF relevance
signal for a safer answer/abstain contract. Common English stop words are
removed in both lexical baselines. This is a small corpus, so the table is
evidence for engineering behavior rather than a claim of general RAG
superiority.

## Failure analysis

- `os-test-006`: the question uses "provide" while the relevant LangSmith text
  uses "support". Both sparse baselines and RRF miss `langchain-readme:0007`,
  motivating a true embedding retriever in the next milestone.
- `os-test-007` and `os-test-008` intentionally have no gold evidence. They
  are retained for abstention evaluation and are excluded from retrieval-score
  denominators.

## End-to-end abstention check

The end-to-end runner now selects its relevance threshold only from the named
development JSONL and writes both the threshold and its source into the report.
For the Hybrid run, the frozen development threshold was `0.120188`; on the
held-out set it produced citation validity 1.00, abstention precision 0.50,
abstention recall 0.33, false-answer rate 0.67, and false-abstain rate 0.11.
Those numbers are deliberately not presented as a success: `os-test-007`
contains plausible LangChain vocabulary but asks for an unsupported
recommendation, so lexical relevance still allows an incorrect answer. This is
the project’s recorded next problem: a valid citation ID is not semantic
support. Any semantic verifier must be calibrated only on development labels
and reported on the held-out split without re-tuning.

## Reproduce

```bash
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever bm25 --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever tfidf --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl --mode grounded --calibration-cases open_source_dev.jsonl
```
