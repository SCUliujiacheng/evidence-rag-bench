# Open-source corpus benchmark results

## Protocol

The benchmark uses three hash-locked open-source README documents: FAISS (MIT),
scikit-learn (BSD-3), and LangChain (MIT). The development and held-out test
sets contain eight cases each; six cases per split have evidence labels and two
exercise ambiguity or out-of-corpus behavior. Metrics below were generated on
2026-09-01 with the default 80-word chunker and `k=3`.

## Held-out retrieval results

| Retriever | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | 0.50 | 0.28 | 0.33 |
| TF-IDF (word + bigram) | 0.50 | 0.22 | 0.29 |
| RRF Hybrid | **0.67** | **0.50** | **0.54** |

The hybrid ranker is selected as the demo baseline because it improves all
three evidence-retrieval measures on the untouched held-out split. This is a
small corpus, so the table is evidence for engineering behavior rather than a
claim of general RAG superiority.

## Failure analysis

- `os-test-002`: the hybrid system retrieves a nearby FAISS trade-off chunk
  (`faiss-readme:0007`) but misses the compression-specific source
  (`faiss-readme:0003`). This points to chunk-boundary and lexical-overlap
  sensitivity.
- `os-test-006`: the question uses "provide" while the relevant LangSmith text
  uses "support". Both sparse baselines and RRF miss `langchain-readme:0007`,
  motivating a true embedding retriever in the next milestone.
- `os-test-007` and `os-test-008` intentionally have no gold evidence. They
  are retained for abstention evaluation and are excluded from retrieval-score
  denominators.

## Reproduce

```bash
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever bm25 --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever tfidf --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
```
