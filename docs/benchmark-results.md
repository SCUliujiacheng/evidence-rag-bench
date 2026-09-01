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
| BM25 | 0.83 | 0.58 | 0.65 |
| TF-IDF (word + bigram) | 0.83 | 0.58 | 0.65 |
| RRF Hybrid | **0.83** | **0.67** | **0.71** |

The hybrid ranker is selected as the demo baseline because it improves ranking
quality over the single retrievers on the untouched held-out split. Common
English stop words are removed in both lexical baselines; this change was made
before this reported rerun and applies identically to every retriever. This is
a small corpus, so the table is evidence for engineering behavior rather than a
claim of general RAG superiority.

## Failure analysis

- `os-test-006`: the question uses "provide" while the relevant LangSmith text
  uses "support". Both sparse baselines and RRF miss `langchain-readme:0007`,
  motivating a true embedding retriever in the next milestone.
- `os-test-007` and `os-test-008` intentionally have no gold evidence. They
  are retained for abstention evaluation and are excluded from retrieval-score
  denominators.

## End-to-end abstention check

The API calibrates its relevance threshold only from the development split
(`0.184` in this run). When that frozen threshold is applied to the held-out
set, citation validity is 1.00 but abstention precision/recall are 0.25/0.50;
the false-answer and false-abstain rates are both 0.50. In particular,
`os-test-007` contains plausible LangChain vocabulary but asks for an
unsupported recommendation, while several concise FAISS questions fall below
the lexical relevance threshold. This is an intentional recorded limitation:
ranking confidence alone is not a semantic-support verifier. The next
iteration needs a separately evaluated entailment or structured LLM judge,
without allowing that judge to inspect held-out labels during calibration.

## Reproduce

```bash
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever bm25 --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever tfidf --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
```
