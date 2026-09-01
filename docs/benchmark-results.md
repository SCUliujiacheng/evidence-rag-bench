# Open-source corpus benchmark results

## Protocol

The benchmark uses six hash-locked open-source technical documents from the
FAISS (MIT), scikit-learn (BSD-3), and LangChain (MIT) repositories. The
development and held-out test sets contain twelve cases each; nine cases per
split have evidence labels and three exercise ambiguity or out-of-corpus
behavior. Metrics below were generated on 2026-09-01 with the default 80-word
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
