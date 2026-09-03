# Open-source corpus benchmark results

## Protocol

The benchmark uses fifteen hash-locked open-source technical documents from the
FAISS (MIT), scikit-learn (BSD-3), and LangChain (MIT) repositories. The
development and held-out test sets contain twenty-five cases each; twenty-one
cases per split have evidence labels and four exercise ambiguity or out-of-corpus
behavior. Metrics below were generated on 2026-09-04 with the default 80-word
chunker and `k=3`.

## Held-out retrieval results

| Retriever | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | **0.90** | 0.66 | 0.72 |
| TF-IDF (word + bigram) | 0.86 | 0.62 | 0.68 |
| RRF Hybrid | **0.90** | 0.67 | 0.73 |
| Hybrid + MiniLM CrossEncoder re-rank | 0.86 | **0.74** | **0.77** |

BM25 and Hybrid tie on held-out Recall@3; Hybrid has a small first-rank and
graded-ranking edge. The demo retains Hybrid because it also exposes a positive
TF-IDF relevance signal for a safer answer/abstain contract. Common English
stop words are removed in both lexical baselines. This is a compact corpus, so
the table is evidence for engineering behavior rather than a claim of general
RAG superiority.

The optional semantic run uses `cross-encoder/ms-marco-MiniLM-L6-v2`
(Apache-2.0), re-ranking the top ten Hybrid candidates on CPU. It improves MRR
and nDCG over Hybrid but not Recall@3, so it cannot recover evidence absent
from the lexical candidate set. It remains opt-in because it has a materially
higher latency.

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
For the Hybrid run, the frozen development threshold was `0.146054`; on the
held-out set it produced citation validity 1.00, citation precision/recall
against gold 0.41/0.43, abstention precision 0.33, abstention recall 0.25,
false-answer rate 0.75, and false-abstain rate 0.10.
Those numbers are deliberately not presented as a success: `os-test-007`
contains plausible LangChain vocabulary but asks for an unsupported
recommendation, so lexical relevance still allows an incorrect answer. This is
the project’s recorded next problem: a valid citation ID is not semantic
support. Any semantic verifier must be calibrated only on development labels
and reported on the held-out split without re-tuning.

The optional CrossEncoder run used only the development JSONL to select a
threshold of `2.463407`. On the held-out set it recorded citation precision/
recall against gold 0.68/0.62, abstention precision 0.67, abstention recall
1.00, false-answer rate 0.00, false-abstain rate 0.10, citation-valid rate
1.00, p50 latency about 400ms, and p95 latency about 690ms. Here
`false-answer rate` means an answer was returned for a non-answerable case; it
does **not** establish that every answer to an answerable case is entailed by
its citation. The latter remains an explicit future semantic-support
evaluation.

## Reproduce

```bash
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever bm25 --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever tfidf --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_test.jsonl --mode grounded --calibration-cases open_source_dev.jsonl
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever semantic-rerank --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --split test --k 3 --retriever semantic-rerank --manifest open_source_manifest.jsonl --cases open_source_test.jsonl --mode grounded --calibration-cases open_source_dev.jsonl
```
