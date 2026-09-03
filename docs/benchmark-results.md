# Open-source corpus benchmark results

## Protocol

The benchmark uses ten hash-locked open-source technical documents from the
FAISS (MIT), scikit-learn (BSD-3), and LangChain (MIT) repositories. The
development and held-out test sets contain sixteen cases each; thirteen cases
per split have evidence labels and three exercise ambiguity or out-of-corpus
behavior. Metrics below were generated on 2026-09-04 with the default 80-word
chunker and `k=3`.

## Held-out retrieval results

| Retriever | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | **0.92** | **0.69** | **0.75** |
| TF-IDF (word + bigram) | 0.77 | 0.60 | 0.65 |
| RRF Hybrid | 0.77 | 0.60 | 0.65 |
| Hybrid + MiniLM CrossEncoder re-rank | 0.77 | 0.65 | 0.68 |

BM25 leads every held-out retrieval metric after this corpus expansion. The
demo retains Hybrid because it exposes a positive TF-IDF relevance signal for a
safer answer/abstain contract; it is an engineering default, not a claim that
RRF is the strongest baseline. Common English stop words are removed in both
lexical baselines. This is a small corpus, so the table is evidence for
engineering behavior rather than a claim of general RAG superiority.

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
For the Hybrid run, the frozen development threshold was `0.128796`; on the
held-out set it produced citation validity 1.00, abstention precision 0.33,
abstention recall 0.33, false-answer rate 0.67, and false-abstain rate 0.15.
Those numbers are deliberately not presented as a success: `os-test-007`
contains plausible LangChain vocabulary but asks for an unsupported
recommendation, so lexical relevance still allows an incorrect answer. This is
the project’s recorded next problem: a valid citation ID is not semantic
support. Any semantic verifier must be calibrated only on development labels
and reported on the held-out split without re-tuning.

The optional CrossEncoder run used only the development JSONL to select a
threshold of `2.463407`. On the held-out set it recorded abstention precision
0.60, abstention recall 1.00, false-answer rate 0.00, false-abstain rate 0.15,
citation-valid rate 1.00, p50 latency 377ms, and p95 latency 468ms. Here
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
