# English and Chinese corpus benchmark results

## `en-v1` protocol

This run uses fifteen hash-locked open-source technical documents from the
FAISS (MIT), scikit-learn (BSD-3), and LangChain (MIT) repositories. The
development and fixed test sets contain twenty-five cases each; twenty-one
cases per split have evidence labels and four exercise ambiguity or out-of-corpus
behavior. The test results were viewed during development, so this split is a
regression snapshot rather than a blind evaluation. Metrics below were generated
on 2026-09-06 with the default 80-word chunker and `k=3`. Per-query Recall
is the fraction of all labeled qrels present in the top three, averaged over
answerable cases; it is not a hit-rate alias.

## Fixed test snapshot

| Retriever | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | **0.90** | 0.66 | 0.72 |
| TF-IDF (word + bigram) | 0.86 | 0.62 | 0.68 |
| RRF Hybrid | **0.90** | 0.67 | 0.73 |
| Hybrid + MiniLM CrossEncoder re-rank | 0.86 | **0.74** | **0.77** |

BM25 and Hybrid tie on test Recall@3; Hybrid is slightly better at the
first rank and on binary-relevance nDCG. The demo uses Hybrid because its positive
TF-IDF relevance signal also gives the answer/abstain rule something to work
with. Both lexical baselines remove common English stop words. With a corpus
this small, the table describes this implementation, not RAG in general.

The optional semantic run uses `cross-encoder/ms-marco-MiniLM-L6-v2`
(Apache-2.0), re-ranking the top ten Hybrid candidates on CPU. It improves MRR
and nDCG over Hybrid but not Recall@3, so it cannot recover evidence absent
from the lexical candidate set. It remains opt-in because it has a materially
higher latency.

## Failure analysis

- `os-test-006`: the question uses "provide" while the relevant LangSmith text
  uses "support". Both sparse baselines and RRF miss `langchain-readme:0007`,
  suggesting an embedding retriever as the next experiment.
- `os-test-007` and `os-test-008` intentionally have no gold evidence. They
  are retained for abstention evaluation and are excluded from retrieval-score
  denominators.

## End-to-end abstention check

The end-to-end runner now selects its relevance threshold only from the named
development JSONL and writes both the threshold and its source into the report.
For the Hybrid run, the frozen development threshold was `0.146054`; on the
fixed test set it produced citation validity 1.00, citation precision/recall
against gold 0.41/0.43, abstention precision 0.33, abstention recall 0.25,
false-answer rate 0.75, and false-abstain rate 0.10.
`os-test-007` keeps this from being a success story: it contains plausible
LangChain vocabulary but asks for an unsupported recommendation, so lexical
relevance still permits an incorrect answer. The next problem is plain: a
valid citation ID is not semantic support. Any semantic verifier must be
calibrated only on development labels and reported on the fixed test split
without re-tuning. Calling a result held out would require a new, sealed set.

The optional CrossEncoder run used only the development JSONL to select a
threshold of `2.463407`. On the fixed test set it recorded citation precision/
recall against gold 0.68/0.62, abstention precision 0.67, abstention recall
1.00, false-answer rate 0.00, false-abstain rate 0.10, citation-valid rate
1.00, p50 latency about 410ms, and p95 latency about 480ms in the current CPU run. Here
`false-answer rate` means an answer was returned for a non-answerable case; it
does **not** establish that every answer to an answerable case is entailed by
its citation. The latter remains an explicit future semantic-support
evaluation.

## `zh-v1` snapshot

`zh-v1` uses three Chinese README files from Milvus, PaddlePaddle, and
FlagEmbedding. Every source URL points to a full Git commit and every local
file is checked against the manifest hash. The complete Milvus file remains in
the repository, but its manifest stops indexing at `### All contributors` so
110 avatar-heavy chunks do not dominate the corpus. Reports record both that
rule and the SHA-256 of the bytes actually indexed. The Unicode chunker keeps
source punctuation and spacing, with a 220-unit window and 40-unit overlap.
Retrieval uses NFKC-normalized CJK character bigrams alongside lowercase Latin
words and numbers; it does not need a Chinese segmentation package.

The development and fixed test JSONL files each contain nine cases: six with
gold evidence plus one ambiguous, one insufficient, and one out-of-corpus
question. These measurements were generated on 2026-09-06 at `k=3`.

| Retriever | Dev Recall@3 | Dev MRR@3 | Test Recall@3 | Test MRR@3 | Test nDCG@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 + mixed tokenizer | **1.00** | **1.00** | 0.92 | **1.00** | 0.94 |
| TF-IDF + mixed tokenizer | **1.00** | 0.92 | **1.00** | **1.00** | **0.99** |
| RRF Hybrid | **1.00** | **1.00** | **1.00** | **1.00** | **0.99** |

Perfect Hybrid Recall@3 and MRR@3 need context. The corpus has only
three documents, the six answerable questions use terminology present in those
documents, and I viewed the cases while checking the new profile. It is a
useful regression check for the tokenizer and chunk IDs, not evidence that the
retriever generalizes to Chinese RAG workloads.

The threshold selected from `zh_v1_dev.jsonl` is `0.175402`. On the fixed test
set, the grounded Hybrid run records citation validity 1.00, citation
precision/recall against gold 0.71/0.71, abstention precision 0.50, abstention
recall 0.33, false-answer rate 0.67, and false-abstain rate 0.17. The ambiguous
RRF-vs-Weighted-Scoring question and the underspecified PaddlePaddle strategy
question still retrieve relevant passages strongly enough to answer. That is
the current failure: lexical relevance can find the right topic without
showing that the requested conclusion is present. In this small check the
threshold is better at rejecting clearly out-of-domain questions than at
deciding semantic sufficiency, so the UI presents returned text as evidence to
verify rather than a proven answer.

The English-only MiniLM checkpoint is not run on `zh-v1`; the CLI rejects that
combination instead of reporting a misleading multilingual number.

## Reproduce

```bash
uv run python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --k 3 --retriever bm25
uv run python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --k 3 --retriever tfidf
uv run python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --k 3 --retriever hybrid
uv run python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --k 3 --retriever hybrid --mode grounded
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --k 3 --retriever semantic-rerank
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --k 3 --retriever semantic-rerank --mode grounded

uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever bm25
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever tfidf
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever hybrid
uv run python -m evidence_rag_bench.evaluation.runner --profile zh-v1 --split test --k 3 --retriever hybrid --mode grounded
```
