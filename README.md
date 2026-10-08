# Evidence RAG Bench

Retrieval experiments on small English and Chinese document collections.

[简体中文](https://github.com/SCUliujiacheng/evidence-rag-bench-zh) · [Results](docs/benchmark-results.md) · [Design notes](docs/decision-log.md) · [CI](https://github.com/SCUliujiacheng/evidence-rag-bench/actions/workflows/ci.yml)

A passage can mention the right topic without containing the detail a question asks for. One case here asks for a search algorithm used by PaddlePaddle: the README describes automatic parallelization, but never names the algorithm. Lexical retrieval finds that passage and still lets the request through.

This repository compares BM25, TF-IDF, and reciprocal-rank fusion on fixed document snapshots, then checks citations and abstention. The viewer shows the retrieved text; the evaluation runner saves rankings and decisions for each case. The default implementation returns source excerpts and runs locally without a generation model, API key, or GPU.

![Retrieval viewer showing source passages](docs/screenshots/evidence-viewer.png)

## Experiment setup

| Profile | Documents | Chunking | Dev / test cases |
| --- | --- | --- | ---: |
| `en-v1` | 15 documents from FAISS, scikit-learn, and LangChain | 80 words, overlap 20 | 25 / 25 |
| `zh-v1` | 3 Chinese READMEs from Milvus, PaddlePaddle, and FlagEmbedding | 220 visible Unicode units, overlap 40 | 9 / 9 |

Each profile has its own index, evidence labels, and development threshold. Chinese retrieval uses character bigrams alongside Latin words and numbers. Both corpora are stored in the repository and checked by SHA-256; the Chinese sources also point to exact upstream commits. See [corpus attribution](docs/data-attribution.md) for sources, licenses, and the exclusion of the Milvus contributor-avatar section from indexing.

The English fixed test set gives these results at `k=3`, over its 21 answerable cases:

| Retriever | Recall@3 | MRR@3 | nDCG@3 |
| --- | ---: | ---: | ---: |
| BM25 | 0.90 | 0.66 | 0.72 |
| TF-IDF | 0.86 | 0.62 | 0.68 |
| RRF Hybrid | 0.90 | 0.67 | 0.73 |
| Hybrid + MiniLM reranker | 0.86 | 0.74 | 0.77 |

The reranker improves the first relevant rank but loses some top-three coverage. Hybrid also answers 3 of the 4 English cases labeled as non-answerable, despite valid citation IDs. The Chinese Hybrid run retrieves all labeled chunks in the top three, but answers 2 of its 3 non-answerable cases. These failures are included in the [full results](docs/benchmark-results.md).

## Run locally

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run from the repository root. These commands work in PowerShell and Bash:

```text
uv sync --python 3.12
uv run uvicorn evidence_rag_bench.api.app:create_app --factory --port 8000
```

Open `http://127.0.0.1:8000/`. The English repository starts on `en-v1`; the corpus selector also offers `zh-v1`. Try `How does FAISS support cosine similarity?`

In another terminal:

```text
uv run python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --retriever hybrid --k 3
uv run python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --retriever hybrid --k 3 --mode grounded
uv run pytest -v
```

The first evaluation measures retrieval; `--mode grounded` adds citation and abstention checks. Change the profile to `zh-v1` for Chinese, or the retriever to `bm25` or `tfidf` for a baseline. Reports go to `artifacts/reports/` with the configuration, corpus hashes, Git revision, and per-case traces.

The optional English reranking experiment needs an extra dependency and a model download:

```text
uv sync --extra semantic --python 3.12
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --profile en-v1 --split test --retriever semantic-rerank --k 3
```

The MiniLM checkpoint used here was trained on English MS MARCO; `semantic-rerank` rejects `zh-v1`. Details are in the [reranking notes](docs/semantic-reranking.md).

## API

Interactive documentation is at `http://127.0.0.1:8000/docs`. Send this body to `POST /v1/ask`:

```json
{
  "profile": "en-v1",
  "question": "How does FAISS support cosine similarity?",
  "top_k": 3
}
```

The response includes the profile, status, source excerpt, evidence, citations, latency, and trace ID. The existing field name `answer` holds the top-ranked excerpt. `status=answer` means the relevance threshold and citation-ID check passed; it does not establish that the passage supports every part of the question.

## Design notes

- **Keep the baselines inspectable.** All three local retrievers rank the same chunks. Hybrid combines the rankings and retains a separate TF-IDF score for abstention; the RRF rank score is not used as confidence.
- **Calibrate per corpus.** Each profile selects its threshold from its own development cases. Tokenization, chunking, and thresholds do not carry across languages.
- **Keep reranking optional.** MiniLM adds a useful comparison, but it also adds CPU latency and does not verify entailment. Hybrid remains the default.

The [architecture diagram](docs/architecture/evidence-rag-bench-architecture.html) and its [JSON source](docs/architecture/evidence-rag-bench.architecture.json) show how retrieval, threshold selection, and evaluation fit together.

## Limits and open questions

Both test sets were inspected during development. They are fixed regression sets, and the corpora are too small to support broad claims about RAG or Chinese retrieval. The Chinese questions also share terminology with the source documents.

Paraphrases and questions with missing details are useful next cases: they test lexical coverage and evidence sufficiency separately. A semantic-support check would need its own annotations. New model or threshold choices belong on development data; a generalization claim needs a separate, unseen test set.

See the [decision log](docs/decision-log.md), [manual evaluation rubric](docs/evaluation-rubric.md), and [corpus attribution](docs/data-attribution.md). Code is under the [MIT License](LICENSE); source documents retain their upstream licenses.
