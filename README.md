# Evidence RAG Bench

**English** | [简体中文](https://github.com/SCUliujiacheng/evidence-rag-bench-zh)

An evaluation-first, evidence-grounded RAG reference implementation. It validates corpus provenance, compares BM25/TF-IDF/RRF-hybrid retrieval, checks citation IDs against returned evidence, and abstains when evidence is insufficient.

## Why this is useful

- **Reproducible evidence:** 15 license-attributed source documents are hash-locked before deterministic chunking.
- **Measured retrieval:** a versioned 25-case regression protocol (21 evidence-labelled cases) compares lexical, hybrid, and optional local semantic ranking.
- **Grounded behavior:** every citation is checked against the evidence returned to the caller; low-confidence requests return a structured abstention.
- **Inspectable delivery:** reports retain the configuration, manifest hash, Git revision, metrics, latency, and per-case traces.

On the current protocol-v0.1 test snapshot at `k=3`, RRF hybrid reaches **0.90 Recall@3** and **0.73 nDCG@3**. The optional local CrossEncoder reaches **0.74 MRR@3** and **0.77 nDCG@3**; it improves ranking quality but is deliberately documented as a relevance model, not an entailment verifier. See the [full protocol and results](docs/benchmark-results.md).

## Architecture

[Open the interactive architecture](docs/architecture/evidence-rag-bench-architecture.html) for the request, retrieval, grounding, and evaluation paths. The checked-in specification lives beside it in [JSON](docs/architecture/evidence-rag-bench.architecture.json).

```text
Evidence Viewer -> FastAPI -> Grounding Guardrail -> Local Retrieval -> Versioned Corpus
                                              \-> Evaluation Runner -> Provenance-rich JSON reports
                                     (optional) \-> local CrossEncoder re-ranker
```

## Run locally

```bash
uv sync --python 3.12
uv run pytest -v
uv run python -m evidence_rag_bench.evaluation.runner --split dev --k 3 --retriever hybrid --manifest open_source_manifest.jsonl --cases open_source_dev.jsonl
uv run uvicorn evidence_rag_bench.api.app:create_app --factory --port 8000
```

Open `http://127.0.0.1:8000/`. The demo provides either evidence-bound citations or an explicit abstention; no API key or GPU is required.

To reproduce the optional semantic re-ranking experiment, install its extra and select the retriever explicitly:

```bash
uv sync --extra semantic --python 3.12
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner --split test --retriever semantic-rerank --k 3 --manifest open_source_manifest.jsonl --cases open_source_test.jsonl
```

## API example

```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "content-type: application/json" \
  -d "{\"question\": \"How can FAISS implement cosine similarity?\", \"top_k\": 3}"
```

Responses expose the decision (`answer` or `abstain`), a deterministic answer string, confidence, and chunk-level citations. Citation IDs in an `answer` response always refer to the returned evidence; an abstention never invents a citation.

## What is evaluated

Versioned JSONL development and test cases measure Recall@k, MRR@k and nDCG@k over gold evidence IDs. The loader rejects duplicate case IDs, duplicate normalized questions, cross-split question reuse, and labels that conflict with answerability. Abstention thresholds are selected from development cases only. Generated reports record corpus-manifest hash, Git revision, time and configuration under ignored `artifacts/reports/`.

The test results were inspected during early development and influenced the demo retriever choice; the split also grew from 8 to 25 cases as the corpus expanded. The current v0.1 test data is therefore a reproducible regression snapshot, not evidence of blind generalization. From v0.1 onward, new model and parameter choices are made on development data first; expanding the test split requires a new protocol version that preserves the old snapshot. A new generalization claim requires a newly sealed, unseen test set. The historical decision is recorded in the [decision log](docs/decision-log.md).

The default demo uses fifteen hash-locked, license-attributed open-source technical documents (FAISS, scikit-learn, and LangChain) and a 25-case versioned test snapshot. The local retrievers trade off coverage and ranking quality; see [benchmark results](docs/benchmark-results.md) for the protocol, exact results, failures, and reproduction commands. This is a compact benchmark, not a general performance claim.

See the [design](docs/superpowers/specs/2026-09-01-evidence-rag-bench-design.md), [implementation plan](docs/superpowers/plans/2026-09-01-evidence-rag-bench-mvp.md), [data attribution](docs/data-attribution.md), [benchmark results](docs/benchmark-results.md), [optional semantic re-ranking protocol](docs/semantic-reranking.md), [decision log](docs/decision-log.md), and [manual evaluation rubric](docs/evaluation-rubric.md).

## License

The project code is released under the [MIT License](LICENSE). Corpus documents
retain their upstream licenses; see [data attribution](docs/data-attribution.md).
