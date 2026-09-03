# Evidence RAG Bench

An evaluation-first, evidence-grounded RAG reference implementation. It validates corpus provenance, compares BM25/TF-IDF/RRF-hybrid retrieval, checks citation IDs against returned evidence, and abstains when evidence is insufficient.

## Run locally

```bash
uv sync --python 3.12
uv run pytest -v
uv run python -m evidence_rag_bench.evaluation.runner --split dev --k 3
uv run uvicorn evidence_rag_bench.api.app:create_app --factory --port 8000
```

Open `http://127.0.0.1:8000/`. The demo provides either evidence-bound citations or an explicit abstention; no API key or GPU is required.

## What is evaluated

Versioned JSONL development and held-out test cases measure Recall@k, MRR@k and nDCG@k over gold evidence IDs. The loader rejects duplicate case IDs, duplicate normalized questions, cross-split question reuse, and labels that conflict with answerability; this makes the development/test boundary executable rather than a convention. Generated reports record corpus-manifest hash, Git revision, time and configuration under ignored `artifacts/reports/`.

The default demo uses six hash-locked, license-attributed open-source technical documents (FAISS, scikit-learn, and LangChain) and a 12-case held-out retrieval split. The three local retrievers trade off coverage and ranking quality; see [benchmark results](docs/benchmark-results.md) for the protocol, exact results, failures, and reproduction commands. This is a small benchmark, not a general performance claim.

See the [design](docs/superpowers/specs/2026-09-01-evidence-rag-bench-design.md), [implementation plan](docs/superpowers/plans/2026-09-01-evidence-rag-bench-mvp.md), [data attribution](docs/data-attribution.md), [benchmark results](docs/benchmark-results.md), [decision log](docs/decision-log.md), and [manual evaluation rubric](docs/evaluation-rubric.md).
