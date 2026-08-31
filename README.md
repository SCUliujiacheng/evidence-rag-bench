# Evidence RAG Bench

An evaluation-first, evidence-grounded RAG reference implementation. It validates corpus provenance, retrieves source chunks with a local BM25 baseline, checks citation IDs against returned evidence, and abstains when evidence is insufficient.

## Run locally

```bash
uv sync --python 3.12
uv run pytest -v
uv run python -m evidence_rag_bench.evaluation.runner --split dev --k 3
uv run uvicorn evidence_rag_bench.api.app:create_app --factory --port 8000
```

Open `http://127.0.0.1:8000/`. The demo provides either evidence-bound citations or an explicit abstention; no API key or GPU is required.

## What is evaluated

Versioned JSONL development and held-out test cases measure Recall@k, MRR@k and nDCG@k over gold evidence IDs. Generated reports record corpus-manifest hash, Git revision, time and configuration under ignored `artifacts/reports/`.

The committed CC-BY fixture corpus is intentionally tiny, so its perfect fixture result is not a real-world performance claim. The next milestone expands to licensed public technical sources, 50--100 reviewed questions, dense/hybrid retrieval and manual error analysis.

See the [design](docs/superpowers/specs/2026-09-01-evidence-rag-bench-design.md) and [implementation plan](docs/superpowers/plans/2026-09-01-evidence-rag-bench-mvp.md).
