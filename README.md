<div align="center">

# Evidence RAG Bench

**A small RAG benchmark for checking what an answer actually retrieved.**

<p>
  <a href="https://github.com/SCUliujiacheng/evidence-rag-bench/actions/workflows/ci.yml"><img src="https://github.com/SCUliujiacheng/evidence-rag-bench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0B7285.svg" alt="MIT License"></a>
</p>

[Why I built this](#why-i-built-this) · [What I wanted to check](#what-i-wanted-to-check) · [Architecture](#architecture) · [Run locally](#run-locally)

**English** · [简体中文](https://github.com/SCUliujiacheng/evidence-rag-bench-zh)

</div>

## Why I built this

The question behind this repo is simple: when a RAG answer looks plausible, what did it actually retrieve? A fluent sentence is not much help if its path back to the source is blurry.

This is a small, inspectable way to test that boundary. The inputs are versioned, retrieval is measured, citation IDs are checked against the evidence returned to the reader, and a weak match becomes an explicit abstention.

<p align="center">
  <img src="docs/screenshots/evidence-viewer.png" alt="Evidence RAG Bench showing an evidence-grounded answer with inspectable source chunks" width="100%">
</p>

<p align="center"><sub>The useful part is not the answer alone; it is being able to inspect the passage behind it.</sub></p>

## What I wanted to check

| Question | What the repository does about it |
| --- | --- |
| Can someone rerun the same retrieval? | The 15 license-attributed source documents are hash-locked before deterministic chunking; the [corpus manifest](data/corpus/open_source_manifest.jsonl) and [validation code](src/evidence_rag_bench/corpus/manifest.py) are both here. |
| Does one retriever merely sound better? | A versioned protocol with 25 development and 25 test cases (21 evidence-labelled cases per split) compares lexical, hybrid, and optional local semantic ranking; see the [benchmark results](docs/benchmark-results.md). |
| Is a citation actually from this answer's evidence? | Each citation must belong to the returned evidence. Low-confidence requests take the structured abstention path in the [grounding service](src/evidence_rag_bench/grounding/service.py). |
| What happened on a particular run? | The [evaluation runner](src/evidence_rag_bench/evaluation/runner.py) keeps configuration, manifest hash, Git revision, metrics, latency, and per-case traces. |

> **Current protocol-v0.1 snapshot at `k=3`:** RRF hybrid reaches **0.90 Recall@3** and **0.73 nDCG@3**. The optional local CrossEncoder reaches **0.74 MRR@3** and **0.77 nDCG@3**. It improves ranking quality, but it is a relevance model—not an entailment verifier. The [full protocol and results](docs/benchmark-results.md) include exact settings, failures, and limitations.

## Architecture

[Open the interactive architecture](docs/architecture/evidence-rag-bench-architecture.html) for the request, retrieval, grounding, and evaluation paths. The [JSON source](docs/architecture/evidence-rag-bench.architecture.json) sits beside it so the diagram is reviewable too.

```mermaid
flowchart LR
    UI[Evidence Viewer] --> API[FastAPI]
    API --> Guard[Grounding Guardrail]
    Guard --> Retrieval[Local Retrieval]
    Retrieval --> Corpus[(Versioned Corpus)]
    Retrieval -. optional .-> Reranker[Local CrossEncoder]
    Retrieval --> Eval[Evaluation Runner]
    Eval --> Reports[Provenance-rich Reports]
```

## Run locally

The default hybrid path needs Python 3.12 and [`uv`](https://docs.astral.sh/uv/), but no API key or GPU:

**PowerShell**

```powershell
uv sync --python 3.12
uv run pytest -v
uv run python -m evidence_rag_bench.evaluation.runner `
  --split dev `
  --k 3 `
  --retriever hybrid `
  --manifest open_source_manifest.jsonl `
  --cases open_source_dev.jsonl
uv run uvicorn evidence_rag_bench.api.app:create_app `
  --factory `
  --port 8000
```

<details>
<summary>Bash / Git Bash equivalent</summary>

```bash
uv sync --python 3.12
uv run pytest -v
uv run python -m evidence_rag_bench.evaluation.runner \
  --split dev \
  --k 3 \
  --retriever hybrid \
  --manifest open_source_manifest.jsonl \
  --cases open_source_dev.jsonl
uv run uvicorn evidence_rag_bench.api.app:create_app \
  --factory \
  --port 8000
```

</details>

Open `http://127.0.0.1:8000/`. The viewer returns either evidence-bound citations or an explicit abstention.

<details>
<summary>An existing Windows checkout reports a corpus checksum mismatch</summary>

The byte-preservation rule applies automatically to fresh checkouts. If the repository was checked out before that rule existed, first make sure `git status --short -- data/corpus` prints nothing, then refresh only the tracked corpus files once:

```text
git rm -r --cached -- data/corpus
git restore --source=HEAD --staged --worktree -- data/corpus
```

</details>

To reproduce the optional semantic re-ranking experiment:

```powershell
uv sync --extra semantic --python 3.12
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner `
  --split test `
  --retriever semantic-rerank `
  --k 3 `
  --manifest open_source_manifest.jsonl `
  --cases open_source_test.jsonl
```

<details>
<summary>Bash / Git Bash equivalent</summary>

```bash
uv sync --extra semantic --python 3.12
uv run --extra semantic python -m evidence_rag_bench.evaluation.runner \
  --split test \
  --retriever semantic-rerank \
  --k 3 \
  --manifest open_source_manifest.jsonl \
  --cases open_source_test.jsonl
```

</details>

## API example

```powershell
$body = @{
  question = "How can FAISS implement cosine similarity?"
  top_k = 3
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/v1/ask" `
  -ContentType "application/json" `
  -Body $body
```

<details>
<summary>Bash / Git Bash equivalent</summary>

```bash
curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "content-type: application/json" \
  -d '{"question":"How can FAISS implement cosine similarity?","top_k":3}'
```

</details>

Responses include the decision (`answer` or `abstain`), a deterministic answer string, confidence, and chunk-level citations. In an `answer` response, every citation ID refers to returned evidence; an abstention has no invented citation.

## What is evaluated

Versioned JSONL development and test cases measure Recall@k, MRR@k, and nDCG@k over gold evidence IDs. The loader rejects duplicate case IDs, duplicate normalized questions, cross-split question reuse, and labels that conflict with answerability. Abstention thresholds are selected from development cases only. Generated reports record the corpus-manifest hash, Git revision, timestamp, and configuration under ignored `artifacts/reports/`.

The test results were inspected during early development and influenced the demo retriever choice; the split also grew from 8 to 25 cases as the corpus expanded. The current v0.1 test data is therefore a reproducible regression snapshot, not evidence of blind generalization. From v0.1 onward, new model and parameter choices are made on development data first. Expanding the test split requires a new protocol version that preserves the old snapshot, and a new generalization claim requires a newly sealed, unseen test set. The historical choice is recorded in the [decision log](docs/decision-log.md).

The default demo uses fifteen hash-locked, license-attributed technical documents from FAISS, scikit-learn, and LangChain. This is a compact benchmark, not a general performance claim.

## Notes and results

For the details behind the demo:

- [Data attribution](docs/data-attribution.md)
- [Benchmark results](docs/benchmark-results.md)
- [Optional semantic re-ranking protocol](docs/semantic-reranking.md)
- [Decision log](docs/decision-log.md)
- [Manual evaluation rubric](docs/evaluation-rubric.md)

## License

The project code is released under the [MIT License](LICENSE). Corpus documents retain their upstream licenses; see [data attribution](docs/data-attribution.md).
