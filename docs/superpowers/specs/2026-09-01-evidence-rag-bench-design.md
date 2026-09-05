# Evidence RAG Bench — Design Specification

## 1. Purpose

`evidence-rag-bench` is a research-oriented reference implementation for
evidence-grounded retrieval-augmented generation (RAG). It explores more
than a chat interface: given a fixed, licensed technical corpus, the system
retrieves evidence, returns answer claims with traceable citations, and
explicitly abstains when the corpus does not justify an answer.

The repository explores practical Applied AI / ML Engineering concerns:
data provenance, retrieval baselines, evaluation design, reliability
guardrails, reproducibility, API delivery, and engineering hygiene.

## 2. Goals

- Provide a repeatable pipeline from a versioned corpus manifest to indexed
  chunks, retrieval results, grounded answers, and evaluation artifacts.
- Compare BM25 and dense retrieval under one evaluation protocol; support a
  hybrid retriever and an optional reranker without making either mandatory.
- Produce answer text containing stable citation identifiers such as
  `[doc_id:chunk_id]`; reject citations that do not belong to the retrieved
  context.
- Make a deterministic answer / abstain decision from retrieval confidence and
  citation-validation signals, with the threshold tuned only on a development
  split.
- Publish versioned evaluation cases with answerable, ambiguous, insufficient,
  and unanswerable questions, separate development and test splits, and clear
  metric definitions.
- Expose the system through a small FastAPI service and a minimal browser UI.
- Run locally on Windows with CPython 3.12 and `uv`; use provider APIs only for
  generation and embeddings when configured, so the project remains usable on
  a 6 GB consumer GPU or CPU-only machine.

## 3. Non-goals

- It is not a general web search engine, autonomous research agent, or
  production multi-tenant SaaS.
- It does not claim factual correctness outside its fixed corpus.
- It does not require local LLM serving, Docker, a vector database service, or
  paid APIs for the deterministic retrieval/evaluation path.
- It will not commit copyrighted source PDFs or credentials. Corpus records
  reference public sources and record their licenses and integrity hashes.

## 4. Default Corpus and Provenance

The first release uses a deliberately narrow, public technical corpus: OpenAI
and Hugging Face documentation pages plus a small set of openly licensed or
author-authorized AI systems papers. The initial curated corpus should contain
roughly 15--30 documents, enough to create believable overlap and hard
negative questions while still being inspectable by a reviewer.

`data/corpus/manifest.jsonl` is the source of truth. Each record contains:

- `doc_id`, `title`, `source_url`, `license`, and `retrieved_at`;
- a local normalized text path and SHA-256 checksum;
- a short `scope_note` explaining why the document belongs in the benchmark.

Ingestion refuses a record with a missing URL, license, local file, or checksum
mismatch. Downloading is an explicit command, never an import-time side
effect. Normalized source text is retained only where licensing permits;
otherwise the build emits a documented instruction for the user to fetch it.

## 5. Architecture

```text
corpus manifest + source text
          |
          v
validate provenance -> deterministic chunking -> chunks.jsonl
          |                                      |
          |                                      +--> BM25 index
          |                                      +--> dense embeddings/index (optional)
          v
versioned evaluation cases --------------------> retrieval runner
                                                     |
                                                     v
                                      fused/reranked evidence candidates
                                                     |
                                                     v
  query -> API/UI -> grounded generator -> citation validator -> abstention policy
                                  |                    |                 |
                                  +--------------------+-----------------+
                                                       v
                                    JSON response, traces, metrics, reports
```

The deterministic core is independent of the LLM provider. A provider adapter
receives only the question and the selected evidence chunks, returns structured
claims and citations, and is optional in tests. A rule-based answer formatter
keeps the end-to-end demo runnable without an API key.

## 6. Repository Layout

```text
src/evidence_rag_bench/
  api/            FastAPI app, request/response schemas, dependency wiring
  corpus/         manifest models, validation, normalization, chunking
  retrieval/      BM25, dense adapter, fusion, reranking interfaces
  grounding/      generation adapter, citation parsing/validation, abstention
  evaluation/     case models, metrics, runners, report rendering
  ui/             small static browser client served by the API
  config.py       typed settings and paths
data/
  corpus/         manifest and small permitted fixtures only
  eval/           versioned development and held-out test cases
artifacts/        ignored, reproducible indexes/traces/reports
tests/            unit, integration, API, and evaluation tests
```

The package uses `src/` layout, immutable Pydantic models at boundaries, and
JSONL for auditable datasets and traces. `artifacts/` is generated and ignored
by Git; input manifests, evaluation cases, schemas, and compact test fixtures
are version controlled.

## 7. Retrieval and Grounding Behavior

### Retrieval

The required baseline is BM25 over deterministic chunks. The dense retriever
is an adapter backed by a configurable embedding provider or local
sentence-transformers model. A hybrid retriever fuses normalized BM25 and dense
scores using reciprocal-rank fusion. An optional reranker implements the same
`rank(query, candidates, k)` interface and is disabled by default.

Every result records `doc_id`, `chunk_id`, source URL, chunk text, score, and
retrieval stage. Chunk IDs are stable across a given manifest and chunking
configuration.

### Grounding

The generation prompt instructs the provider to return a JSON object with an
`answer`, `claims`, and `citations` list. Citation IDs must exactly match
retrieved chunk IDs. The validator rejects unknown, duplicate-only, or
malformed citations and records validation failures in the trace. The response
contains both a human-readable answer and machine-readable evidence records.

The initial abstention policy returns `abstain` if either (a) no evidence passes
the score threshold or (b) citation validation fails. Its threshold is selected
on the development split and frozen before the held-out test run. The response
explains that the available corpus lacks sufficient support; it must not invent
an answer.

## 8. Evaluation Protocol

Evaluation cases are JSONL records with `case_id`, `split`, `question`,
`answerability` (`answerable`, `ambiguous`, `insufficient`, or `unanswerable`),
`gold_chunk_ids`, `reference_answer`, and `notes`. The first benchmark target is
50--100 cases, with at least 20 held-out test cases. Test cases are never used
to select retrieval weights or the abstention threshold.

Metrics are emitted per retriever and per split:

- Retrieval: Recall@1/3/5, MRR@5, and nDCG@5 on cases with gold evidence.
- Citation: valid-citation rate, citation precision/recall against gold chunks,
  and unsupported-citation count.
- Abstention: abstention precision, abstention recall, false-answer rate on
  non-answerable cases, and false-abstain rate on answerable cases.
- Delivery: p50/p95 latency and configured provider cost estimate per query.

Answer quality is reported separately from evidence validity. The initial
automated answer measure is exact normalized matching against compact reference
answers, supplemented by a checked-in manual review rubric for a sampled set.
The README must distinguish automated measurements from human judgments.

## 9. Public Interfaces

The service exposes:

- `GET /health` — build/configuration status without exposing secrets.
- `POST /v1/ask` — accepts `{ "question": str, "top_k": int }`; returns an
  answer-or-abstain decision, citations, evidence, latency, and trace ID.
- `POST /v1/evaluations/run` — runs a named split with a selected retriever and
  writes a JSON report under `artifacts/reports/`.
- `GET /v1/evaluations/{report_id}` — returns a saved report.

The browser page sends questions to `/v1/ask`, displays the answer status,
renders clickable source links, and visibly labels abstentions. It contains no
authentication, persistence, or hidden background network calls.

## 10. Error Handling and Responsible Failure

- Missing corpus artifacts, corrupt manifests, and invalid settings fail with
  actionable messages before indexing.
- A missing generation provider key does not break BM25 or evaluation; the API
  uses the deterministic formatter and labels the mode in its response.
- Provider timeouts or malformed output return a controlled abstention with a
  traceable reason, not a fabricated answer.
- Requests enforce a modest question length and `top_k` range.
- Reports record software version, Git revision when available, corpus manifest
  hash, retrieval configuration, random seed, timestamp, and provider mode.

## 11. Quality, Reproducibility, and Delivery

`pyproject.toml` declares Python `>=3.12,<3.13`, runtime dependencies, and
development commands. `uv sync`, `uv run pytest`, `uv run ruff check .`, and
`uv run ruff format --check .` are the canonical local checks. Tests do not call
remote providers. GitHub Actions runs lint and tests on pull requests and main.

The README opens with a result table, architecture diagram, one-command
quickstart, a reproducible benchmark command, documented limitations, and real
failure examples. A small `docs/decision-log.md` explains design tradeoffs such
as choosing BM25 first, abstention policy design, and corpus scope.

## 12. Delivery Milestones

1. Project scaffold, packaging, fixtures, BM25 corpus pipeline, and CI.
2. Evaluation data schema, metrics, benchmark runner, and baseline report.
3. Grounded answer contract, citation validator, abstention calibration, and
   API tests.
4. FastAPI demo UI, reproducibility metadata, and polished README.
5. Dense/hybrid retrieval and optional reranking, compared honestly against
   BM25.
6. Curated public corpus expansion, 50--100 evaluation cases, manual error
   analysis, screenshots/demo video, and public GitHub presentation.

## 13. Acceptance Criteria for the First Public Release

- A clean Windows setup can run the BM25 benchmark and local demo without a
  paid API key.
- The repository contains a manifest, documented corpus provenance, development
  and held-out test cases, and a reproducible report generated from them.
- Every non-abstaining API response has parseable citations that refer only to
  returned evidence.
- The held-out report includes retrieval, citation, abstention, and latency
  metrics with configuration metadata.
- Unit/integration/API tests and CI are green; no secrets or unlicensed corpus
  files are committed.
- The README presents both results and limitations in language a technical
  reader can validate quickly.
