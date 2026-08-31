# Evidence RAG Bench MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, reproducible evidence-grounded RAG benchmark that retrieves auditable source chunks, validates citations, abstains safely, and exposes a small FastAPI demo.

**Architecture:** A deterministic corpus pipeline creates stable chunks from a provenance manifest. BM25 retrieval feeds an optional provider-independent grounded-answer formatter; citation validation and a calibrated abstention policy protect the response. Evaluation runs against versioned JSONL cases and writes metadata-rich reports; FastAPI and a static UI consume the same service layer.

**Tech Stack:** CPython 3.12, uv, FastAPI, Pydantic v2, rank-bm25, scikit-learn, pytest, Ruff, httpx, GitHub Actions, vanilla HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-01-evidence-rag-bench-design.md`

## Global Constraints

- Python version is `>=3.12,<3.13`; run every command with `uv run` after `uv sync`.
- The required retrieval baseline is deterministic BM25; remote LLM and embedding calls are optional and never made in tests.
- Keep inputs and evaluation cases as version-controlled JSONL; generated indexes, traces, and reports belong under ignored `artifacts/`.
- Every corpus record needs a public source URL, license string, local text path, retrieval timestamp, and SHA-256 hash.
- Do not commit credentials, copyrighted source PDFs, generated provider output, or dependency virtual environments.
- Non-abstaining answers may cite only returned evidence chunk identifiers of the form `[doc_id:chunk_id]`.
- Tune the abstention threshold on the `dev` split only; preserve the `test` split as held out.
- Tests run without network access or provider keys; CI runs lint, formatting checks, and tests.
- Use conventional commits and keep each task independently testable.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | package metadata, dependencies, Ruff and pytest settings |
| `.gitignore` | ignores artifacts, virtual environments, and local secrets |
| `src/evidence_rag_bench/config.py` | typed settings and repository path resolution |
| `src/evidence_rag_bench/models.py` | shared Pydantic request, chunk, citation, and answer models |
| `src/evidence_rag_bench/corpus/manifest.py` | manifest validation and file-integrity verification |
| `src/evidence_rag_bench/corpus/chunking.py` | deterministic text chunking and stable chunk IDs |
| `src/evidence_rag_bench/retrieval/bm25.py` | BM25 index construction and ranked retrieval |
| `src/evidence_rag_bench/grounding/service.py` | answer formatter, citation validation, and abstention decision |
| `src/evidence_rag_bench/evaluation/*.py` | JSONL cases, retrieval/grounding metrics, runner, report writing |
| `src/evidence_rag_bench/api/app.py` | FastAPI routes and dependency assembly |
| `src/evidence_rag_bench/ui/*` | static one-page evidence viewer |
| `data/corpus/*` | permitted fixture sources and auditable manifest |
| `data/eval/*.jsonl` | development and held-out benchmark cases |
| `tests/*` | deterministic unit, integration, and API coverage |
| `.github/workflows/ci.yml` | lint, formatting, and test automation |
| `README.md`, `docs/decision-log.md` | recruiter-first explanation, reproducibility, and tradeoffs |

### Task 1: Bootstrap the reproducible Python package

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/evidence_rag_bench/__init__.py`
- Create: `src/evidence_rag_bench/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings(project_root: Path, artifacts_dir: Path, corpus_dir: Path, eval_dir: Path)` and `get_settings(project_root: Path | None = None) -> Settings`.
- Consumes: no earlier project interfaces.

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path

from evidence_rag_bench.config import get_settings


def test_settings_resolve_project_paths(tmp_path: Path) -> None:
    settings = get_settings(tmp_path)
    assert settings.project_root == tmp_path.resolve()
    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.corpus_dir == tmp_path / "data" / "corpus"
```

- [ ] **Step 2: Verify the test fails before the package exists**

Run: `uv run pytest tests/test_config.py -v`

Expected: `ModuleNotFoundError: No module named 'evidence_rag_bench'`.

- [ ] **Step 3: Add packaging, ignore rules, and minimal settings implementation**

```toml
[project]
name = "evidence-rag-bench"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["fastapi>=0.115", "pydantic>=2.9", "rank-bm25>=0.2.2", "uvicorn>=0.30"]

[dependency-groups]
dev = ["httpx>=0.27", "pytest>=8.3", "ruff>=0.8"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
@dataclass(frozen=True)
class Settings:
    project_root: Path
    artifacts_dir: Path
    corpus_dir: Path
    eval_dir: Path


def get_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or Path.cwd()).resolve()
    return Settings(root, root / "artifacts", root / "data" / "corpus", root / "data" / "eval")
```

Add `.gitignore` entries for `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.env`, and `artifacts/`.

- [ ] **Step 4: Install and verify package checks**

Run: `uv sync --python 3.12; uv run pytest tests/test_config.py -v; uv run ruff check .; uv run ruff format --check .`

Expected: one passing test and no lint or format violations.

- [ ] **Step 5: Commit the scaffold**

```bash
git add pyproject.toml .gitignore src/evidence_rag_bench tests/test_config.py uv.lock
git commit -m "build: scaffold reproducible Python package"
```

### Task 2: Implement provenance validation and deterministic chunking

**Files:**
- Create: `src/evidence_rag_bench/models.py`
- Create: `src/evidence_rag_bench/corpus/__init__.py`
- Create: `src/evidence_rag_bench/corpus/manifest.py`
- Create: `src/evidence_rag_bench/corpus/chunking.py`
- Create: `data/corpus/manifest.jsonl`
- Create: `data/corpus/fixtures/retrieval_notes.txt`
- Create: `tests/corpus/test_manifest.py`
- Create: `tests/corpus/test_chunking.py`

**Interfaces:**
- Produces: `DocumentRecord`, `Chunk`, `load_manifest(path: Path) -> list[DocumentRecord]`, `validate_manifest(records: list[DocumentRecord], project_root: Path) -> None`, and `chunk_document(record: DocumentRecord, project_root: Path, chunk_size_words: int = 80, overlap_words: int = 20) -> list[Chunk]`.
- Consumes: `Settings` paths from Task 1.

- [ ] **Step 1: Write failing manifest and chunk-ID tests**

```python
def test_manifest_rejects_checksum_mismatch(tmp_path: Path) -> None:
    record = DocumentRecord(doc_id="notes", title="Notes", source_url="https://example.org/notes", license="CC-BY-4.0", retrieved_at="2026-09-01", text_path="data/corpus/fixtures/notes.txt", sha256="0" * 64, scope_note="fixture")
    (tmp_path / "data/corpus/fixtures").mkdir(parents=True)
    (tmp_path / record.text_path).write_text("evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_manifest([record], tmp_path)


def test_chunking_uses_stable_ids(tmp_path: Path) -> None:
    record = write_valid_record(tmp_path, "one two three four five six")
    chunks = chunk_document(record, tmp_path, chunk_size_words=3, overlap_words=1)
    assert [chunk.chunk_id for chunk in chunks] == ["notes:0000", "notes:0001", "notes:0002"]
```

- [ ] **Step 2: Verify both tests fail**

Run: `uv run pytest tests/corpus/test_manifest.py tests/corpus/test_chunking.py -v`

Expected: import failure because the corpus modules and models do not exist.

- [ ] **Step 3: Implement strict models, manifest checks, and word-window chunking**

```python
class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    source_url: AnyHttpUrl
    license: str
    retrieved_at: date
    text_path: str
    sha256: str
    scope_note: str


class Chunk(BaseModel):
    doc_id: str
    chunk_id: str
    source_url: str
    text: str
    ordinal: int
```

Validate required non-empty metadata, resolve paths below `project_root`, compare `hashlib.sha256(path.read_bytes()).hexdigest()` to the manifest, and make chunk windows advance by `chunk_size_words - overlap_words`. Reject invalid sizes where overlap is not smaller than chunk size.

Add one small CC-BY fixture record and calculate its real SHA-256 with `Get-FileHash -Algorithm SHA256` before adding it to the manifest.

- [ ] **Step 4: Verify corpus behavior and all checks**

Run: `uv run pytest tests/corpus -v; uv run ruff check .; uv run ruff format --check .`

Expected: all corpus tests pass, including a valid fixture-manifest validation test.

- [ ] **Step 5: Commit corpus foundation**

```bash
git add src/evidence_rag_bench/models.py src/evidence_rag_bench/corpus data/corpus tests/corpus
git commit -m "feat: add validated corpus chunking"
```

### Task 3: Add BM25 retrieval, versioned evaluation cases, and benchmark reports

**Files:**
- Create: `src/evidence_rag_bench/retrieval/__init__.py`
- Create: `src/evidence_rag_bench/retrieval/bm25.py`
- Create: `src/evidence_rag_bench/evaluation/__init__.py`
- Create: `src/evidence_rag_bench/evaluation/cases.py`
- Create: `src/evidence_rag_bench/evaluation/metrics.py`
- Create: `src/evidence_rag_bench/evaluation/runner.py`
- Create: `data/eval/dev.jsonl`
- Create: `data/eval/test.jsonl`
- Create: `tests/retrieval/test_bm25.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_runner.py`

**Interfaces:**
- Produces: `BM25Retriever(chunks: Sequence[Chunk])`, `BM25Retriever.search(query: str, k: int) -> list[RetrievedChunk]`, `EvaluationCase`, `load_cases(path: Path) -> list[EvaluationCase]`, `retrieval_metrics(results: Mapping[str, list[str]], cases: Sequence[EvaluationCase], k: int) -> dict[str, float]`, and `run_retrieval_benchmark(retriever: BM25Retriever, cases: Sequence[EvaluationCase], k: int, metadata: dict[str, str]) -> BenchmarkReport`.
- Consumes: `Chunk` from Task 2.

- [ ] **Step 1: Write failing BM25 ranking and metric tests**

```python
def test_bm25_returns_matching_chunk_first() -> None:
    retriever = BM25Retriever([chunk("a:0000", "retrieval uses sparse lexical matching"), chunk("b:0000", "citations connect claims to sources")])
    assert retriever.search("lexical retrieval", k=1)[0].chunk_id == "a:0000"


def test_recall_at_k_counts_gold_evidence() -> None:
    cases = [case("q1", ["a:0000"]), case("q2", ["b:0000"])]
    metrics = retrieval_metrics({"q1": ["a:0000"], "q2": ["a:0000"]}, cases, k=1)
    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr_at_1"] == 0.5
```

- [ ] **Step 2: Verify retrieval and metric tests fail**

Run: `uv run pytest tests/retrieval/test_bm25.py tests/evaluation/test_metrics.py -v`

Expected: import failure because retrieval and evaluation modules do not exist.

- [ ] **Step 3: Implement deterministic retrieval and reportable metrics**

```python
class RetrievedChunk(Chunk):
    score: float
    stage: Literal["bm25"] = "bm25"


class BM25Retriever:
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class EvaluationCase(BaseModel):
    case_id: str
    split: Literal["dev", "test"]
    question: str
    answerability: Literal["answerable", "ambiguous", "insufficient", "unanswerable"]
    gold_chunk_ids: list[str]
    reference_answer: str | None = None
    notes: str
```

Tokenize by lowercase whitespace for the baseline. Define `recall_at_k`, `mrr_at_k`, and `ndcg_at_k` as `0.0` for cases without gold chunks and exclude those cases from the retrieval denominator. Have the runner save a JSON report containing results per case, aggregate metrics, `corpus_manifest_sha256`, `retriever`, `k`, `created_at`, and `git_revision`.

Seed the development and test JSONL files with at least four synthetic but realistic fixture questions each, including one insufficient and one unanswerable case per split. Use only IDs present in the fixture manifest.

- [ ] **Step 4: Verify benchmark report generation**

Run: `uv run pytest tests/retrieval tests/evaluation -v; uv run python -m evidence_rag_bench.evaluation.runner --split dev --k 3`

Expected: all tests pass and a JSON report appears under `artifacts/reports/`.

- [ ] **Step 5: Commit retrieval and benchmark baseline**

```bash
git add src/evidence_rag_bench/retrieval src/evidence_rag_bench/evaluation data/eval tests/retrieval tests/evaluation
git commit -m "feat: add BM25 benchmark baseline"
```

### Task 4: Implement grounded answers, citation validation, and calibrated abstention

**Files:**
- Create: `src/evidence_rag_bench/grounding/__init__.py`
- Create: `src/evidence_rag_bench/grounding/citations.py`
- Create: `src/evidence_rag_bench/grounding/service.py`
- Create: `src/evidence_rag_bench/grounding/calibration.py`
- Create: `tests/grounding/test_citations.py`
- Create: `tests/grounding/test_service.py`
- Create: `tests/grounding/test_calibration.py`

**Interfaces:**
- Produces: `Citation(chunk_id: str)`, `validate_citations(citations: Sequence[Citation], evidence: Sequence[RetrievedChunk]) -> CitationValidation`, `AskResult`, `answer_question(question: str, retriever: BM25Retriever, threshold: float, top_k: int) -> AskResult`, and `select_threshold(dev_scores: Sequence[ScoredCase]) -> float`.
- Consumes: `BM25Retriever`, `RetrievedChunk`, and `EvaluationCase` from Task 3.

- [ ] **Step 1: Write failing evidence and abstention tests**

```python
def test_validator_rejects_citation_not_in_evidence() -> None:
    result = validate_citations([Citation(chunk_id="other:0000")], [retrieved("notes:0000", 2.0)])
    assert result.is_valid is False
    assert result.invalid_ids == ["other:0000"]


def test_low_retrieval_score_abstains() -> None:
    result = answer_question("unknown question", retriever_with_score(0.1), threshold=0.5, top_k=3)
    assert result.status == "abstain"
    assert result.reason == "insufficient_evidence"
    assert result.citations == []


def test_threshold_selection_uses_best_dev_f1() -> None:
    assert select_threshold([scored(0.2, False), scored(0.8, True)]) == 0.8
```

- [ ] **Step 2: Verify grounding tests fail**

Run: `uv run pytest tests/grounding -v`

Expected: import failure because the grounding modules do not exist.

- [ ] **Step 3: Add a deterministic formatter and strict response contract**

```python
class AskResult(BaseModel):
    status: Literal["answer", "abstain"]
    answer: str
    reason: str | None
    citations: list[Citation]
    evidence: list[RetrievedChunk]
    latency_ms: float
    trace_id: str
    mode: Literal["deterministic"] = "deterministic"
```

Use the highest-scoring chunk as the deterministic answer text and attach its exact chunk ID. If the top score is below threshold, evidence is empty, or validation fails, produce `status="abstain"`, a short corpus-limitation explanation, no citations, and an explicit reason. In `select_threshold`, enumerate unique development scores in ascending order, compute F1 for classifying answerable cases, and use the higher threshold on ties to reduce unsupported answers.

- [ ] **Step 4: Verify grounding behavior and regression suite**

Run: `uv run pytest tests/grounding -v; uv run pytest -v; uv run ruff check .; uv run ruff format --check .`

Expected: all citation, abstention, calibration, and existing tests pass.

- [ ] **Step 5: Commit grounded-response safeguards**

```bash
git add src/evidence_rag_bench/grounding tests/grounding
git commit -m "feat: add citation validation and abstention"
```

### Task 5: Deliver a tested FastAPI service and evidence-first UI

**Files:**
- Create: `src/evidence_rag_bench/api/__init__.py`
- Create: `src/evidence_rag_bench/api/app.py`
- Create: `src/evidence_rag_bench/ui/index.html`
- Create: `src/evidence_rag_bench/ui/app.js`
- Create: `src/evidence_rag_bench/ui/styles.css`
- Create: `tests/api/test_app.py`

**Interfaces:**
- Produces: `create_app(project_root: Path | None = None) -> FastAPI` and routes `GET /health`, `POST /v1/ask`, `POST /v1/evaluations/run`, and `GET /v1/evaluations/{report_id}`.
- Consumes: `get_settings`, BM25 pipeline functions, `answer_question`, and benchmark runner from Tasks 1--4.

- [ ] **Step 1: Write failing API contract tests**

```python
def test_health_reports_ready_client() -> None:
    response = TestClient(create_app(fixture_root)).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_evidence_bound_citations() -> None:
    response = TestClient(create_app(fixture_root)).post("/v1/ask", json={"question": "How does lexical retrieval work?", "top_k": 3})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] in {"answer", "abstain"}
    assert {citation["chunk_id"] for citation in body["citations"]} <= {item["chunk_id"] for item in body["evidence"]}
```

- [ ] **Step 2: Verify the API tests fail**

Run: `uv run pytest tests/api/test_app.py -v`

Expected: import failure because `create_app` does not exist.

- [ ] **Step 3: Implement routes and a static, accessible UI**

```python
def create_app(project_root: Path | None = None) -> FastAPI:
    app = FastAPI(title="Evidence RAG Bench", version="0.1.0")
    app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui")
    return app
```

Use Pydantic request validation with `question` trimmed to 1--1000 characters and `top_k` constrained to 1--10. Return `422` for invalid requests, `404` for a missing report, and a clear `503` JSON error if the configured corpus cannot initialize. The HTML must include one question field, submit button, answer status, abstention reason, latency, and evidence cards with source links. JavaScript must use `fetch('/v1/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(...)})` and render text with `textContent`, never `innerHTML` for model content.

- [ ] **Step 4: Verify API and manual demo flow**

Run: `uv run pytest tests/api -v; uv run uvicorn evidence_rag_bench.api.app:create_app --factory --port 8000`

Expected: API tests pass; opening `http://127.0.0.1:8000/` shows an answer-or-abstain result and source evidence for a fixture query.

- [ ] **Step 5: Commit service and UI**

```bash
git add src/evidence_rag_bench/api src/evidence_rag_bench/ui tests/api
git commit -m "feat: add evidence RAG API and demo UI"
```

### Task 6: Add CI, recruiter-facing documentation, and final verification

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `README.md`
- Create: `docs/decision-log.md`
- Create: `docs/evaluation-rubric.md`
- Modify: `data/eval/dev.jsonl`
- Modify: `data/eval/test.jsonl`
- Create: `tests/test_no_network.py`

**Interfaces:**
- Produces: a public quickstart, a documented evaluation command, CI status, and a no-network guarantee for test execution.
- Consumes: all commands and data contracts from Tasks 1--5.

- [ ] **Step 1: Write failing safety and dataset-split tests**

```python
def test_dev_and_test_case_ids_are_disjoint() -> None:
    dev_ids = {case.case_id for case in load_cases(Path("data/eval/dev.jsonl"))}
    test_ids = {case.case_id for case in load_cases(Path("data/eval/test.jsonl"))}
    assert dev_ids.isdisjoint(test_ids)


def test_test_modules_do_not_import_remote_provider_clients() -> None:
    forbidden = ("openai", "requests", "httpx.Client")
    text = "\n".join(path.read_text(encoding="utf-8") for path in Path("tests").rglob("*.py"))
    assert not any(token in text for token in forbidden)
```

- [ ] **Step 2: Verify documentation guardrail tests fail before they are added**

Run: `uv run pytest tests/test_no_network.py -v`

Expected: file-not-found error because the safety test has not been created.

- [ ] **Step 3: Add CI, polished documentation, and enough benchmark cases for the MVP**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install 3.12
      - run: uv sync --dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -v
```

The README must state the benchmark goal, architecture, headline local fixture result table, quickstart (`uv sync --python 3.12`, benchmark command, API command), metrics definitions, sample API request/response, limitations, data provenance, and extension path for dense retrieval. The decision log must record the choice of BM25 baseline, fixed corpus, deterministic formatter, and dev-only threshold tuning. The rubric must define how a human reviewer scores answer support, citation support, abstention quality, and clarity. Expand each split to at least 10 cases while preserving answerability diversity and disjoint IDs.

- [ ] **Step 4: Run the complete local release gate and inspect the report**

Run: `uv run pytest -v; uv run ruff check .; uv run ruff format --check .; uv run python -m evidence_rag_bench.evaluation.runner --split dev --k 3; uv run python -m evidence_rag_bench.evaluation.runner --split test --k 3; git status --short`

Expected: all checks pass; both report JSON files include retrieval, citation, abstention, latency, corpus hash, configuration, and Git revision; Git status shows only deliberately generated ignored artifacts.

- [ ] **Step 5: Commit the release-ready MVP and push**

```bash
git add .github README.md docs data/eval tests/test_no_network.py
git commit -m "docs: publish reproducible benchmark MVP"
git push origin main
```

## Plan Self-Review

**Spec coverage:** Task 1 implements the Python 3.12/uv contract. Task 2 implements manifest provenance, integrity checks, and deterministic chunks. Task 3 implements required BM25, JSONL cases, retrieval metrics, reports, and held-out split preservation. Task 4 implements citation-only evidence, abstention reasons, and development-only threshold calibration. Task 5 implements the API and browser demo. Task 6 implements CI, reproducibility documentation, manual rubric, data expansion, and the release gate. Dense/hybrid retrieval and reranking are intentionally milestone-five extensions from the design document rather than MVP requirements.

**Placeholder scan:** All tasks name exact files, concrete tests, executable commands, interfaces, and expected behavior. No deferred implementation markers are present.

**Type consistency:** `DocumentRecord` and `Chunk` are introduced in Task 2; `RetrievedChunk` extends `Chunk` in Task 3; grounding and API consume `RetrievedChunk` thereafter. `AskResult` is the answer route response contract. Evaluation uses `EvaluationCase` consistently for both splits.
