"""Run BM25 retrieval benchmarks and write provenance-rich JSON reports."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evidence_rag_bench.config import get_settings
from evidence_rag_bench.corpus.chunking import chunk_document
from evidence_rag_bench.corpus.manifest import load_manifest, validate_manifest
from evidence_rag_bench.evaluation.cases import EvaluationCase, load_cases
from evidence_rag_bench.evaluation.metrics import retrieval_metrics
from evidence_rag_bench.retrieval.bm25 import BM25Retriever


class CaseResult(BaseModel):
    """One retrieval trace included in a benchmark report."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    retrieved_chunk_ids: list[str]


class BenchmarkReport(BaseModel):
    """A serializable aggregate result for one retrieval run."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    case_results: list[CaseResult]
    metadata: dict[str, str]


def run_retrieval_benchmark(
    retriever: BM25Retriever,
    cases: list[EvaluationCase],
    k: int,
    metadata: dict[str, str],
) -> BenchmarkReport:
    """Retrieve evidence for every case and calculate aggregate metrics."""

    case_results = [
        CaseResult(
            case_id=case.case_id,
            retrieved_chunk_ids=[result.chunk_id for result in retriever.search(case.question, k)],
        )
        for case in cases
    ]
    result_mapping = {result.case_id: result.retrieved_chunk_ids for result in case_results}
    return BenchmarkReport(
        metrics=retrieval_metrics(result_mapping, cases, k),
        case_results=case_results,
        metadata={**metadata, "retriever": "bm25", "k": str(k)},
    )


def git_revision(project_root: Path) -> str:
    """Return the current revision, or an explicit fallback outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_split(project_root: Path, split: str, k: int) -> tuple[BenchmarkReport, Path]:
    """Build a local BM25 index, evaluate one split, and persist the report."""

    settings = get_settings(project_root)
    manifest_path = settings.corpus_dir / "manifest.jsonl"
    records = load_manifest(manifest_path)
    validate_manifest(records, settings.project_root)
    chunks = [
        chunk for record in records for chunk in chunk_document(record, settings.project_root)
    ]
    cases = [
        case for case in load_cases(settings.eval_dir / f"{split}.jsonl") if case.split == split
    ]
    if not cases:
        raise ValueError(f"no {split} cases found")
    report = run_retrieval_benchmark(
        BM25Retriever(chunks),
        cases,
        k,
        {
            "corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "git_revision": git_revision(settings.project_root),
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
        },
    )
    report_dir = settings.artifacts_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"bm25-{split}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, report_path


def main() -> None:
    """Run a named benchmark split from the command line."""

    parser = argparse.ArgumentParser(description="Run the Evidence RAG BM25 benchmark.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--k", type=int, default=3)
    arguments = parser.parse_args()
    _, report_path = run_split(Path.cwd(), arguments.split, arguments.k)
    print(json.dumps({"report_path": str(report_path)}))


if __name__ == "__main__":
    main()
