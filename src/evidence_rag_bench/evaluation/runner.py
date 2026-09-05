"""Run retrieval benchmarks and write JSON reports with run metadata."""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evidence_rag_bench.config import get_settings
from evidence_rag_bench.corpus.chunking import chunk_document, indexed_document_bytes
from evidence_rag_bench.corpus.manifest import load_manifest, validate_manifest
from evidence_rag_bench.evaluation.cases import EvaluationCase, load_cases, validate_case_protocol
from evidence_rag_bench.evaluation.grounding_metrics import abstention_metrics
from evidence_rag_bench.evaluation.metrics import retrieval_metrics
from evidence_rag_bench.grounding.calibration import ScoredCase, select_threshold
from evidence_rag_bench.grounding.service import AskResult, answer_question
from evidence_rag_bench.models import Chunk, DocumentRecord
from evidence_rag_bench.profiles import DEFAULT_PROFILE_ID, PROFILES, get_profile
from evidence_rag_bench.retrieval.bm25 import BM25Retriever, mixed_cjk_bigram_tokenize
from evidence_rag_bench.retrieval.hybrid import HybridRetriever
from evidence_rag_bench.retrieval.rerank import (
    PassageScorer,
    SemanticReranker,
    SentenceTransformersCrossEncoder,
)
from evidence_rag_bench.retrieval.tfidf import TfidfRetriever

SEMANTIC_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def indexed_corpus_sha256(records: Sequence[DocumentRecord], project_root: Path) -> str:
    """Hash the ordered document IDs and the exact byte ranges sent to chunking."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(record.doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(indexed_document_bytes(record, project_root))
        digest.update(b"\0")
    return digest.hexdigest()


def index_scope(records: Sequence[DocumentRecord]) -> str:
    """Describe any manifest rule that excludes a raw-document suffix."""

    marked = [
        f"{record.doc_id}:end-before:{record.index_end_marker}"
        for record in records
        if record.index_end_marker is not None
    ]
    return ";".join(marked) if marked else "full-document"


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


class GroundedCaseResult(BaseModel):
    """One answer-or-abstain trace for end-to-end evaluation."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    status: str
    citation_ids: list[str]
    evidence_ids: list[str]
    latency_ms: float


class GroundedBenchmarkReport(BaseModel):
    """Aggregate end-to-end grounding behavior for a fixed case set."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, float]
    case_results: list[GroundedCaseResult]
    metadata: dict[str, str]


def run_retrieval_benchmark(
    retriever: BM25Retriever | TfidfRetriever | HybridRetriever | SemanticReranker,
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
        metadata={**metadata, "k": str(k)},
    )


def run_grounded_benchmark(
    retriever: BM25Retriever | TfidfRetriever | HybridRetriever | SemanticReranker,
    cases: list[EvaluationCase],
    top_k: int,
    threshold: float,
    metadata: dict[str, str] | None = None,
) -> GroundedBenchmarkReport:
    """Run answer/abstain behavior and measure citation validity and latency."""

    answers: list[tuple[EvaluationCase, AskResult]] = [
        (case, answer_question(case.question, retriever, threshold, top_k)) for case in cases
    ]
    statuses = {case.case_id: answer.status for case, answer in answers}
    case_results = [
        GroundedCaseResult(
            case_id=case.case_id,
            status=answer.status,
            citation_ids=[citation.chunk_id for citation in answer.citations],
            evidence_ids=[evidence.chunk_id for evidence in answer.evidence],
            latency_ms=answer.latency_ms,
        )
        for case, answer in answers
    ]
    citation_valid_count = sum(
        set(result.citation_ids).issubset(result.evidence_ids)
        for result in case_results
        if result.status == "answer"
    )
    answer_count = sum(result.status == "answer" for result in case_results)
    cases_by_id = {case.case_id: case for case in cases}
    citation_total = sum(len(result.citation_ids) for result in case_results)
    supported_citation_count = sum(
        len(set(result.citation_ids) & set(cases_by_id[result.case_id].gold_chunk_ids))
        for result in case_results
    )
    gold_total = sum(len(case.gold_chunk_ids) for case in cases)
    sorted_latencies = sorted(result.latency_ms for result in case_results)
    percentile_index = max(0, round(0.95 * len(sorted_latencies)) - 1)
    return GroundedBenchmarkReport(
        metrics={
            **abstention_metrics(statuses, cases),
            "citation_valid_rate": citation_valid_count / answer_count if answer_count else 0.0,
            "citation_precision_against_gold": (
                supported_citation_count / citation_total if citation_total else 0.0
            ),
            "citation_recall_against_gold": (
                supported_citation_count / gold_total if gold_total else 0.0
            ),
            "unsupported_citation_count": float(citation_total - supported_citation_count),
            "latency_p50_ms": sorted_latencies[len(sorted_latencies) // 2],
            "latency_p95_ms": sorted_latencies[percentile_index],
        },
        case_results=case_results,
        metadata=metadata or {},
    )


def build_retriever(
    retriever_name: str,
    chunks: list[Chunk],
    semantic_scorer: PassageScorer | None = None,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> BM25Retriever | TfidfRetriever | HybridRetriever | SemanticReranker:
    """Construct one named local retrieval baseline over the same chunks."""

    profile = get_profile(profile_id)
    tokenizer = (
        mixed_cjk_bigram_tokenize if profile.tokenizer_name == "cjk-bigram-latin-word" else None
    )
    if retriever_name == "bm25":
        return BM25Retriever(chunks) if tokenizer is None else BM25Retriever(chunks, tokenizer)
    if retriever_name == "tfidf":
        return TfidfRetriever(chunks, tokenizer)
    if retriever_name == "hybrid":
        return HybridRetriever(chunks, tokenizer=tokenizer)
    if retriever_name == "semantic-rerank":
        if not profile.semantic_rerank_supported:
            raise ValueError(
                f"semantic-rerank model {SEMANTIC_RERANK_MODEL} does not support profile {profile_id}"
            )
        scorer = semantic_scorer or SentenceTransformersCrossEncoder(SEMANTIC_RERANK_MODEL)
        return SemanticReranker(HybridRetriever(chunks, tokenizer=tokenizer), scorer)
    raise ValueError(f"unsupported retriever: {retriever_name}")


def git_revision(project_root: Path) -> str:
    """Return the current revision, or an explicit fallback outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_split(
    project_root: Path,
    split: str,
    k: int,
    retriever_name: str = "bm25",
    manifest_filename: str | None = None,
    case_filename: str | None = None,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> tuple[BenchmarkReport, Path]:
    """Build a local BM25 index, evaluate one split, and persist the report."""

    settings = get_settings(project_root)
    profile = get_profile(profile_id)
    expected_case_filename = (
        profile.dev_case_filename if split == "dev" else profile.test_case_filename
    )
    for label, provided, expected in (
        ("manifest", manifest_filename, profile.manifest_filename),
        ("case file", case_filename, expected_case_filename),
    ):
        if provided is not None and provided != expected:
            raise ValueError(
                f"{label} {provided} does not belong to profile {profile.profile_id}; "
                f"expected {expected}"
            )
    manifest_filename = profile.manifest_filename
    case_filename = expected_case_filename
    manifest_path = settings.corpus_dir / manifest_filename
    records = load_manifest(manifest_path)
    validate_manifest(records, settings.project_root)
    chunks = [
        chunk
        for record in records
        for chunk in chunk_document(
            record,
            settings.project_root,
            chunk_size_words=profile.chunk_size,
            overlap_words=profile.overlap,
            mode=profile.chunking_mode,
        )
    ]
    cases_path = settings.eval_dir / case_filename
    cases = [case for case in load_cases(cases_path) if case.split == split]
    if not cases:
        raise ValueError(f"no {split} cases found")
    report = run_retrieval_benchmark(
        build_retriever(retriever_name, chunks, profile_id=profile_id),
        cases,
        k,
        {
            "corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "git_revision": git_revision(settings.project_root),
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
            "retriever": retriever_name,
            "manifest_filename": manifest_filename,
            "case_filename": cases_path.name,
            "profile": profile_id,
            "chunking_mode": profile.chunking_mode,
            "chunk_size": str(profile.chunk_size),
            "chunk_overlap": str(profile.overlap),
            "tokenizer": profile.tokenizer_name,
            "indexed_corpus_sha256": indexed_corpus_sha256(records, settings.project_root),
            "index_scope": index_scope(records),
            **(
                {
                    "semantic_model": SEMANTIC_RERANK_MODEL,
                    "semantic_candidate_k": str(max(k, 10)),
                }
                if retriever_name == "semantic-rerank"
                else {}
            ),
        },
    )
    report_dir = settings.artifacts_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    corpus_label = Path(manifest_filename).stem.replace("_manifest", "")
    report_path = report_dir / f"{corpus_label}-{retriever_name}-{split}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, report_path


def run_grounded_split(
    project_root: Path,
    split: str,
    top_k: int,
    retriever_name: str = "hybrid",
    manifest_filename: str | None = None,
    case_filename: str | None = None,
    threshold: float | None = None,
    calibration_case_filename: str | None = None,
    profile_id: str = DEFAULT_PROFILE_ID,
) -> tuple[GroundedBenchmarkReport, Path]:
    """Run end-to-end answer/abstain evaluation and persist a JSON report."""

    settings = get_settings(project_root)
    profile = get_profile(profile_id)
    expected_case_filename = (
        profile.dev_case_filename if split == "dev" else profile.test_case_filename
    )
    for label, provided, expected in (
        ("manifest", manifest_filename, profile.manifest_filename),
        ("case file", case_filename, expected_case_filename),
        ("calibration case file", calibration_case_filename, profile.dev_case_filename),
    ):
        if provided is not None and provided != expected:
            raise ValueError(
                f"{label} {provided} does not belong to profile {profile.profile_id}; "
                f"expected {expected}"
            )
    manifest_filename = profile.manifest_filename
    case_filename = expected_case_filename
    manifest_path = settings.corpus_dir / manifest_filename
    records = load_manifest(manifest_path)
    validate_manifest(records, settings.project_root)
    chunks = [
        chunk
        for record in records
        for chunk in chunk_document(
            record,
            settings.project_root,
            chunk_size_words=profile.chunk_size,
            overlap_words=profile.overlap,
            mode=profile.chunking_mode,
        )
    ]
    cases_path = settings.eval_dir / case_filename
    cases = [case for case in load_cases(cases_path) if case.split == split]
    if not cases:
        raise ValueError(f"no {split} cases found")
    retriever = build_retriever(retriever_name, chunks, profile_id=profile_id)
    if threshold is None and calibration_case_filename is None:
        calibration_case_filename = profile.dev_case_filename
    calibration_path = (
        settings.eval_dir / calibration_case_filename if calibration_case_filename else None
    )
    if threshold is None and calibration_path is not None:
        calibration_cases = [case for case in load_cases(calibration_path) if case.split == "dev"]
        if not calibration_cases:
            raise ValueError("no dev cases available for threshold calibration")
        if calibration_path != cases_path:
            validate_case_protocol([*cases, *calibration_cases])
        scored_cases = []
        for case in calibration_cases:
            results = retriever.search(case.question, top_k)
            score = results[0].relevance_score if results else 0.0
            scored_cases.append(
                ScoredCase(score=score or 0.0, answerable=case.answerability == "answerable")
            )
        threshold = select_threshold(scored_cases)
    effective_threshold = threshold if threshold is not None else 0.0
    report = run_grounded_benchmark(
        retriever,
        cases,
        top_k=top_k,
        threshold=effective_threshold,
        metadata={
            "corpus_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "git_revision": git_revision(settings.project_root),
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
            "retriever": retriever_name,
            "manifest_filename": manifest_filename,
            "case_filename": cases_path.name,
            "profile": profile_id,
            "chunking_mode": profile.chunking_mode,
            "chunk_size": str(profile.chunk_size),
            "chunk_overlap": str(profile.overlap),
            "tokenizer": profile.tokenizer_name,
            "indexed_corpus_sha256": indexed_corpus_sha256(records, settings.project_root),
            "index_scope": index_scope(records),
            "top_k": str(top_k),
            "abstention_threshold": str(effective_threshold),
            "threshold_source": calibration_path.name
            if calibration_path
            else "explicit_or_default",
            **(
                {
                    "semantic_model": SEMANTIC_RERANK_MODEL,
                    "semantic_candidate_k": str(max(top_k, 10)),
                }
                if retriever_name == "semantic-rerank"
                else {}
            ),
        },
    )
    report_dir = settings.artifacts_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    corpus_label = Path(manifest_filename).stem.replace("_manifest", "")
    report_path = report_dir / f"{corpus_label}-{retriever_name}-{split}-grounded.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report, report_path


def main() -> None:
    """Run a named benchmark split from the command line."""

    parser = argparse.ArgumentParser(description="Run an Evidence RAG retrieval benchmark.")
    parser.add_argument("--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE_ID)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=("bm25", "tfidf", "hybrid", "semantic-rerank"),
        default="bm25",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--cases")
    parser.add_argument(
        "--calibration-cases",
        help="development JSONL used to choose the grounded abstention threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="explicit grounded abstention threshold; overrides calibration",
    )
    parser.add_argument("--mode", choices=("retrieval", "grounded"), default="retrieval")
    arguments = parser.parse_args()
    if arguments.mode == "grounded":
        _, report_path = run_grounded_split(
            Path.cwd(),
            arguments.split,
            arguments.k,
            arguments.retriever,
            arguments.manifest,
            arguments.cases,
            arguments.threshold,
            arguments.calibration_cases,
            arguments.profile,
        )
    else:
        _, report_path = run_split(
            Path.cwd(),
            arguments.split,
            arguments.k,
            arguments.retriever,
            arguments.manifest,
            arguments.cases,
            arguments.profile,
        )
    print(json.dumps({"report_path": str(report_path)}))


if __name__ == "__main__":
    main()
