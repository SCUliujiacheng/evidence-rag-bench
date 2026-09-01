import hashlib
import json
from pathlib import Path

from evidence_rag_bench.evaluation.cases import EvaluationCase
from evidence_rag_bench.evaluation.runner import (
    build_retriever,
    run_retrieval_benchmark,
    run_split,
)
from evidence_rag_bench.models import Chunk
from evidence_rag_bench.retrieval.bm25 import BM25Retriever


def test_runner_returns_case_results_and_metadata() -> None:
    retriever = BM25Retriever(
        [
            Chunk(
                doc_id="notes",
                chunk_id="notes:0000",
                source_url="https://example.org/notes",
                text="BM25 uses lexical query terms",
                ordinal=0,
            )
        ]
    )
    cases = [
        EvaluationCase(
            case_id="dev-001",
            split="dev",
            question="What does BM25 use?",
            answerability="answerable",
            gold_chunk_ids=["notes:0000"],
            reference_answer="lexical query terms",
            notes="fixture",
        )
    ]

    report = run_retrieval_benchmark(
        retriever,
        cases,
        k=1,
        metadata={"corpus_manifest_sha256": "fixture", "git_revision": "test"},
    )

    assert report.metrics["recall_at_1"] == 1.0
    assert report.case_results[0].retrieved_chunk_ids == ["notes:0000"]
    assert report.metadata["git_revision"] == "test"


def test_build_retriever_selects_tfidf_baseline() -> None:
    chunks = [
        Chunk(
            doc_id="notes",
            chunk_id="notes:0000",
            source_url="https://example.org/notes",
            text="BM25 uses lexical query terms",
            ordinal=0,
        )
    ]

    retriever = build_retriever("tfidf", chunks)

    assert retriever.search("lexical", k=1)[0].stage == "tfidf"


def test_run_split_accepts_a_named_manifest_and_case_file(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "data" / "corpus"
    eval_dir = tmp_path / "data" / "eval"
    corpus_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    text_path = corpus_dir / "source.txt"
    text_path.write_text("A vector index supports similarity search.", encoding="utf-8")
    manifest = {
        "doc_id": "source",
        "title": "Source",
        "source_url": "https://example.org/source.txt",
        "license": "MIT",
        "retrieved_at": "2026-09-01",
        "text_path": "data/corpus/source.txt",
        "sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "scope_note": "fixture",
    }
    (corpus_dir / "custom.jsonl").write_text(json.dumps(manifest), encoding="utf-8")
    case = {
        "case_id": "custom-001",
        "split": "dev",
        "question": "What does a vector index support?",
        "answerability": "answerable",
        "gold_chunk_ids": ["source:0000"],
        "reference_answer": "similarity search",
        "notes": "fixture",
    }
    (eval_dir / "custom_dev.jsonl").write_text(json.dumps(case), encoding="utf-8")

    report, _ = run_split(
        tmp_path,
        "dev",
        1,
        manifest_filename="custom.jsonl",
        case_filename="custom_dev.jsonl",
    )

    assert report.metrics["recall_at_1"] == 1.0
