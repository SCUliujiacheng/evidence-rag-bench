from evidence_rag_bench.evaluation.cases import EvaluationCase
from evidence_rag_bench.evaluation.runner import run_retrieval_benchmark
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
