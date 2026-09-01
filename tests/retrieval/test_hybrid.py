from evidence_rag_bench.models import Chunk
from evidence_rag_bench.retrieval.hybrid import HybridRetriever


def test_hybrid_returns_fused_results_with_stable_chunk_ids() -> None:
    chunks = [
        Chunk(
            doc_id="bm25",
            chunk_id="bm25:0000",
            source_url="https://example.org/bm25",
            text="BM25 ranks documents with lexical query terms.",
            ordinal=0,
        ),
        Chunk(
            doc_id="citation",
            chunk_id="citation:0000",
            source_url="https://example.org/citation",
            text="Citations connect generated claims to source evidence.",
            ordinal=0,
        ),
    ]
    retriever = HybridRetriever(chunks)

    results = retriever.search("lexical retrieval query", k=1)

    assert results[0].chunk_id == "bm25:0000"
    assert results[0].stage == "hybrid"


def test_hybrid_returns_no_evidence_for_an_unseen_query() -> None:
    retriever = HybridRetriever(
        [
            Chunk(
                doc_id="retrieval",
                chunk_id="retrieval:0000",
                source_url="https://example.org/retrieval",
                text="retrieval ranks source passages",
                ordinal=0,
            )
        ]
    )

    results = retriever.search("galactic orchestras", k=3)

    assert results == []
