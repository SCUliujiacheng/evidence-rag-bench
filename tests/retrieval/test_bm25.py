from evidence_rag_bench.models import Chunk
from evidence_rag_bench.retrieval.bm25 import BM25Retriever


def test_bm25_returns_matching_chunk_first() -> None:
    retriever = BM25Retriever(
        [
            Chunk(
                doc_id="a",
                chunk_id="a:0000",
                source_url="https://example.org/a",
                text="retrieval uses sparse lexical matching",
                ordinal=0,
            ),
            Chunk(
                doc_id="b",
                chunk_id="b:0000",
                source_url="https://example.org/b",
                text="citations connect claims to sources",
                ordinal=0,
            ),
        ]
    )

    results = retriever.search("lexical retrieval", k=1)

    assert results[0].chunk_id == "a:0000"
    assert results[0].stage == "bm25"
