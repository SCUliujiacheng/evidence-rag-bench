from evidence_rag_bench.models import Chunk, RetrievedChunk
from evidence_rag_bench.retrieval.rerank import SemanticReranker


class StaticRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        return self.results[:k]


class StaticScorer:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.2, 0.9][: len(passages)]


def test_semantic_reranker_reorders_candidates_using_the_passage_scores() -> None:
    first = Chunk(
        doc_id="docs",
        chunk_id="docs:0000",
        source_url="https://example.org",
        text="first",
        ordinal=0,
    )
    second = Chunk(
        doc_id="docs",
        chunk_id="docs:0001",
        source_url="https://example.org",
        text="second",
        ordinal=1,
    )
    retriever = StaticRetriever(
        [
            RetrievedChunk(**first.model_dump(), score=0.8, stage="hybrid"),
            RetrievedChunk(**second.model_dump(), score=0.7, stage="hybrid"),
        ]
    )

    results = SemanticReranker(retriever, StaticScorer()).search("query", k=2)

    assert [result.chunk_id for result in results] == ["docs:0001", "docs:0000"]
    assert [result.score for result in results] == [0.9, 0.2]
    assert all(result.stage == "rerank" for result in results)
