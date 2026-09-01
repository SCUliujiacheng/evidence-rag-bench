"""Reciprocal-rank fusion of local BM25 and TF-IDF retrievers."""

from collections.abc import Sequence

from evidence_rag_bench.models import Chunk, RetrievedChunk
from evidence_rag_bench.retrieval.bm25 import BM25Retriever
from evidence_rag_bench.retrieval.tfidf import TfidfRetriever


class HybridRetriever:
    """Fuse independent lexical rankings with reciprocal-rank fusion."""

    def __init__(self, chunks: Sequence[Chunk], rrf_k: int = 60) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least one")
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = BM25Retriever(chunks)
        self._tfidf = TfidfRetriever(chunks)
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Fuse all local candidates and return the top ``k`` stable results."""

        if k < 1:
            raise ValueError("k must be at least one")
        candidate_count = len(self._chunks_by_id)
        fused_scores: dict[str, float] = {}
        for ranking in (
            self._bm25.search(query, candidate_count),
            self._tfidf.search(query, candidate_count),
        ):
            for rank, result in enumerate(ranking, start=1):
                if result.score <= 0:
                    continue
                fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1 / (
                    self._rrf_k + rank
                )
        ranked_ids = sorted(fused_scores, key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id))[
            :k
        ]
        return [
            RetrievedChunk(
                **self._chunks_by_id[chunk_id].model_dump(),
                score=fused_scores[chunk_id],
                stage="hybrid",
            )
            for chunk_id in ranked_ids
        ]
