"""Optional semantic re-ranking over a deterministic candidate retriever."""

from collections.abc import Sequence
from typing import Protocol

from evidence_rag_bench.models import RetrievedChunk


class CandidateRetriever(Protocol):
    """Return candidate passages for a textual query."""

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return a stable candidate ranking."""


class PassageScorer(Protocol):
    """Score query/passage pairs with a semantic cross-encoder."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage in input order."""


class SemanticReranker:
    """Re-rank fixed lexical candidates while retaining their provenance."""

    def __init__(
        self, retriever: CandidateRetriever, scorer: PassageScorer, candidate_k: int = 10
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least one")
        self._retriever = retriever
        self._scorer = scorer
        self._candidate_k = candidate_k

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the highest semantic-scoring subset of lexical candidates."""

        if k < 1:
            raise ValueError("k must be at least one")
        candidates = self._retriever.search(query, max(k, self._candidate_k))
        scores = self._scorer.score(query, [candidate.text for candidate in candidates])
        if len(scores) != len(candidates):
            raise ValueError("passage scorer must return one score per candidate")
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda item: -item[1])[:k]
        return [
            RetrievedChunk(
                **candidate.model_dump(exclude={"score", "relevance_score", "stage"}),
                score=score,
                relevance_score=score,
                stage="rerank",
            )
            for candidate, score in ranked
        ]


class SentenceTransformersCrossEncoder:
    """Lazy adapter for a local sentence-transformers CrossEncoder model."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "install the semantic extra with `uv sync --extra semantic` to use a CrossEncoder"
            ) from error
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Run the local model for each query/passage pair."""

        if not passages:
            return []
        scores: Sequence[float] = self._model.predict([(query, passage) for passage in passages])
        return [float(score) for score in scores]
