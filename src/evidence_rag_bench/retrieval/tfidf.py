"""Offline TF-IDF vector retrieval baseline."""

from collections.abc import Callable, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from evidence_rag_bench.models import Chunk, RetrievedChunk


class TfidfRetriever:
    """Rank chunks with deterministic word and bigram TF-IDF vectors."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("TfidfRetriever requires at least one chunk")
        self._chunks = list(chunks)
        if tokenizer is None:
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                stop_words="english",
            )
        else:
            self._vectorizer = TfidfVectorizer(
                analyzer=tokenizer,
                lowercase=False,
            )
        self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in self._chunks)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks by cosine similarity to the query vector."""

        if k < 1:
            raise ValueError("k must be at least one")
        query_vector = self._vectorizer.transform([query])
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        ranked_indices = sorted(
            range(len(self._chunks)), key=lambda index: (-scores[index], index)
        )[:k]
        return [
            RetrievedChunk(
                **self._chunks[index].model_dump(), score=float(scores[index]), stage="tfidf"
            )
            for index in ranked_indices
        ]
