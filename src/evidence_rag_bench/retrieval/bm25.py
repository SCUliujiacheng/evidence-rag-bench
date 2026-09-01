"""A deterministic BM25 retrieval baseline."""

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from evidence_rag_bench.models import Chunk, RetrievedChunk


def tokenize(text: str) -> list[str]:
    """Tokenize the baseline with lowercase whitespace terms."""

    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in ENGLISH_STOP_WORDS
    ]


class BM25Retriever:
    """Rank chunks with a local lexical BM25 index."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self._chunks = list(chunks)
        self._index = BM25Okapi([tokenize(chunk.text) for chunk in self._chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks in descending BM25 score order."""

        if k < 1:
            raise ValueError("k must be at least one")
        scores = self._index.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(self._chunks)), key=lambda index: (-scores[index], index))[:k]
        return [
            RetrievedChunk(
                **self._chunks[index].model_dump(), score=float(scores[index]), stage="bm25"
            )
            for index in ranked_indices
        ]
