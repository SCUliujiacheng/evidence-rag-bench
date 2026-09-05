"""A deterministic BM25 retrieval baseline."""

import re
import unicodedata
from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from evidence_rag_bench.models import Chunk, RetrievedChunk


def tokenize(text: str) -> list[str]:
    """Tokenize the baseline with lowercase whitespace terms."""

    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in ENGLISH_STOP_WORDS
    ]


def mixed_cjk_bigram_tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and Latin text without external segmenters."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for match in re.finditer(
        r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002a6df]+|[a-z][a-z0-9]*|[0-9]+",
        normalized,
    ):
        term = match.group(0)
        if _is_cjk(term[0]):
            tokens.extend(
                [term]
                if len(term) == 1
                else [term[index : index + 2] for index in range(len(term) - 1)]
            )
        else:
            tokens.append(term)
    return tokens


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x2A6DF
    )


class BM25Retriever:
    """Rank chunks with a local lexical BM25 index."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")
        self._chunks = list(chunks)
        self._tokenizer = tokenizer
        self._index = BM25Okapi([self._tokenizer(chunk.text) for chunk in self._chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks in descending BM25 score order."""

        if k < 1:
            raise ValueError("k must be at least one")
        scores = self._index.get_scores(self._tokenizer(query))
        ranked_indices = sorted(
            range(len(self._chunks)), key=lambda index: (-scores[index], index)
        )[:k]
        return [
            RetrievedChunk(
                **self._chunks[index].model_dump(), score=float(scores[index]), stage="bm25"
            )
            for index in ranked_indices
        ]
