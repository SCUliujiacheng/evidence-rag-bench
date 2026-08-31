"""Deterministic, word-window corpus chunking."""

from pathlib import Path

from evidence_rag_bench.corpus.manifest import resolve_document_path
from evidence_rag_bench.models import Chunk, DocumentRecord


def chunk_document(
    record: DocumentRecord,
    project_root: Path,
    chunk_size_words: int = 80,
    overlap_words: int = 20,
) -> list[Chunk]:
    """Create stable, overlapping chunks for a source document."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_size_words")

    words = resolve_document_path(record, project_root).read_text(encoding="utf-8").split()
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    for ordinal, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunks.append(
            Chunk(
                doc_id=record.doc_id,
                chunk_id=f"{record.doc_id}:{ordinal:04d}",
                source_url=str(record.source_url),
                text=" ".join(window),
                ordinal=ordinal,
            )
        )
        if start + chunk_size_words >= len(words):
            break
    return chunks
