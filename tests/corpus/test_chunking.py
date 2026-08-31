import hashlib
from datetime import date
from pathlib import Path

from evidence_rag_bench.corpus.chunking import chunk_document
from evidence_rag_bench.models import DocumentRecord


def test_chunking_uses_stable_ids(tmp_path: Path) -> None:
    relative_path = "data/corpus/fixtures/notes.txt"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text("one two three four five six", encoding="utf-8")
    record = DocumentRecord(
        doc_id="notes",
        title="Notes",
        source_url="https://example.org/notes",
        license="CC-BY-4.0",
        retrieved_at=date(2026, 9, 1),
        text_path=relative_path,
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        scope_note="fixture",
    )

    chunks = chunk_document(record, tmp_path, chunk_size_words=3, overlap_words=1)

    assert [chunk.chunk_id for chunk in chunks] == ["notes:0000", "notes:0001", "notes:0002"]
    assert [chunk.text for chunk in chunks] == ["one two three", "three four five", "five six"]
