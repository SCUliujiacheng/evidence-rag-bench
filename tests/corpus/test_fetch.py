import hashlib
from pathlib import Path

from evidence_rag_bench.corpus.fetch import fetch_document
from evidence_rag_bench.models import DocumentRecord


def test_fetch_document_writes_bytes_after_checksum_validation(tmp_path: Path) -> None:
    payload = b"retrieval systems need auditable evidence"
    record = DocumentRecord(
        doc_id="source",
        title="Source",
        source_url="https://example.org/source.txt",
        license="MIT",
        retrieved_at="2026-09-01",
        text_path="data/corpus/sources/source.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        scope_note="test fixture",
    )

    fetch_document(record, tmp_path, fetch_bytes=lambda _: payload)

    assert (tmp_path / record.text_path).read_bytes() == payload
