import hashlib
from pathlib import Path

from evidence_rag_bench.corpus.fetch import fetch_manifest
from evidence_rag_bench.models import DocumentRecord


def test_fetch_manifest_writes_each_checksum_locked_document(tmp_path: Path) -> None:
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

    fetched = fetch_manifest([record], tmp_path, fetch_bytes=lambda _: payload)

    assert (tmp_path / record.text_path).read_bytes() == payload
    assert fetched == [tmp_path / record.text_path]
