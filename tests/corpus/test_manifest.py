from datetime import date
from pathlib import Path

import pytest

from evidence_rag_bench.corpus.manifest import validate_manifest
from evidence_rag_bench.models import DocumentRecord


def test_manifest_rejects_checksum_mismatch(tmp_path: Path) -> None:
    relative_path = "data/corpus/fixtures/notes.txt"
    source_path = tmp_path / relative_path
    source_path.parent.mkdir(parents=True)
    source_path.write_text("evidence", encoding="utf-8")
    record = DocumentRecord(
        doc_id="notes",
        title="Notes",
        source_url="https://example.org/notes",
        license="CC-BY-4.0",
        retrieved_at=date(2026, 9, 1),
        text_path=relative_path,
        sha256="0" * 64,
        scope_note="fixture",
    )

    with pytest.raises(ValueError, match="checksum"):
        validate_manifest([record], tmp_path)
