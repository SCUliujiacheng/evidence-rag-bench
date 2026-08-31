"""Read and validate auditable corpus manifests."""

import hashlib
import json
from pathlib import Path

from evidence_rag_bench.models import DocumentRecord


def load_manifest(path: Path) -> list[DocumentRecord]:
    """Parse a non-empty JSONL manifest into validated document records."""

    records: list[DocumentRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(DocumentRecord.model_validate_json(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid manifest record on line {line_number}: {error}") from error
    if not records:
        raise ValueError("manifest contains no document records")
    return records


def resolve_document_path(record: DocumentRecord, project_root: Path) -> Path:
    """Resolve one record path while preventing paths outside the project root."""

    root = project_root.resolve()
    path = (root / record.text_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"document path escapes project root: {record.text_path}")
    return path


def validate_manifest(records: list[DocumentRecord], project_root: Path) -> None:
    """Confirm document identifiers are unique and source bytes match the manifest."""

    seen_ids: set[str] = set()
    for record in records:
        if record.doc_id in seen_ids:
            raise ValueError(f"duplicate document id: {record.doc_id}")
        seen_ids.add(record.doc_id)

        path = resolve_document_path(record, project_root)
        if not path.is_file():
            raise ValueError(f"source text is missing: {record.text_path}")
        actual_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_checksum != record.sha256:
            raise ValueError(f"checksum mismatch for document {record.doc_id}")
