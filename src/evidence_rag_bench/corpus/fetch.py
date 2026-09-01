"""Controlled retrieval of manifest-locked public corpus documents."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from urllib.request import Request, urlopen

from evidence_rag_bench.corpus.manifest import resolve_document_path
from evidence_rag_bench.models import DocumentRecord


def download_bytes(url: str) -> bytes:
    """Download one public source with a bounded timeout and identifying user agent."""

    request = Request(url, headers={"User-Agent": "evidence-rag-bench/0.1"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def fetch_document(
    record: DocumentRecord,
    project_root: Path,
    fetch_bytes: Callable[[str], bytes] = download_bytes,
) -> Path:
    """Fetch one source only when its bytes match the immutable manifest hash."""

    payload = fetch_bytes(str(record.source_url))
    actual_checksum = hashlib.sha256(payload).hexdigest()
    if actual_checksum != record.sha256:
        raise ValueError(f"checksum mismatch for downloaded document {record.doc_id}")
    destination = resolve_document_path(record, project_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination
