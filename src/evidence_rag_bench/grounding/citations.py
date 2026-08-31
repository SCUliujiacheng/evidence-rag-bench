"""Validate that answer citations point to supplied evidence only."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from evidence_rag_bench.models import RetrievedChunk


class Citation(BaseModel):
    """One stable chunk identifier used to support a response claim."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=r"^[^:\s]+:\d{4}$")


class CitationValidation(BaseModel):
    """Machine-readable citation validation result."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool
    invalid_ids: list[str]
    duplicate_ids: list[str]


def validate_citations(
    citations: Sequence[Citation], evidence: Sequence[RetrievedChunk]
) -> CitationValidation:
    """Reject empty, unknown, and duplicate-only citation lists."""

    evidence_ids = {item.chunk_id for item in evidence}
    citation_ids = [citation.chunk_id for citation in citations]
    invalid_ids = [chunk_id for chunk_id in citation_ids if chunk_id not in evidence_ids]
    duplicate_ids = sorted(
        {chunk_id for chunk_id in citation_ids if citation_ids.count(chunk_id) > 1}
    )
    return CitationValidation(
        is_valid=bool(citation_ids) and not invalid_ids and not duplicate_ids,
        invalid_ids=invalid_ids,
        duplicate_ids=duplicate_ids,
    )
