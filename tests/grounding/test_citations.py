from evidence_rag_bench.grounding.citations import Citation, validate_citations
from evidence_rag_bench.models import RetrievedChunk


def test_validator_rejects_citation_not_in_evidence() -> None:
    evidence = [
        RetrievedChunk(
            doc_id="notes",
            chunk_id="notes:0000",
            source_url="https://example.org/notes",
            text="evidence",
            ordinal=0,
            score=2.0,
        )
    ]

    result = validate_citations([Citation(chunk_id="other:0000")], evidence)

    assert result.is_valid is False
    assert result.invalid_ids == ["other:0000"]
