from evidence_rag_bench.grounding.service import answer_question
from evidence_rag_bench.models import RetrievedChunk


class LowScoreRetriever:
    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                doc_id="notes",
                chunk_id="notes:0000",
                source_url="https://example.org/notes",
                text="retrieval evidence",
                ordinal=0,
                score=0.1,
            )
        ]


def test_low_retrieval_score_abstains() -> None:
    result = answer_question("unknown question", LowScoreRetriever(), threshold=0.5, top_k=3)

    assert result.status == "abstain"
    assert result.reason == "insufficient_evidence"
    assert result.citations == []
