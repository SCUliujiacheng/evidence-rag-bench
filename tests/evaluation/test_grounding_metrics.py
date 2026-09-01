from evidence_rag_bench.evaluation.cases import EvaluationCase
from evidence_rag_bench.evaluation.grounding_metrics import abstention_metrics


def test_abstention_metrics_distinguish_false_answers_and_false_abstentions() -> None:
    cases = [
        EvaluationCase(
            case_id="answerable",
            split="dev",
            question="supported",
            answerability="answerable",
            gold_chunk_ids=["doc:0000"],
            reference_answer="answer",
            notes="fixture",
        ),
        EvaluationCase(
            case_id="insufficient",
            split="dev",
            question="unsupported",
            answerability="insufficient",
            gold_chunk_ids=[],
            reference_answer=None,
            notes="fixture",
        ),
    ]

    metrics = abstention_metrics({"answerable": "abstain", "insufficient": "answer"}, cases)

    assert metrics["abstention_precision"] == 0.0
    assert metrics["abstention_recall"] == 0.0
    assert metrics["false_answer_rate"] == 1.0
    assert metrics["false_abstain_rate"] == 1.0
