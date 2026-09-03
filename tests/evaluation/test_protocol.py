import pytest

from evidence_rag_bench.evaluation.cases import EvaluationCase, validate_case_protocol


def _case(**overrides: object) -> EvaluationCase:
    payload: dict[str, object] = {
        "case_id": "dev-001",
        "split": "dev",
        "question": "What does the source support?",
        "answerability": "answerable",
        "gold_chunk_ids": ["source:0000"],
        "reference_answer": "Similarity search.",
        "notes": "fixture",
    }
    payload.update(overrides)
    return EvaluationCase.model_validate(payload)


def test_protocol_rejects_an_answerable_case_without_gold_evidence() -> None:
    with pytest.raises(ValueError, match="answerable cases require gold_chunk_ids"):
        validate_case_protocol([_case(gold_chunk_ids=[])])


def test_protocol_rejects_the_same_question_across_dev_and_test() -> None:
    with pytest.raises(ValueError, match="reused across splits"):
        validate_case_protocol([_case(), _case(case_id="test-001", split="test")])


def test_protocol_accepts_disjoint_answerable_and_unanswerable_cases() -> None:
    validate_case_protocol(
        [
            _case(),
            _case(
                case_id="test-001",
                split="test",
                question="Which galaxy won a prize?",
                answerability="unanswerable",
                gold_chunk_ids=[],
                reference_answer=None,
            ),
        ]
    )
