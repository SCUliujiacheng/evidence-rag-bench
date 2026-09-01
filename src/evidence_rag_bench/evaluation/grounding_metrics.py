"""Metrics for answer-versus-abstain behavior on labeled benchmark cases."""

from collections.abc import Mapping, Sequence

from evidence_rag_bench.evaluation.cases import EvaluationCase


def abstention_metrics(
    statuses: Mapping[str, str], cases: Sequence[EvaluationCase]
) -> dict[str, float]:
    """Measure whether abstentions match the case answerability labels."""

    answerable_cases = [case for case in cases if case.answerability == "answerable"]
    nonanswerable_cases = [case for case in cases if case.answerability != "answerable"]
    abstentions = {case_id for case_id, status in statuses.items() if status == "abstain"}
    correct_abstentions = sum(case.case_id in abstentions for case in nonanswerable_cases)
    false_abstentions = sum(case.case_id in abstentions for case in answerable_cases)
    false_answers = sum(case.case_id not in abstentions for case in nonanswerable_cases)
    total_abstentions = len(abstentions)
    return {
        "abstention_precision": correct_abstentions / total_abstentions
        if total_abstentions
        else 0.0,
        "abstention_recall": (
            correct_abstentions / len(nonanswerable_cases) if nonanswerable_cases else 0.0
        ),
        "false_answer_rate": false_answers / len(nonanswerable_cases)
        if nonanswerable_cases
        else 0.0,
        "false_abstain_rate": false_abstentions / len(answerable_cases)
        if answerable_cases
        else 0.0,
    }
