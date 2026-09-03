"""Load version-controlled evaluation cases."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    """One question and its evidence-level evaluation annotation."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    split: Literal["dev", "test"]
    question: str = Field(min_length=1)
    answerability: Literal["answerable", "ambiguous", "insufficient", "unanswerable"]
    gold_chunk_ids: list[str]
    reference_answer: str | None = None
    notes: str = Field(min_length=1)


def validate_case_protocol(cases: Iterable[EvaluationCase]) -> None:
    """Reject labels that would invalidate evidence or split-level evaluation."""

    seen_case_ids: set[str] = set()
    question_splits: dict[str, str] = {}
    for case in cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        normalized_question = " ".join(case.question.lower().split())
        existing_split = question_splits.get(normalized_question)
        if existing_split is not None and existing_split != case.split:
            raise ValueError(f"question reused across splits: {case.question}")
        if existing_split is not None:
            raise ValueError(f"duplicate question within split: {case.question}")
        question_splits[normalized_question] = case.split
        if case.answerability == "answerable":
            if not case.gold_chunk_ids:
                raise ValueError("answerable cases require gold_chunk_ids")
            if not case.reference_answer:
                raise ValueError("answerable cases require reference_answer")
        elif case.gold_chunk_ids or case.reference_answer:
            raise ValueError("non-answerable cases cannot include answer evidence")


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load non-blank JSONL evaluation case records."""

    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("evaluation file contains no cases")
    validate_case_protocol(cases)
    return cases


def cases_with_gold(cases: Iterable[EvaluationCase]) -> list[EvaluationCase]:
    """Return cases that have evidence labels for retrieval metrics."""

    return [case for case in cases if case.gold_chunk_ids]
