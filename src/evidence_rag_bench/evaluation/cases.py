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


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load non-blank JSONL evaluation case records."""

    cases = [
        EvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("evaluation file contains no cases")
    return cases


def cases_with_gold(cases: Iterable[EvaluationCase]) -> list[EvaluationCase]:
    """Return cases that have evidence labels for retrieval metrics."""

    return [case for case in cases if case.gold_chunk_ids]
