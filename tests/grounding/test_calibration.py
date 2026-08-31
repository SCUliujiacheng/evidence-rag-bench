from evidence_rag_bench.grounding.calibration import ScoredCase, select_threshold


def test_threshold_selection_uses_best_dev_f1() -> None:
    threshold = select_threshold(
        [
            ScoredCase(score=0.2, answerable=False),
            ScoredCase(score=0.8, answerable=True),
        ]
    )

    assert threshold == 0.8
