from pathlib import Path

from evidence_rag_bench.evaluation.cases import load_cases


def test_dev_and_test_case_ids_are_disjoint() -> None:
    dev_ids = {case.case_id for case in load_cases(Path("data/eval/dev.jsonl"))}
    test_ids = {case.case_id for case in load_cases(Path("data/eval/test.jsonl"))}

    assert dev_ids.isdisjoint(test_ids)


def test_test_modules_do_not_import_remote_provider_clients() -> None:
    forbidden = ("openai", "requests", "httpx.Client")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("tests").rglob("*.py")
        if path.name != "test_no_network.py"
    )

    assert not any(token in text for token in forbidden)
