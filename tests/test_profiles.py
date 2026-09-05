import pytest

from evidence_rag_bench.profiles import DEFAULT_PROFILE_ID, get_profile


def test_english_profile_keeps_the_existing_benchmark_files() -> None:
    profile = get_profile("en-v1")

    assert DEFAULT_PROFILE_ID == "en-v1"
    assert profile.manifest_filename == "open_source_manifest.jsonl"
    assert profile.dev_case_filename == "open_source_dev.jsonl"
    assert profile.test_case_filename == "open_source_test.jsonl"
    assert profile.chunking_mode == "word-window"
    assert profile.tokenizer_name == "english-word"
    assert profile.semantic_rerank_supported is True


def test_chinese_profile_uses_its_own_corpus_cases_and_retrieval_rules() -> None:
    profile = get_profile("zh-v1")

    assert profile.manifest_filename == "zh_v1_manifest.jsonl"
    assert profile.dev_case_filename == "zh_v1_dev.jsonl"
    assert profile.test_case_filename == "zh_v1_test.jsonl"
    assert profile.chunking_mode == "unicode-window"
    assert profile.tokenizer_name == "cjk-bigram-latin-word"
    assert profile.semantic_rerank_supported is False


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported profile: xx-v1"):
        get_profile("xx-v1")
