"""Versioned corpus profile contracts."""

from dataclasses import dataclass
from typing import Literal

ProfileId = Literal["en-v1", "zh-v1"]
DEFAULT_PROFILE_ID: ProfileId = "en-v1"


@dataclass(frozen=True)
class BenchmarkProfile:
    """Files and deterministic retrieval rules for one corpus version."""

    profile_id: ProfileId
    manifest_filename: str
    dev_case_filename: str
    test_case_filename: str
    chunking_mode: str
    tokenizer_name: str
    chunk_size: int
    overlap: int
    semantic_rerank_supported: bool


PROFILES: dict[ProfileId, BenchmarkProfile] = {
    "en-v1": BenchmarkProfile(
        profile_id="en-v1",
        manifest_filename="open_source_manifest.jsonl",
        dev_case_filename="open_source_dev.jsonl",
        test_case_filename="open_source_test.jsonl",
        chunking_mode="word-window",
        tokenizer_name="english-word",
        chunk_size=80,
        overlap=20,
        semantic_rerank_supported=True,
    ),
    "zh-v1": BenchmarkProfile(
        profile_id="zh-v1",
        manifest_filename="zh_v1_manifest.jsonl",
        dev_case_filename="zh_v1_dev.jsonl",
        test_case_filename="zh_v1_test.jsonl",
        chunking_mode="unicode-window",
        tokenizer_name="cjk-bigram-latin-word",
        chunk_size=220,
        overlap=40,
        semantic_rerank_supported=False,
    ),
}


def get_profile(profile_id: str) -> BenchmarkProfile:
    """Return one known benchmark profile."""

    try:
        return PROFILES[profile_id]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"unsupported profile: {profile_id}") from error
