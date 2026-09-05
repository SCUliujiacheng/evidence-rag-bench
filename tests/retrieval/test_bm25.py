from evidence_rag_bench.models import Chunk
from evidence_rag_bench.retrieval.bm25 import BM25Retriever, mixed_cjk_bigram_tokenize


def test_mixed_cjk_tokenizer_keeps_latin_words_and_builds_chinese_bigrams() -> None:
    assert mixed_cjk_bigram_tokenize("BGE-M3 支持稠密、稀疏检索，RAG 2.0") == [
        "bge",
        "m3",
        "支持",
        "持稠",
        "稠密",
        "稀疏",
        "疏检",
        "检索",
        "rag",
        "2",
        "0",
    ]


def test_mixed_cjk_tokenizer_keeps_a_single_chinese_character() -> None:
    assert mixed_cjk_bigram_tokenize("用 GPU") == ["用", "gpu"]


def test_bm25_returns_matching_chunk_first() -> None:
    retriever = BM25Retriever(
        [
            Chunk(
                doc_id="a",
                chunk_id="a:0000",
                source_url="https://example.org/a",
                text="retrieval uses sparse lexical matching",
                ordinal=0,
            ),
            Chunk(
                doc_id="b",
                chunk_id="b:0000",
                source_url="https://example.org/b",
                text="citations connect claims to sources",
                ordinal=0,
            ),
        ]
    )

    results = retriever.search("lexical retrieval", k=1)

    assert results[0].chunk_id == "a:0000"
    assert results[0].stage == "bm25"


def test_bm25_can_rank_unsegmented_chinese_with_the_profile_tokenizer() -> None:
    retriever = BM25Retriever(
        [
            Chunk(
                doc_id="dense",
                chunk_id="dense:0000",
                source_url="https://example.org/dense",
                text="稠密检索会比较向量表示。",
                ordinal=0,
            ),
            Chunk(
                doc_id="install",
                chunk_id="install:0000",
                source_url="https://example.org/install",
                text="安装命令可以使用 pip。",
                ordinal=0,
            ),
        ],
        tokenizer=mixed_cjk_bigram_tokenize,
    )

    assert retriever.search("怎样做稠密检索？", k=1)[0].chunk_id == "dense:0000"
