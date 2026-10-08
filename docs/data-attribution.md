# Corpus attribution

The files under `data/corpus/open_source/` and `data/corpus/zh_v1/` are exact,
hash-locked upstream documents. They are downloaded only through
`evidence_rag_bench.corpus.fetch` and validated against the matching manifest
before indexing. They are included so a run can be repeated, not to imply
endorsement by their authors.

## `en-v1`

| Document | Upstream repository | License | Source |
| --- | --- | --- | --- |
| FAISS README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Benchmarks README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Installation Guide | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS C API Installation Guide | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Demos README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| scikit-learn README | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn Contributing Guide | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn Getting Started Guide | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn FAQ | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| LangChain README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Package README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Core README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Text Splitters README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Standard Tests README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain OpenAI Integration README | LangChain | MIT | https://github.com/langchain-ai/langchain |

## `zh-v1`

| Document | Fixed upstream revision | License | SHA-256 |
| --- | --- | --- | --- |
| [Milvus 中文 README](https://raw.githubusercontent.com/milvus-io/milvus/88997528bb78437217d795d3fc995cf9a8f4f0a8/README_CN.md) | [`88997528`](https://github.com/milvus-io/milvus/commit/88997528bb78437217d795d3fc995cf9a8f4f0a8) | Apache-2.0 | `f939969055f2ce1c4d9f4e73076a7a8baa22b5718c98b85760f97889e90e919a` |
| [PaddlePaddle 中文 README](https://raw.githubusercontent.com/PaddlePaddle/Paddle/df1fbe1dd882ee4b82bb5d9741a1357e27dc51e6/README_cn.md) | [`df1fbe1d`](https://github.com/PaddlePaddle/Paddle/commit/df1fbe1dd882ee4b82bb5d9741a1357e27dc51e6) | Apache-2.0 | `298952feff920a35512b47c9ce025fb801fbfd9958029a457cd55e00276e21d8` |
| [FlagEmbedding 中文 README](https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/fd1a2bdf69488ffebe0327999d4400d8c8058a0b/README_zh.md) | [`fd1a2bdf`](https://github.com/FlagOpen/FlagEmbedding/commit/fd1a2bdf69488ffebe0327999d4400d8c8058a0b) | MIT | `2f8388a058157f4777bb8c9a46da120c0fa062b40fe1299bb5fa563fa8c941b1` |

The manifest is [`data/corpus/zh_v1_manifest.jsonl`](../data/corpus/zh_v1_manifest.jsonl).
License files from the same three commits are kept under
[`data/corpus/licenses/zh_v1/`](../data/corpus/licenses/zh_v1/) and hash-locked
in [`zh_v1_license_manifest.jsonl`](../data/corpus/zh_v1_license_manifest.jsonl).
The two Apache repositories do not contain a `NOTICE` file at these revisions.

The Milvus README is stored as the complete raw file, including the long
contributor-avatar section near its end. Its manifest sets
`index_end_marker: "### All contributors"`, so chunking stops before those
avatars. The stored upstream file and its SHA-256 remain unchanged, while the
non-technical tail is excluded from retrieval. Reports include the marker rule and a
separate `indexed_corpus_sha256` for the exact bytes sent to chunking.

With the `zh-v1` 220-unit window and 40-unit overlap, the indexed ranges make
54 chunks: 16 from Milvus, 8 from PaddlePaddle, and 30 from FlagEmbedding. The
combined indexed-corpus SHA-256 is
`4eee4a7e0b302febe9570533c94c722e46f3335a7164c13523615b5a96ca7ef6`.
It is calculated in manifest order by hashing, for each record, the UTF-8
`doc_id`, a NUL byte, the exact indexed bytes, and another NUL byte. That makes
the reported hash sensitive to both source order and the byte range actually
used by retrieval.

Each source's original license remains applicable. The manifests record the
direct raw source URL, date, local path, SHA-256 checksum, and intended scope.
Refreshing a source means creating a new profile version after reviewing its
license and re-annotating chunk IDs; a locked profile should not move with an
upstream branch.
