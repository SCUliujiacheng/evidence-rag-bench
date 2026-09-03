# Open-source corpus attribution

The files in `data/corpus/open_source/` are exact, hash-locked upstream README
documents. They are downloaded only through `evidence_rag_bench.corpus.fetch`
and validated against `data/corpus/open_source_manifest.jsonl` before indexing.
They are included to make the benchmark reproducible, not to imply endorsement
by their authors.

| Document | Upstream repository | License | Source |
| --- | --- | --- | --- |
| FAISS README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Benchmarks README | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| FAISS Installation Guide | Meta FAISS | MIT | https://github.com/facebookresearch/faiss |
| scikit-learn README | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn Contributing Guide | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| scikit-learn Getting Started Guide | scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn |
| LangChain README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Package README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Core README | LangChain | MIT | https://github.com/langchain-ai/langchain |
| LangChain Text Splitters README | LangChain | MIT | https://github.com/langchain-ai/langchain |

Each source's original license remains applicable. The corpus manifest records
the direct raw source URL, date, local path, SHA-256 checksum, and intended
benchmark scope. To refresh a source, first update its manifest hash and
attribution after reviewing upstream license changes; do not overwrite a locked
file in place.
