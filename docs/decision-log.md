# Decision log

## Use a fixed, license-attributed corpus

The manifest keeps source URLs, license labels, retrieval date, local paths,
and SHA-256 checksums in versioned JSONL. That lets someone trace a result back
to the exact input, instead of trusting that an upstream page stayed still.

`en-v1` keeps the original fifteen-document English benchmark unchanged.
`zh-v1` is a separate three-document Chinese snapshot with its own manifest,
development questions, test questions, chunk IDs, and threshold. Selecting a
language never mixes the two corpora in one index, and the runner rejects file
overrides that do not belong to the selected profile.

The raw Milvus README includes a large generated contributor-avatar block.
`zh-v1` keeps those upstream bytes for attribution but sets an explicit
`index_end_marker` before the block. The runner hashes the ordered byte ranges
that were actually indexed and writes both that digest and the scope rule into
every report.

## Tokenize Chinese without hiding a model dependency

The English path still uses the original word processing. For `zh-v1`, the
retrievers share a deterministic tokenizer: normalize with Unicode NFKC,
lowercase Latin text, keep Latin words and numbers, and turn each continuous
CJK span into overlapping character bigrams. A one-character CJK span stays as
one token.

Chinese chunks use 220 visible Unicode units with a 40-unit overlap. The
chunker prefers a sentence-ending mark in the latter half of a window and
returns an untouched slice of the source, so punctuation, newlines, and Latin
model names stay readable. This is deliberately simpler than learned word
segmentation and makes every chunk reproducible without another package.

## Compare three local retrievers before adding providers

BM25, word/bigram TF-IDF, and reciprocal-rank fusion run offline and on the
same chunks. Early test runs influenced the choice of RRF Hybrid because it
improved MRR@3 and nDCG@3. Those results are now treated as a regression set.

After the corpus grew, BM25 led the development split on retrieval coverage,
while Hybrid retained a positive TF-IDF relevance signal that lets the API
abstain on an unseen query. That is why the demo defaults to Hybrid. The tables
keep every retriever visible for direct comparison.

## Keep rank score separate from abstention confidence

RRF scores only describe rank position. The system now retains a TF-IDF
relevance score for abstention and calibrates its threshold from development
cases only. The fixed test snapshot shows that lexical confidence is still an
insufficient semantic-support signal; see `benchmark-results.md`.

Each profile calibrates this threshold from its own development JSONL. The
Chinese fixed test results show the limit clearly: topic-relevant passages for
an ambiguous or underspecified question can still score highly and produce a
false answer.

## Do not run the English MiniLM experiment on Chinese

`cross-encoder/ms-marco-MiniLM-L6-v2` is kept as the optional `en-v1`
experiment. `semantic-rerank` rejects `zh-v1` instead of silently using an
English-only checkpoint and presenting the result as Chinese support. A future
multilingual reranker needs a named model, a development protocol, and a new
reported comparison.

## Citation IDs are not factuality checks

The deterministic formatter guarantees that cited IDs belong to returned
evidence. This is a provenance property, not proof that a natural-language
claim is entailed by a passage. A future semantic verifier needs a new, sealed
support annotation set.
