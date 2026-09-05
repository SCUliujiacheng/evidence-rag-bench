# Decision log

## Use a fixed, license-attributed corpus

The manifest keeps source URLs, license labels, retrieval date, local paths,
and SHA-256 checksums in versioned JSONL. That lets someone trace a result back
to the exact input, instead of trusting that an upstream page stayed still.

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

## Citation IDs are not factuality checks

The deterministic formatter guarantees that cited IDs belong to returned
evidence. This is a provenance property, not proof that a natural-language
claim is entailed by a passage. A future semantic verifier needs a new, sealed
support annotation set.
