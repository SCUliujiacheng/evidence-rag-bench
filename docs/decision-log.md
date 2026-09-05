# Decision log

## Use a fixed, license-attributed corpus

The manifest keeps source URLs, license labels, retrieval date, local paths,
and SHA-256 checksums in versioned JSONL. That lets someone trace a result back
to the exact input, instead of trusting that an upstream page stayed still.

## Compare three local retrievers before adding providers

BM25, word/bigram TF-IDF, and reciprocal-rank fusion run offline and on the
same chunks. The first held-out run selected RRF Hybrid because it improved
MRR@3 and nDCG@3, not because it was assumed to be better.

After the corpus grew, BM25 led the development split on retrieval coverage,
while Hybrid retained a positive TF-IDF relevance signal that lets the API
abstain on an unseen query. The API therefore defaults to Hybrid for the demo;
benchmark tables still show every retriever rather than pretending there is one
universal winner.

## Keep rank score separate from abstention confidence

RRF scores only describe rank position. The system now retains a TF-IDF
relevance score for abstention and calibrates its threshold from development
cases only. Held-out results demonstrate that lexical confidence is still an
insufficient semantic-support signal; see `benchmark-results.md`.

## Do not present citation validity as factuality

The deterministic formatter guarantees that cited IDs belong to returned
evidence. This is a provenance property, not proof that a natural-language
claim is entailed by a passage. A future semantic verifier must be measured
separately on a held-out support annotation set.
