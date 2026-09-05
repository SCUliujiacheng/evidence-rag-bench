# Manual evaluation rubric

When a profile grows, sample answerable, ambiguous, insufficient, and
unanswerable cases from that profile's fixed test set. Have two reviewers score
each case independently, then write down disagreements before changing a model
or threshold. Do not combine `en-v1` and `zh-v1` scores: their corpora, cases,
chunking rules, and development thresholds are separate.

For answerability labels, `answerable` requires a direct supporting passage;
`ambiguous` has multiple reasonable readings or asks for a preference the
corpus never resolves; `insufficient` stays on topic but lacks the requested
detail, value, algorithm, or causal claim; and `unanswerable` is outside the
fixed corpus or depends on live information absent from the snapshot.

`gold_chunk_ids` follows retrieval-qrels semantics: every passage that still
states both the subject and the supporting fact when read on its own is labeled.
Repeated statements at separate source locations both count; an overlapping
window fragment that loses the subject or model context does not. MRR uses the
first relevant hit, while Recall measures coverage of all labeled relevant
chunks; Recall is not the minimum number of passages needed to answer.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Retrieval support | Gold evidence absent | Relevant but incomplete evidence | Directly supporting evidence retrieved |
| Citation support | Missing or unrelated | Valid ID but partial support | Citation directly supports the displayed answer |
| Abstention | Unsafe answer or needless refusal | Borderline decision | Correct answer/refusal with clear rationale |
| Answer clarity | Misleading | Understandable but vague | Concise and scope-bounded |

Record the case ID, corpus-manifest hash, Git revision, retrieval configuration,
and reviewer rationale. Never use fixed test labels to choose chunk size, RRF
weights, or abstention threshold; propose changes on development cases, then
rerun the fixed regression protocol. Both current test sets were viewed during
development, so call them regression sets rather than unseen or held-out data.
