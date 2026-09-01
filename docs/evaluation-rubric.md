# Manual evaluation rubric

For each future corpus expansion, sample answerable, ambiguous, insufficient,
and unanswerable cases from the held-out set. Two reviewers independently score
each case, then record disagreements before changing any model or threshold.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Retrieval support | Gold evidence absent | Relevant but incomplete evidence | Directly supporting evidence retrieved |
| Citation support | Missing or unrelated | Valid ID but partial support | Citation directly supports the displayed answer |
| Abstention | Unsafe answer or needless refusal | Borderline decision | Correct answer/refusal with clear rationale |
| Answer clarity | Misleading | Understandable but vague | Concise and scope-bounded |

Record the case ID, corpus-manifest hash, Git revision, retrieval configuration,
and reviewer rationale. Never use held-out labels to choose chunk size, RRF
weights, or abstention threshold; propose changes on development cases, then
rerun the frozen held-out protocol.
