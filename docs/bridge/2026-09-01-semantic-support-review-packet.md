BEGIN_CONTEXT_PACKET
packet_id: evidence-rag-semantic-support-20260901
requested_reasoner: GPT-5.6 Sol

## Objective
Choose the smallest credible next architecture step that improves semantic
support verification and answer/abstain behavior in a public, local-first RAG
research project without hiding current limitations.

## Acceptance
- A concrete, independently testable design for a semantic support verifier or
  equivalent guardrail that can be evaluated separately from retrieval.
- It must run on a Windows laptop with 16 GB RAM and an RTX 4050 Laptop GPU
  (about 6 GB VRAM), while retaining an offline/no-provider baseline.
- It must preserve a fixed corpus, development/test separation, provenance,
  and explicit failure reporting.

## Repository State
- Public GitHub repository `SCUliujiacheng/evidence-rag-bench`, branch `main`,
  clean working tree at the start of this review.
- Python 3.12 + uv, FastAPI, Pydantic, rank-bm25, scikit-learn; no Docker.
- Current corpus: 6 hash-locked MIT/BSD-3 open-source technical documents,
  77 deterministic 80-word chunks; development and held-out test JSONL each
  contain 12 cases.

## Evidence
- `src/evidence_rag_bench/retrieval/{bm25,tfidf,hybrid}.py`: BM25, word/bigram
  TF-IDF, and reciprocal-rank fusion local retrievers.
- `docs/benchmark-results.md`: held-out retrieval @3: BM25 recall 0.89/MRR
  0.54/nDCG 0.63; TF-IDF 0.78/0.61/0.65; Hybrid 0.78/0.56/0.61.
- `src/evidence_rag_bench/grounding/service.py`: deterministic formatter cites
  only returned evidence and abstains below an offline relevance threshold.
- `src/evidence_rag_bench/evaluation/grounding_metrics.py`: reports
  abstention precision/recall, false-answer rate, false-abstain rate,
  citation-valid rate, p50/p95 latency.
- `docs/benchmark-results.md`: a development-calibrated lexical threshold had
  held-out false-answer rate 0.50 and false-abstain rate 0.50. Citation validity
  was 1.00, but valid citation IDs do not prove textual entailment.
- Browser/API tests prove cited evidence rendering and an unrelated natural
  language query abstains; the broader semantic-support limitation remains.

## Constraints
- Do not send private data, credentials, user identifiers, source file bodies,
  or non-public code. This packet describes a public repository only.
- Do not recommend changing held-out labels or using them to choose thresholds.
- Avoid a paid API or always-on service as the only implementation.
- Prefer a small local model, an optional provider adapter, or a disciplined
  staged design with deterministic tests. Exact model licences and hardware
  fit must be locally verified before adoption.

## Questions
1. What architecture should this project implement next to credibly distinguish
   "citation ID is valid" from "evidence semantically supports the answer"?
2. How should that verifier be evaluated and integrated without contaminating
   the held-out split or turning the project into an untestable model demo?
3. What risks or claims should the README avoid when presenting this narrow
   benchmark to technical readers?
END_CONTEXT_PACKET
