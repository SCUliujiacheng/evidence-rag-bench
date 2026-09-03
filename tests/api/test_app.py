from pathlib import Path

from fastapi.testclient import TestClient

from evidence_rag_bench.api.app import create_app


def test_health_reports_ready_client() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["corpus_document_count"] == 10
    assert response.json()["retriever"] == "hybrid"
    assert response.json()["abstention_threshold"] > 0


def test_ask_returns_evidence_bound_citations() -> None:
    project_root = Path(__file__).parents[2]
    client = TestClient(create_app(project_root))

    response = client.post(
        "/v1/ask",
        json={"question": "How does lexical retrieval work?", "top_k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] in {"answer", "abstain"}
    assert {citation["chunk_id"] for citation in body["citations"]} <= {
        item["chunk_id"] for item in body["evidence"]
    }


def test_ask_abstains_when_the_corpus_has_no_query_evidence() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).post(
        "/v1/ask",
        json={"question": "Which galactic orchestra won a music prize?", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "abstain"
    assert response.json()["citations"] == []


def test_evaluation_endpoint_uses_the_open_source_benchmark() -> None:
    project_root = Path(__file__).parents[2]
    client = TestClient(create_app(project_root))
    response = client.post(
        "/v1/evaluations/run",
        json={"split": "test", "k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["report"]["metadata"]["manifest_filename"] == "open_source_manifest.jsonl"
    assert body["report"]["metadata"]["retriever"] == "hybrid"

    saved_report = client.get("/v1/evaluations/test")

    assert saved_report.status_code == 200
    assert saved_report.json()["metadata"]["case_filename"] == "open_source_test.jsonl"


def test_demo_serves_a_vector_favicon() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
