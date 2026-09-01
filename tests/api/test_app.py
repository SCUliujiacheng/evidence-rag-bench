from pathlib import Path

from fastapi.testclient import TestClient

from evidence_rag_bench.api.app import create_app


def test_health_reports_ready_client() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["corpus_document_count"] == 3
    assert response.json()["retriever"] == "hybrid"


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
