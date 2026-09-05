from pathlib import Path

from fastapi.testclient import TestClient

from evidence_rag_bench.api.app import create_app


def test_health_reports_ready_client() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["corpus_document_count"] == 15
    assert response.json()["retriever"] == "hybrid"
    assert response.json()["abstention_threshold"] > 0
    assert response.json()["default_profile"] == "en-v1"
    assert response.json()["profiles"] == ["en-v1", "zh-v1"]


def test_ask_returns_evidence_bound_citations() -> None:
    project_root = Path(__file__).parents[2]
    client = TestClient(create_app(project_root))

    response = client.post(
        "/v1/ask",
        json={"question": "How can FAISS implement cosine similarity?", "top_k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answer"
    assert body["citations"] == [{"chunk_id": "faiss-readme:0002"}]
    assert body["evidence"][0]["chunk_id"] == "faiss-readme:0002"
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


def test_ask_can_search_the_chinese_profile() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).post(
        "/v1/ask",
        json={"profile": "zh-v1", "question": "BGE-M3 支持哪些检索方式？", "top_k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["profile"] == "zh-v1"
    assert body["status"] == "answer"
    assert any(item["doc_id"] == "flagembedding-readme-zh" for item in body["evidence"])


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


def test_evaluation_endpoint_keeps_chinese_reports_separate() -> None:
    project_root = Path(__file__).parents[2]
    client = TestClient(create_app(project_root))

    response = client.post(
        "/v1/evaluations/run",
        json={"profile": "zh-v1", "split": "test", "k": 3},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["report_id"] == "zh-v1-test"
    assert body["report"]["metadata"]["profile"] == "zh-v1"

    saved_report = client.get("/v1/evaluations/zh-v1-test")

    assert saved_report.status_code == 200
    assert saved_report.json()["metadata"]["case_filename"] == "zh_v1_test.jsonl"


def test_demo_serves_a_vector_favicon() -> None:
    project_root = Path(__file__).parents[2]
    response = TestClient(create_app(project_root)).get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")


def test_homepage_sets_a_plain_expectation_for_the_evidence_viewer() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/")

    assert response.status_code == 200
    assert "Search the fixed corpus. If the evidence is thin, the demo says so." in response.text
    assert "Find supporting passages" in response.text
    assert "How can FAISS implement cosine similarity?" in response.text
    assert '<select id="profile"' in response.text
    assert '<option value="zh-v1">中文语料' in response.text


def test_demo_explains_abstention_reasons_in_plain_english() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/app.js")

    assert response.status_code == 200
    assert 'insufficient_evidence: "Not enough evidence was retrieved"' in response.text
    assert "RELEVANT EVIDENCE FOUND — VERIFY BELOW" in response.text
    assert "ANSWER — SOURCES BELOW" not in response.text
    assert "Reason:" in response.text
    assert "function readableSource(text)" in response.text
    assert "excerpt(item.text)" in response.text


def test_demo_sends_the_selected_profile_with_each_question() -> None:
    project_root = Path(__file__).parents[2]

    response = TestClient(create_app(project_root)).get("/app.js")

    assert response.status_code == 200
    assert "profile: profile.value" in response.text
