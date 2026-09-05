"""FastAPI application exposing auditable evidence-grounded answers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evidence_rag_bench.config import Settings, get_settings
from evidence_rag_bench.corpus.chunking import chunk_document
from evidence_rag_bench.corpus.manifest import load_manifest, validate_manifest
from evidence_rag_bench.evaluation.cases import load_cases
from evidence_rag_bench.evaluation.runner import run_split
from evidence_rag_bench.grounding.calibration import ScoredCase, select_threshold
from evidence_rag_bench.grounding.service import answer_question
from evidence_rag_bench.profiles import (
    DEFAULT_PROFILE_ID,
    PROFILES,
    BenchmarkProfile,
    ProfileId,
)
from evidence_rag_bench.retrieval.bm25 import mixed_cjk_bigram_tokenize
from evidence_rag_bench.retrieval.hybrid import HybridRetriever


class AskRequest(BaseModel):
    """Validated browser or API question payload."""

    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    profile: ProfileId = DEFAULT_PROFILE_ID


class EvaluationRequest(BaseModel):
    """Validated benchmark execution request."""

    split: Literal["dev", "test"]
    k: int = Field(default=3, ge=1, le=10)
    profile: ProfileId = DEFAULT_PROFILE_ID


@dataclass(frozen=True)
class ProfileServices:
    """One profile's index, corpus count, and independently calibrated threshold."""

    profile: BenchmarkProfile
    retriever: HybridRetriever
    corpus_document_count: int
    abstention_threshold: float


@dataclass(frozen=True)
class AppServices:
    """Immutable runtime dependencies shared by route handlers."""

    settings: Settings
    default_profile: ProfileId
    profiles: dict[ProfileId, ProfileServices]


def build_services(project_root: Path | None) -> AppServices:
    """Build both local profile indexes once at application creation."""

    settings = get_settings(project_root)
    profile_services: dict[ProfileId, ProfileServices] = {}
    for profile_id, profile in PROFILES.items():
        records = load_manifest(settings.corpus_dir / profile.manifest_filename)
        validate_manifest(records, settings.project_root)
        chunks = [
            chunk
            for record in records
            for chunk in chunk_document(
                record,
                settings.project_root,
                profile.chunk_size,
                profile.overlap,
                profile.chunking_mode,
            )
        ]
        tokenizer = (
            mixed_cjk_bigram_tokenize if profile.tokenizer_name == "cjk-bigram-latin-word" else None
        )
        retriever = HybridRetriever(chunks, tokenizer=tokenizer)
        dev_cases = load_cases(settings.eval_dir / profile.dev_case_filename)
        scored_cases = []
        for case in dev_cases:
            results = retriever.search(case.question, 3)
            score = results[0].relevance_score if results else 0.0
            scored_cases.append(
                ScoredCase(score=score or 0.0, answerable=case.answerability == "answerable")
            )
        profile_services[profile_id] = ProfileServices(
            profile=profile,
            retriever=retriever,
            corpus_document_count=len(records),
            abstention_threshold=select_threshold(scored_cases),
        )
    return AppServices(
        settings=settings,
        default_profile=DEFAULT_PROFILE_ID,
        profiles=profile_services,
    )


def create_app(project_root: Path | None = None) -> FastAPI:
    """Create a local API and static evidence viewer."""

    services = build_services(project_root)
    app = FastAPI(title="Evidence RAG Bench", version="0.2.0")
    ui_dir = Path(__file__).parents[1] / "ui"

    @app.get("/health")
    def health() -> dict[str, object]:
        default_services = services.profiles[services.default_profile]
        return {
            "status": "ok",
            "mode": "deterministic",
            "retriever": "hybrid",
            "corpus_document_count": default_services.corpus_document_count,
            "abstention_threshold": default_services.abstention_threshold,
            "default_profile": services.default_profile,
            "profiles": list(services.profiles),
        }

    @app.post("/v1/ask")
    def ask(request: AskRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=422, detail="question must contain non-whitespace characters"
            )
        selected = services.profiles[request.profile]
        result = answer_question(
            question,
            selected.retriever,
            threshold=selected.abstention_threshold,
            top_k=request.top_k,
        )
        return {**result.model_dump(mode="json"), "profile": request.profile}

    @app.post("/v1/evaluations/run")
    def run_evaluation(request: EvaluationRequest) -> dict[str, object]:
        report, report_path = run_split(
            services.settings.project_root,
            request.split,
            request.k,
            retriever_name="hybrid",
            profile_id=request.profile,
        )
        return {
            "report_id": f"{request.profile}-{request.split}",
            "report_path": str(report_path),
            "report": report,
        }

    @app.get("/v1/evaluations/{report_id}")
    def get_evaluation(report_id: str):
        profile_id: ProfileId = services.default_profile
        split = report_id
        for candidate in services.profiles:
            prefix = f"{candidate}-"
            if report_id.startswith(prefix):
                profile_id = candidate
                split = report_id.removeprefix(prefix)
                break
        if split not in {"dev", "test"}:
            raise HTTPException(status_code=404, detail="unknown report id")
        profile = services.profiles[profile_id].profile
        corpus_label = Path(profile.manifest_filename).stem.replace("_manifest", "")
        report_path = (
            services.settings.artifacts_dir / "reports" / f"{corpus_label}-hybrid-{split}.json"
        )
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="report has not been generated")
        return FileResponse(report_path, media_type="application/json")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(ui_dir / "index.html")

    @app.get("/app.js")
    def javascript() -> FileResponse:
        return FileResponse(ui_dir / "app.js", media_type="application/javascript")

    @app.get("/styles.css")
    def stylesheet() -> FileResponse:
        return FileResponse(ui_dir / "styles.css", media_type="text/css")

    @app.get("/favicon.svg")
    def favicon() -> FileResponse:
        return FileResponse(ui_dir / "favicon.svg", media_type="image/svg+xml")

    return app
