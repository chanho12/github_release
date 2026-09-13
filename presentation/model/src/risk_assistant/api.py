"""Optional FastAPI boundary for a web or chat client."""
from __future__ import annotations

import os

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError("Install fastapi and uvicorn to run the API") from exc

from .service import RiskAssistant
from .inference import RegionRiskPredictor
from .graph import GraphRepository


app = FastAPI(title="BC Region-Industry Risk Assistant", version="0.1.0")
require_neo4j = os.getenv("REQUIRE_NEO4J", "false").lower() == "true"
assistant = RiskAssistant(graph=GraphRepository(require_neo4j=require_neo4j))
predictor = RegionRiskPredictor()


class DiagnosisRequest(BaseModel):
    profile: dict = Field(description="Risk model output plus region and industry")
    business_answers: dict = Field(default_factory=dict)


class RegionDiagnosisRequest(BaseModel):
    sido: str
    sigungu: str
    industry: str
    business_answers: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "ok", "llm_available": assistant.llm.available(),
            "embedding_available": assistant.embedder.available(),
            "neo4j_available": assistant.graph.verify_connectivity()}


@app.post("/diagnose")
def diagnose(request: DiagnosisRequest):
    return assistant.diagnose(request.profile, request.business_answers)


@app.post("/diagnose-region")
def diagnose_region(request: RegionDiagnosisRequest):
    profile = predictor.predict(request.sido, request.sigungu, request.industry)
    return assistant.diagnose(profile, request.business_answers)
