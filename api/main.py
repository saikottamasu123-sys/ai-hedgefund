"""
api/main.py

FastAPI application that exposes the investment committee as a REST API.

Endpoints:
  POST /analyze          — run the full committee pipeline
  GET  /health           — service health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # loads a gemini API key from env file

from workflows.committee_graph import run_committee
from workflows.state import AnalystOpinion

app = FastAPI(
    title="AI Investment Committee",
    description="Multi-agent LLM system that simulates a professional investment committee.",
    version="0.1.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., example="NVDA", description="Stock ticker symbol")


class AnalystOpinionResponse(BaseModel):
    role: str
    recommendation: str
    confidence: float
    reasoning: str
    key_points: list[str]


class AnalyzeResponse(BaseModel):
    ticker: str
    final_recommendation: str
    final_confidence: float
    bull_opinion: AnalystOpinionResponse
    bear_opinion: AnalystOpinionResponse
    risk_opinion: AnalystOpinionResponse
    final_report: str
    errors: list[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Run the full investment committee pipeline for the given ticker.

    This is a synchronous endpoint — the pipeline typically takes 20-40 seconds
    due to multiple LLM calls. For production, consider making this async
    with a job queue (Celery, ARQ) and a polling endpoint.
    """
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=422, detail="Ticker cannot be empty")

    try:
        state = run_committee(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def opinion_to_response(op: AnalystOpinion) -> AnalystOpinionResponse:
        return AnalystOpinionResponse(
            role=op.role,
            recommendation=op.recommendation,
            confidence=op.confidence,
            reasoning=op.reasoning,
            key_points=op.key_points,
        )

    return AnalyzeResponse(
        ticker=ticker,
        final_recommendation=state["final_recommendation"],
        final_confidence=state["final_confidence"],
        bull_opinion=opinion_to_response(state["bull_opinion"]),
        bear_opinion=opinion_to_response(state["bear_opinion"]),
        risk_opinion=opinion_to_response(state["risk_opinion"]),
        final_report=state["final_report"],
        errors=state.get("errors", []),
    )
