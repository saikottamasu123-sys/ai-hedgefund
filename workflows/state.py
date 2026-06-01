"""
workflows/state.py

Defines the shared state that flows through the LangGraph pipeline.
Every agent reads from and writes to this state object.
"""

from typing import TypedDict, Optional
from pydantic import BaseModel


# ── Per-agent output ──────────────────────────────────────────────────────────

class AnalystOpinion(BaseModel):
    """Structured output produced by each analyst agent."""

    role: str                   # "bull" | "bear" | "risk"
    recommendation: str         # "BUY" | "HOLD" | "SELL"
    confidence: float           # 0.0 – 1.0
    reasoning: str              # free-text rationale
    key_points: list[str]       # 3-5 bullet points


# ── Financial snapshot passed to agents ──────────────────────────────────────

class FinancialData(BaseModel):
    ticker: str
    company_name: str
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None   # YoY %
    profit_margin: Optional[float] = None    # %
    debt_to_equity: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    summary: str = ""                        # business description


# ── LangGraph state ───────────────────────────────────────────────────────────

class CommitteeState(TypedDict):
    """
    The single state object passed between all nodes in the graph.

    LangGraph merges return dicts into this state — each node only
    needs to return the keys it modifies.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    ticker: str                         # e.g. "NVDA"

    # ── Research Coordinator output ────────────────────────────────────────
    financial_data: Optional[FinancialData]
    research_summary: str               # plain-text context passed to agents
    edgar_filing: Optional[dict]        # { form, filing_date, accession_no, text }

    # ── Analyst agent outputs ──────────────────────────────────────────────
    bull_opinion: Optional[AnalystOpinion]
    bear_opinion: Optional[AnalystOpinion]
    risk_opinion: Optional[AnalystOpinion]

    # ── Rebuttal round outputs ─────────────────────────────────────────────
    bull_rebuttal: Optional[AnalystOpinion]
    bear_rebuttal: Optional[AnalystOpinion]
    risk_rebuttal: Optional[AnalystOpinion]

    # ── Portfolio Manager output ───────────────────────────────────────────
    final_recommendation: str           # "BUY" | "HOLD" | "SELL"
    final_confidence: float
    final_report: str                   # full markdown report

    # ── Metadata ───────────────────────────────────────────────────────────
    errors: list[str]                   # non-fatal errors collected during run