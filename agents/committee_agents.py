"""
agents/committee_agents.py

Each function is a LangGraph node.
Nodes receive the full CommitteeState and return a dict of keys to update.
"""

import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from workflows.state import CommitteeState, AnalystOpinion, FinancialData
from tools.financial_data import fetch_financial_data, format_financial_context
from prompts.agent_prompts import (
    ANALYST_OUTPUT_INSTRUCTIONS,
    COORDINATOR_SYSTEM,
    COORDINATOR_USER,
    BULL_SYSTEM, BULL_USER,
    BEAR_SYSTEM, BEAR_USER,
    RISK_SYSTEM, RISK_USER,
    PORTFOLIO_MANAGER_SYSTEM,
    PORTFOLIO_MANAGER_USER,
)


# ── Shared LLM instance ───────────────────────────────────────────────────────

def get_llm(temperature: float = 0.3):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _call_llm(system: str, user: str, temperature: float = 0.3) -> str:
    """Helper: send a system + user message and return the text response."""
    llm = get_llm(temperature)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    response = llm.invoke(messages)
    return response.content


def _parse_analyst_json(raw: str, role: str) -> AnalystOpinion:
    """
    Parse JSON output from an analyst agent into an AnalystOpinion model.
    Strips markdown fences if the model accidentally adds them.
    """
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    return AnalystOpinion(role=role, **data)


# ── Node 1: Research Coordinator ─────────────────────────────────────────────

def research_coordinator(state: CommitteeState) -> dict:
    """
    Fetches financial data and writes a plain-text research brief.
    Populates: financial_data, research_summary
    """
    ticker = state["ticker"]
    errors = list(state.get("errors", []))

    print(f"[Coordinator] Fetching data for {ticker}...")

    try:
        financial_data = fetch_financial_data(ticker)
    except Exception as e:
        errors.append(f"Financial data fetch failed: {e}")
        # Create a minimal placeholder so downstream agents can still run
        financial_data = FinancialData(
            ticker=ticker,
            company_name=ticker,
            summary="Financial data unavailable — proceeding with limited information.",
        )

    financial_context = format_financial_context(financial_data)

    user_prompt = COORDINATOR_USER.format(financial_context=financial_context)
    research_summary = _call_llm(COORDINATOR_SYSTEM, user_prompt, temperature=0.2)

    print(f"[Coordinator] Research brief ready ({len(research_summary)} chars)")

    return {
        "financial_data": financial_data,
        "research_summary": research_summary,
        "errors": errors,
    }


# ── Node 2: Bull Analyst ──────────────────────────────────────────────────────

def bull_analyst(state: CommitteeState) -> dict:
    """
    Constructs the strongest investment case.
    Populates: bull_opinion
    """
    print("[Bull Analyst] Building bull case...")

    financial_context = format_financial_context(state["financial_data"])
    user_prompt = BULL_USER.format(
        research_summary=state["research_summary"],
        financial_context=financial_context,
        output_instructions=ANALYST_OUTPUT_INSTRUCTIONS,
    )

    raw = _call_llm(BULL_SYSTEM, user_prompt)
    opinion = _parse_analyst_json(raw, role="bull")

    print(f"[Bull Analyst] {opinion.recommendation} ({opinion.confidence:.0%})")
    return {"bull_opinion": opinion}


# ── Node 3: Bear Analyst ──────────────────────────────────────────────────────

def bear_analyst(state: CommitteeState) -> dict:
    """
    Challenges optimistic assumptions.
    Populates: bear_opinion
    """
    print("[Bear Analyst] Building bear case...")

    financial_context = format_financial_context(state["financial_data"])
    user_prompt = BEAR_USER.format(
        research_summary=state["research_summary"],
        financial_context=financial_context,
        output_instructions=ANALYST_OUTPUT_INSTRUCTIONS,
    )

    raw = _call_llm(BEAR_SYSTEM, user_prompt)
    opinion = _parse_analyst_json(raw, role="bear")

    print(f"[Bear Analyst] {opinion.recommendation} ({opinion.confidence:.0%})")
    return {"bear_opinion": opinion}


# ── Node 4: Risk Manager ──────────────────────────────────────────────────────

def risk_manager(state: CommitteeState) -> dict:
    """
    Identifies portfolio and operational risks.
    Populates: risk_opinion
    """
    print("[Risk Manager] Assessing risks...")

    financial_context = format_financial_context(state["financial_data"])
    user_prompt = RISK_USER.format(
        research_summary=state["research_summary"],
        financial_context=financial_context,
        output_instructions=ANALYST_OUTPUT_INSTRUCTIONS,
    )

    raw = _call_llm(RISK_SYSTEM, user_prompt)
    opinion = _parse_analyst_json(raw, role="risk")

    print(f"[Risk Manager] {opinion.recommendation} ({opinion.confidence:.0%})")
    return {"risk_opinion": opinion}


# ── Node 5: Portfolio Manager ─────────────────────────────────────────────────

def portfolio_manager(state: CommitteeState) -> dict:
    """
    Synthesises all analyst opinions into a final recommendation and report.
    Populates: final_recommendation, final_confidence, final_report
    """
    print("[Portfolio Manager] Deliberating...")

    bull = state["bull_opinion"]
    bear = state["bear_opinion"]
    risk = state["risk_opinion"]

    def fmt_points(points: list[str]) -> str:
        return "\n".join(f"  • {p}" for p in points)

    financial_context = format_financial_context(state["financial_data"])

    user_prompt = PORTFOLIO_MANAGER_USER.format(
        ticker=state["ticker"],
        bull_rec=bull.recommendation,
        bull_conf=bull.confidence,
        bull_reasoning=bull.reasoning,
        bull_points=fmt_points(bull.key_points),
        bear_rec=bear.recommendation,
        bear_conf=bear.confidence,
        bear_reasoning=bear.reasoning,
        bear_points=fmt_points(bear.key_points),
        risk_rec=risk.recommendation,
        risk_conf=risk.confidence,
        risk_reasoning=risk.reasoning,
        risk_points=fmt_points(risk.key_points),
        financial_context=financial_context,
    )

    report = _call_llm(PORTFOLIO_MANAGER_SYSTEM, user_prompt, temperature=0.2)

    # Extract the final recommendation and confidence from the report
    final_rec = "HOLD"
    for rec in ("BUY", "SELL", "HOLD"):
        if f"**Recommendation: {rec}**" in report:
            final_rec = rec
            break

    # Simple confidence extraction — look for "Confidence: XX%"
    final_conf = 0.70
    import re
    match = re.search(r"\*\*Confidence:\s*(\d+)%\*\*", report)
    if match:
        final_conf = int(match.group(1)) / 100

    print(f"[Portfolio Manager] Final: {final_rec} ({final_conf:.0%})")
    return {
        "final_recommendation": final_rec,
        "final_confidence": final_conf,
        "final_report": report,
    }
