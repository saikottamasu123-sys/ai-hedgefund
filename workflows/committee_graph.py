"""
workflows/committee_graph.py

Assembles the LangGraph StateGraph for the investment committee.

Graph shape:
  coordinator → [bull, bear, risk in parallel] → [bull_rebuttal, bear_rebuttal, risk_rebuttal in parallel] → portfolio_manager → END
"""

from langgraph.graph import StateGraph, END

from workflows.state import CommitteeState
from agents.committee_agents import (
    research_coordinator,
    bull_analyst,
    bear_analyst,
    risk_manager,
    bull_rebuttal,
    bear_rebuttal,
    risk_rebuttal,
    portfolio_manager,
)


def build_committee_graph() -> StateGraph:
    """
    Constructs and compiles the committee workflow graph.

    Parallelism note:
    LangGraph executes nodes with no dependency on each other in parallel
    automatically when you add edges from one node to multiple nodes.
    Round 1: bull_analyst, bear_analyst, and risk_manager run concurrently.
    Round 2: bull_rebuttal, bear_rebuttal, risk_rebuttal run concurrently
             after all three Round 1 nodes have completed (fan-in enforces this).
    """

    graph = StateGraph(CommitteeState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("coordinator",       research_coordinator)
    graph.add_node("bull_analyst",      bull_analyst)
    graph.add_node("bear_analyst",      bear_analyst)
    graph.add_node("risk_manager",      risk_manager)
    graph.add_node("bull_rebuttal",     bull_rebuttal)
    graph.add_node("bear_rebuttal",     bear_rebuttal)
    graph.add_node("risk_rebuttal",     risk_rebuttal)
    graph.add_node("portfolio_manager", portfolio_manager)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("coordinator")

    # ── coordinator → three analysts (fan-out / parallel) ────────────────────
    graph.add_edge("coordinator", "bull_analyst")
    graph.add_edge("coordinator", "bear_analyst")
    graph.add_edge("coordinator", "risk_manager")

    # ── three analysts → three rebuttals (fan-in then fan-out / parallel) ────
    # Each rebuttal node needs all three Round 1 opinions, so all three
    # analysts must complete before any rebuttal starts. LangGraph's fan-in
    # ensures this: a node only fires once all its incoming edges are satisfied.
    graph.add_edge("bull_analyst", "bull_rebuttal")
    graph.add_edge("bear_analyst", "bull_rebuttal")
    graph.add_edge("risk_manager", "bull_rebuttal")

    graph.add_edge("bull_analyst", "bear_rebuttal")
    graph.add_edge("bear_analyst", "bear_rebuttal")
    graph.add_edge("risk_manager", "bear_rebuttal")

    graph.add_edge("bull_analyst", "risk_rebuttal")
    graph.add_edge("bear_analyst", "risk_rebuttal")
    graph.add_edge("risk_manager", "risk_rebuttal")

    # ── three rebuttals → portfolio manager (fan-in) ──────────────────────────
    graph.add_edge("bull_rebuttal",  "portfolio_manager")
    graph.add_edge("bear_rebuttal",  "portfolio_manager")
    graph.add_edge("risk_rebuttal",  "portfolio_manager")

    # ── portfolio manager → end ───────────────────────────────────────────────
    graph.add_edge("portfolio_manager", END)

    return graph.compile()


# ── Convenience run function ─────────────────────────────────────────────────

def run_committee(ticker: str) -> CommitteeState:
    """
    Run the full investment committee pipeline for a given ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA", "AAPL", "MSFT"

    Returns:
        Final CommitteeState with all agent outputs and the final report.
    """
    app = build_committee_graph()

    initial_state: CommitteeState = {
        "ticker": ticker.upper(),
        "financial_data": None,
        "research_summary": "",
        "edgar_filing": None,
        "bull_opinion": None,
        "bear_opinion": None,
        "risk_opinion": None,
        "bull_rebuttal": None,
        "bear_rebuttal": None,
        "risk_rebuttal": None,
        "final_recommendation": "",
        "final_confidence": 0.0,
        "final_report": "",
        "errors": [],
    }

    print(f"\n{'='*60}")
    print(f"  AI Investment Committee — Analyzing {ticker.upper()}")
    print(f"{'='*60}\n")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"  FINAL: {final_state['final_recommendation']} "
          f"({final_state['final_confidence']:.0%} confidence)")
    print(f"{'='*60}\n")

    return final_state