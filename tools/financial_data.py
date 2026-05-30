"""
tools/financial_data.py

Fetches a financial snapshot for a ticker using yfinance.
Returns a FinancialData model, or raises on failure.
"""

import yfinance as yf
from workflows.state import FinancialData


def fetch_financial_data(ticker: str) -> FinancialData:
    """
    Pull key financials from Yahoo Finance.
    Returns a FinancialData pydantic model.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    def safe_get(key: str, default=None):
        val = info.get(key, default)
        # yfinance sometimes returns "Infinity" or 0 for missing data
        if val in (float("inf"), float("-inf")):
            return None
        return val

    return FinancialData(
        ticker=ticker.upper(),
        company_name=safe_get("longName", ticker),
        current_price=safe_get("currentPrice") or safe_get("regularMarketPrice"),
        market_cap=safe_get("marketCap"),
        pe_ratio=safe_get("trailingPE"),
        revenue_growth=safe_get("revenueGrowth"),   # decimal — multiply by 100 for %
        profit_margin=safe_get("profitMargins"),     # decimal
        debt_to_equity=safe_get("debtToEquity"),
        fifty_two_week_high=safe_get("fiftyTwoWeekHigh"),
        fifty_two_week_low=safe_get("fiftyTwoWeekLow"),
        summary=safe_get("longBusinessSummary", "No business summary available."),
    )


def format_financial_context(data: FinancialData) -> str:
    """
    Render FinancialData as a compact text block to inject into agent prompts.
    Agents receive this instead of the raw JSON.
    """

    def fmt_pct(val):
        if val is None:
            return "N/A"
        return f"{val * 100:.1f}%"

    def fmt_price(val):
        if val is None:
            return "N/A"
        return f"${val:,.2f}"

    def fmt_large(val):
        if val is None:
            return "N/A"
        if val >= 1e12:
            return f"${val / 1e12:.2f}T"
        if val >= 1e9:
            return f"${val / 1e9:.2f}B"
        return f"${val:,.0f}"

    return f"""
COMPANY: {data.company_name} ({data.ticker})

FINANCIALS
  Current Price:    {fmt_price(data.current_price)}
  Market Cap:       {fmt_large(data.market_cap)}
  P/E Ratio:        {data.pe_ratio if data.pe_ratio else 'N/A'}
  Revenue Growth:   {fmt_pct(data.revenue_growth)}
  Profit Margin:    {fmt_pct(data.profit_margin)}
  Debt/Equity:      {data.debt_to_equity if data.debt_to_equity else 'N/A'}
  52-Week High:     {fmt_price(data.fifty_two_week_high)}
  52-Week Low:      {fmt_price(data.fifty_two_week_low)}

BUSINESS OVERVIEW
{data.summary[:600]}{'...' if len(data.summary) > 600 else ''}
""".strip()
