"""
tools/financial_data.py

Fetches a financial snapshot for a ticker using yfinance.
Returns a FinancialData model, or raises on failure.
"""

import os

import yfinance as yf
from edgar import Company, set_identity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

from workflows.state import FinancialData

# Tell SEC EDGAR who we are — required User-Agent string
set_identity(os.getenv("EDGAR_IDENTITY", "ai-investments research@example.com"))


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


def fetch_edgar_filing(ticker: str, max_chars: int = 8000) -> dict | None:
    """
    Original fetch_edgar_filing — kept as fallback.
    Fetches the most recent 10-K or 10-Q and truncates to max_chars.
    Returns None on failure.
    """
    try:
        company = Company(ticker)
        filings = company.get_filings(form=["10-K", "10-Q"]).latest(1)

        if not filings:
            return None

        latest = filings[0] if hasattr(filings, "__getitem__") else filings

        text = latest.text()
        return {
            "form": latest.form,
            "filing_date": str(latest.filing_date),
            "accession_no": latest.accession_no,
            "text": text[:max_chars],
        }
    except Exception:
        return None


QUERY_STRINGS = [
    "revenue growth and financial performance",
    "risk factors and business risks",
    "management discussion and outlook",
    "operating expenses and profit margins",
    "guidance and forward looking statements",
]


def fetch_edgar_filing_rag(ticker: str, top_k: int = 5) -> dict | None:
    """
    Fetches the most recent 10-K or 10-Q for a ticker and uses RAG to retrieve
    the most investment-relevant passages instead of raw truncated text.

    Pipeline:
      1. Fetch full filing text (no truncation)
      2. Chunk with RecursiveCharacterTextSplitter
      3. Embed chunks locally with all-MiniLM-L6-v2 (no API calls, no rate limits)
      4. Store in an ephemeral ChromaDB in-memory collection
      5. Query with fixed investment-relevant questions
      6. Deduplicate and return retrieved chunks

    Returns a dict with keys: form, filing_date, accession_no, text
    Returns None on any failure (non-fatal — pipeline continues without it).
    """
    try:
        # ── 1. Fetch full filing ──────────────────────────────────────────────
        company = Company(ticker)
        filings = company.get_filings(form=["10-K", "10-Q"]).latest(1)

        if not filings:
            return None

        latest = filings[0] if hasattr(filings, "__getitem__") else filings
        full_text = latest.text()

        if not full_text or not full_text.strip():
            return None

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(full_text)

        if not chunks:
            return None

        # ── 3 & 4. Embed + store in ephemeral ChromaDB ────────────────────────
        # Runs locally — no API calls, no rate limits
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        # EphemeralClient keeps the index purely in-memory — discarded after run
        chroma_client = chromadb.EphemeralClient()
        collection_name = f"edgar_{ticker.lower()}"

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            client=chroma_client,
            collection_name=collection_name,
        )

        # ── 5. Query with investment-relevant questions ────────────────────────
        seen = set()
        retrieved_chunks = []

        for query in QUERY_STRINGS:
            results = vectorstore.similarity_search(query, k=1)
            for doc in results:
                # Deduplicate by exact text
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    retrieved_chunks.append(doc.page_content)

        if not retrieved_chunks:
            return None

        # ── 6. Join and return in the same shape as fetch_edgar_filing ─────────
        combined_text = "\n\n---\n\n".join(retrieved_chunks)

        return {
            "form": latest.form,
            "filing_date": str(latest.filing_date),
            "accession_no": latest.accession_no,
            "text": combined_text,
        }

    except Exception:
        return None


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