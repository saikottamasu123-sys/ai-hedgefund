# AI Investment Committee

A multi-agent LLM system that simulates a professional investment committee. Given any stock ticker, it fetches real market data, pulls the latest SEC filing, and runs a structured two-round debate between analyst personas before issuing a final BUY / HOLD / SELL recommendation.

---

## How It Works

The pipeline runs as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine with 8 nodes:

```
                    ┌─────────────────────┐
                    │  Research           │
                    │  Coordinator        │  ← yfinance + SEC EDGAR
                    └────────┬────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    [Bull Analyst]    [Bear Analyst]    [Risk Manager]    ← Round 1 (parallel)
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
   [Bull Rebuttal]  [Bear Rebuttal]  [Risk Rebuttal]     ← Round 2 (parallel)
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────────┐
                    │  Portfolio Manager  │  ← Final report
                    └─────────────────────┘
```

**Round 1 — Initial positions:** Bull, Bear, and Risk analysts each receive the research brief and financial snapshot and independently produce a structured opinion (recommendation, confidence score, reasoning, key points).

**Round 2 — Rebuttals:** Each analyst reads the other two's Round 1 arguments and must respond directly — conceding where warranted, pushing back where not.

**Final verdict:** The Portfolio Manager weighs all six opinions across both rounds and writes a full markdown investment committee report with a final recommendation and confidence score.

---

## Project Structure

```
ai-investment-committee/
├── agents/
│   └── committee_agents.py     # All 8 LangGraph node functions
├── workflows/
│   ├── state.py                # CommitteeState TypedDict + Pydantic models
│   └── committee_graph.py      # LangGraph StateGraph definition + run_committee()
├── tools/
│   └── financial_data.py       # yfinance fetching, SEC EDGAR RAG pipeline, prompt formatting
├── prompts/
│   └── agent_prompts.py        # All system + user prompt templates
├── api/
│   └── main.py                 # FastAPI app (POST /analyze, GET /health)
├── run.py                      # CLI entry point
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Add your key to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Optionally set your SEC EDGAR identity (required by EDGAR's terms of use):

```
EDGAR_IDENTITY=Your Name your@email.com
```

### 3. Run from the CLI

```bash
python run.py NVDA
python run.py AAPL
python run.py MSFT
```

The full pipeline takes roughly 20–40 seconds. The final markdown report is printed to stdout, along with any non-fatal errors (e.g. if an EDGAR filing was unavailable).

### 4. Or start the API server

```bash
uvicorn api.main:app --reload
```

Then POST to `http://localhost:8000/analyze`:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'
```

Interactive API docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /analyze`

Run the full committee pipeline for a ticker.

**Request body:**
```json
{ "ticker": "NVDA" }
```

**Response:**
```json
{
  "ticker": "NVDA",
  "final_recommendation": "BUY",
  "final_confidence": 0.78,
  "bull_opinion": { "role": "bull", "recommendation": "BUY", "confidence": 0.85, "reasoning": "...", "key_points": ["..."] },
  "bear_opinion": { ... },
  "risk_opinion": { ... },
  "final_report": "# Investment Committee Report: NVDA\n...",
  "errors": []
}
```

**Note:** This is a synchronous endpoint. For production use, consider wrapping it in a job queue (Celery, ARQ) with a polling endpoint.

### `GET /health`

```json
{ "status": "ok" }
```

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| [yfinance](https://github.com/ranaroussi/yfinance) | Current price, market cap, P/E ratio, revenue growth, profit margins, debt/equity, 52-week range, business summary |
| [SEC EDGAR](https://www.sec.gov/edgar) | Latest 10-K or 10-Q filing, retrieved via RAG to extract the most investment-relevant passages |

The EDGAR pipeline chunks the full filing text, embeds with **Google `text-embedding-004`**, stores in an ephemeral ChromaDB instance, then queries with five fixed investment questions (revenue growth, risk factors, management outlook, margins, forward guidance) to retrieve the most signal-dense passages.

---

## Embedding Model

The project supports two embedding backends for the EDGAR RAG pipeline. Swap by toggling comments in `tools/financial_data.py`:

```python
# OPTION A: HuggingFace — runs locally, no API key required
# from langchain_huggingface import HuggingFaceEmbeddings

# OPTION B: Google text-embedding-004 — free tier, requires GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

The same toggle exists where the embeddings are instantiated inside `fetch_edgar_filing_rag()`.

---

## Models Used

| Component | Model |
|-----------|-------|
| All analyst agents | `gemini-2.5-flash-lite` |
| Embeddings (EDGAR RAG) | `models/text-embedding-004` |

---

## Requirements

```
langchain>=0.3.0
langchain-google-genai>=2.0.0
langchain-text-splitters>=0.3.0
langchain-community>=0.3.0
langchain-huggingface>=0.1.0
langgraph>=0.2.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
yfinance>=0.2.40
edgartools>=2.0.0
python-dotenv>=1.0.0
chromadb>=0.6.0
sentence-transformers>=3.0.0
```

## Next Up

- [ ] Persist committee results to PostgreSQL (store reports, recommendations, and confidence scores over time)
- [ ] Build a Next.js frontend dashboard to visualize reports and track sentiment across multiple tickers
