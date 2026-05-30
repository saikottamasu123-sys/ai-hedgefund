# AI Investment Committee

A multi-agent LLM system that simulates a professional investment committee.

## Architecture

```
User
 │
 ▼
Research Coordinator  ←  yfinance (real market data)
 │
 ├──────────────────────────────┐
 ▼            ▼                 ▼
Bull Analyst  Bear Analyst  Risk Manager   ← run in parallel
 │            │                 │
 └────────────┴─────────────────┘
                   │
                   ▼
          Portfolio Manager
                   │
                   ▼
            Final Report
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run from the command line

```bash
python run.py NVDA
python run.py AAPL
python run.py MSFT
```

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

Interactive docs: http://localhost:8000/docs

---

## Project Structure

```
ai-investment-committee/
├── agents/
│   └── committee_agents.py   # All 5 agent node functions
├── workflows/
│   ├── state.py               # CommitteeState TypedDict + Pydantic models
│   └── committee_graph.py     # LangGraph StateGraph definition
├── tools/
│   └── financial_data.py      # yfinance data fetching + formatting
├── prompts/
│   └── agent_prompts.py       # All system + user prompt templates
├── api/
│   └── main.py                # FastAPI application
├── run.py                     # CLI entry point
├── requirements.txt
└── .env.example
```

---

## Phase 2 Roadmap

- [ ] Add SEC EDGAR document retrieval
- [ ] Add earnings call transcript ingestion
- [ ] Implement RAG with Chroma vector store
- [ ] Add structured debate / rebuttal round between analysts
- [ ] Persist results to PostgreSQL
- [ ] Add memory layer (track sentiment over time)
- [ ] Human-in-the-loop review step in the graph
- [ ] Next.js frontend dashboard
