# Portfolio Risk Agent Demo

A minimal full-stack demo for agentic portfolio analytics with dummy data.

## What it includes

- Dummy portfolio data in `data/positions.csv`
- Dummy historical returns in `data/returns.csv`
- Deterministic Python risk tools in `app/risk_tools.py`
- OpenAI Agents SDK backend in `app/agent_backend.py`
- Streamlit UI in `app/streamlit_app.py`
- Optional FastAPI endpoint in `app/api.py`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
```

## Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Run API

```bash
uvicorn app.api:app --reload --port 8000
```

Then POST:

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarize portfolio risk and run a stress test"}'
```

## MCP integration path

For MVP, the Python functions act like local tools. In production, wrap each system behind MCP servers:

- portfolio-db-mcp: positions, trades, books, NAV
- market-data-mcp: prices, curves, vol surfaces, credit spreads
- risk-engine-mcp: VaR, stress, Greeks, liquidity
- docs-mcp: mandates, IC memos, research, compliance docs
- reporting-mcp: PDF, Excel, PowerPoint generation

The agent remains the orchestrator; MCP servers become permissioned tool boundaries.
