# Claims Copilot

An AI-powered claims operations assistant for motor insurance, built as a portfolio project demonstrating end-to-end ML system design.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Handler UI (Streamlit)                │
│  Summary │ Facts │ Complexity │ Fraud │ Next Actions     │
└──────────┬──────────────────────────────────┬───────────┘
           │                                  │ feedback
     ┌─────▼─────┐                      ┌─────▼─────┐
     │  FastAPI   │◄────────────────────►│ PostgreSQL │
     │  serving   │                      │  + events  │
     └─────┬──────┘                      └───────────┘
           │
   ┌───────┼────────┬──────────────┐
   ▼       ▼        ▼              ▼
 LLM    Routing   Fraud      Next-Best
Extract  Model    Signal      Action
(Claude) (GBDT)  (GBDT+AD)  (Rules+ML)
```

## Layers

1. Synthetic data generation: In progress
2. LLM extraction and summarization: Planned
3. Complexity and routing model: Planned
4. Fraud and anomaly signal: Planned
5. Next best action engine: Planned
6. Handler UI: Planned

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up project (creates venv + installs everything)
uv sync

# Set up PostgreSQL
createdb claims_copilot
psql claims_copilot < sql/schema.sql

# Generate synthetic data
uv run python scripts/generate_claims.py -n 5000
```

## Tech Stack

- Python 3.10+
- PostgreSQL 14+
- scikit-learn / XGBoost
- Anthropic Claude API (extraction layer)
- FastAPI (serving)
- Streamlit (UI)
