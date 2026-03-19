# Claims Copilot

An AI-assisted claims operations prototype for motor insurance. The repo is organized like a small AI product: synthetic data generation, LLM extraction, ML-based claim scoring, a rules-based next-best-action engine, and a Streamlit handler UI.

## Current Status

Implemented in this repo:

1. Synthetic relational claims data generation
2. Local LLM extraction with Ollama + structured schemas
3. Feature engineering for complexity, handling-time, and fraud models
4. XGBoost training pipeline with time-based splits
5. Rules-based next-best-action engine
6. Streamlit UI that ties the layers together

Not implemented yet:

1. FastAPI service boundary
2. Persistent application database integration in the running app
3. Production deployment, monitoring, auth, and feedback ingestion

## Code Map

- [`src/data/generator.py`](src/data/generator.py): synthetic policyholders, policies, vehicles, claims, labels, documents, and events
- [`src/models/features.py`](src/models/features.py): time-correct feature engineering
- [`src/models/training.py`](src/models/training.py): model training, evaluation, artifact loading, and prediction wrapper
- [`src/extraction/pipeline.py`](src/extraction/pipeline.py): Ollama-based JSON extraction pipeline
- [`src/serving/next_best_action.py`](src/serving/next_best_action.py): deterministic action/routing engine
- [`src/ui/app.py`](src/ui/app.py): Streamlit handler UI
- [`scripts/`](scripts): entrypoints for generation, extraction, and training
- [`tests/`](tests): lightweight regression coverage

A repo-specific study guide lives in [`docs/codebase-tour.md`](docs/codebase-tour.md).

## Quickstart

```bash
# Install all dev dependencies
uv sync --extra dev

# Generate synthetic data
uv run python scripts/generate_claims.py -n 1000

# Run tests
uv run pytest -q

# Train models
uv run python scripts/train_models.py --csv-dir data --model-dir models

# Start the UI
uv run streamlit run src/ui/app.py
```

## LLM Extraction

The extraction layer currently uses a local Ollama model, not a hosted API.

```bash
# Optional: run extraction on a sample
uv run python scripts/run_extraction.py --n-sample 20 --eval
```

Recommended local models are documented in [`src/extraction/pipeline.py`](src/extraction/pipeline.py).

## Learning Goal

If you want to learn this codebase well enough to reproduce similar systems, study it in this order:

1. Data generation
2. Feature engineering
3. Training and artifact flow
4. LLM extraction
5. Rules engine
6. UI composition

That order mirrors how the product is assembled and makes the AI-specific design choices easier to reason about.

ollama serve
  ollama pull mistral:7b-instruct-q4_K_M
  uv run python scripts/run_extraction.py --n-sample 50 --eval
  uv run python scripts/train_models.py --csv-dir data --model-dir models --with-extractions data/extractions/extractions.json
  uv run streamlit run src/ui/app.py