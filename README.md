# Merchant Growth AI

AI-powered business growth advisor for small merchants in India. Combines a regression model, growth roadmap, and LLM reasoning to generate actionable, data-driven recommendations.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend   │────▶│   FastAPI Backend │────▶│  LLM Providers  │
│  (Streamlit) │     │                  │     │  Gemini / Groq   │
└──────────────┘     └──────────────────┘     └─────────────────┘
                            │                         │
                     ┌──────┴──────┐            ┌─────┴─────┐
                     │  Regression │            │  Roadmap   │
                     │   Engine    │            │  Retriever │
                     └─────────────┘            └───────────┘
```

## Components

| Module | Description |
|---|---|
| `backend/app.py` | FastAPI REST API with merchant CRUD and recommendation endpoint |
| `backend/llm_orchestrator.py` | Provider-agnostic LLM wrapper (Gemini primary, Groq fallback) |
| `backend/model_loader.py` | Loads pre-trained LinearRegression model for revenue growth prediction |
| `backend/merchant_store.py` | SQLite-backed merchant profile store |
| `backend/roadmap_retriever.py` | Matches regression signals to growth roadmap stages |
| `frontend/app.py` | Streamlit demo UI |
| `scripts/train_regression.py` | Trains the regression model on synthetic merchant data |
| `scripts/seed_merchants.py` | Seeds demo merchant profiles into SQLite |
| `roadmap/growth_roadmap.json` | 5-stage growth playbook with strategies |
| `models/model_metadata.json` | Model metrics and feature importance |

## Growth Roadmap Stages

1. **Pricing & Transaction Value Optimization** — value bundles, tiered pricing, markdowns
2. **Customer Retention & Loyalty** — punch cards, follow-ups, VIP service
3. **Local Marketing & Footfall Growth** — signage, micro-promotions, cross-promotions
4. **Digital Payments & Tech Adoption** — UPI incentives, POS setup, contact collection
5. **Inventory & Operations Efficiency** — ABC analysis, supplier terms, daily reviews

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Seed merchant data
python scripts/seed_merchants.py

# Train regression model
python scripts/train_regression.py

# Run the API server
uvicorn backend.app:app --reload

# Run the Streamlit frontend
streamlit run frontend/app.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (primary LLM) |
| `GROQ_API_KEY` | Groq API key (fallback LLM) |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Component status |
| `GET` | `/merchants` | List all merchant personas |
| `GET` | `/merchants/{id}/profile` | Get full merchant profile |
| `POST` | `/merchants/{id}/recommend` | Generate growth recommendation |

## Model Performance

- **Algorithm:** LinearRegression with preprocessing pipeline
- **R² Score:** 0.8513
- **MAE:** 2.43
- **RMSE:** 2.94
- **Training samples:** 1,000

## Performance

Heavy imports (sklearn, pandas, LLM SDKs) are lazy-loaded — only imported when first needed. This keeps initial page load under 300ms. The regression model, merchant store, and roadmap retriever are cached via `@st.cache_resource` so subsequent loads are instant.

## Tech Stack

- **Backend:** FastAPI, Python
- **ML:** scikit-learn, pandas, joblib
- **LLMs:** Google Gemini 3.6 Flash, Groq Compound Mini
- **Frontend:** Streamlit, Plotly
- **Database:** SQLite
