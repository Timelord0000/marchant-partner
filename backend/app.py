"""
FastAPI Backend — ties together regression engine, merchant store,
roadmap retriever, and LLM orchestrator.
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load env vars from .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from backend.model_loader import RegressionEngine
from backend.merchant_store import MerchantStore
from backend.roadmap_retriever import RoadmapRetriever
from backend.llm_orchestrator import generate_recommendation

# ---------------------------------------------------------------------------
# Init components
# ---------------------------------------------------------------------------
app = FastAPI(title="Merchant Growth AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

regression_engine = RegressionEngine()
merchant_store = MerchantStore()
roadmap_retriever = RoadmapRetriever()

# Load model at startup
regression_engine.load()
roadmap_retriever.load()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class MerchantSummary(BaseModel):
    merchant_id: str
    business_name: str
    business_type: str
    sales_trend: str


class RecommendationResponse(BaseModel):
    merchant_id: str
    business_name: str
    recommendation: str
    cited_signal: str
    cited_roadmap_stage: str
    explanation: str
    regression_signal: dict[str, Any]
    provider: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Merchant Growth AI", "version": "0.1.0"}


@app.get("/merchants", response_model=list[MerchantSummary])
def list_merchants():
    """List all demo merchant personas."""
    return merchant_store.list_merchants()


@app.get("/merchants/{merchant_id}/profile")
def get_merchant_profile(merchant_id: str):
    """Get full profile for a merchant."""
    profile = merchant_store.get_profile(merchant_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found")
    return profile


@app.post("/merchants/{merchant_id}/recommend", response_model=RecommendationResponse)
def recommend(merchant_id: str):
    """
    Full pipeline: load profile → run regression → retrieve roadmap → call LLM → return.
    """
    # 1. Load merchant profile
    profile = merchant_store.get_profile(merchant_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Merchant '{merchant_id}' not found")

    # 2. Run regression
    features = merchant_store.get_regression_features(merchant_id)
    regression_signal = regression_engine.predict(features)

    # 3. Retrieve roadmap stages
    # Use profile keywords to help retrieval
    keywords = []
    if profile["sales_trend"] == "declining":
        keywords.extend(["retention", "customers", "churn"])
    if profile["digital_payment_adoption"] < 50:
        keywords.extend(["digital", "payment", "upi"])
    if profile["has_loyalty_program"] == 0:
        keywords.extend(["loyalty", "retention"])
    if profile["footfall_per_day"] < 50:
        keywords.extend(["marketing", "footfall", "traffic"])
    if profile["inventory_turnover"] < 4:
        keywords.extend(["inventory", "stock"])

    roadmap_chunks = roadmap_retriever.retrieve(
        regression_signal["feature_contributions"],
        keywords=keywords,
        top_n=2,
    )

    # 4. Generate recommendation via LLM
    profile_summary = merchant_store.get_profile_summary(merchant_id)
    llm_result = generate_recommendation(
        regression_signal=regression_signal,
        merchant_profile=profile_summary,
        roadmap_chunks=roadmap_chunks,
    )

    return RecommendationResponse(
        merchant_id=merchant_id,
        business_name=profile["business_name"],
        recommendation=llm_result["recommendation"],
        cited_signal=llm_result["cited_signal"],
        cited_roadmap_stage=llm_result["cited_roadmap_stage"],
        explanation=llm_result["explanation"],
        regression_signal=regression_signal,
        provider=llm_result.get("_provider", "unknown"),
    )


@app.get("/health")
def health_check():
    return {
        "model_loaded": regression_engine.pipeline is not None,
        "roadmap_loaded": len(roadmap_retriever.stages) > 0,
        "merchants_seeded": len(merchant_store.list_merchants()) > 0,
    }
