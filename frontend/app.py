"""
Merchant Growth AI — Streamlit Demo UI
Run: streamlit run frontend/app.py
"""

import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go

from backend.model_loader import RegressionEngine
from backend.merchant_store import MerchantStore
from backend.roadmap_retriever import RoadmapRetriever
from backend.llm_orchestrator import generate_recommendation

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Merchant Growth AI",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load components (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    engine = RegressionEngine()
    engine.load()
    return engine

@st.cache_resource
def load_store():
    return MerchantStore()

@st.cache_resource
def load_retriever():
    retriever = RoadmapRetriever()
    retriever.load()
    return retriever

regression_engine = load_engine()
merchant_store = load_store()
roadmap_retriever = load_retriever()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Merchant Growth AI")
st.caption("AI-powered business growth advisor for small merchants")

# ---------------------------------------------------------------------------
# Sidebar — Merchant selector
# ---------------------------------------------------------------------------
st.sidebar.header("Select Merchant")
merchants = merchant_store.list_merchants()
merchant_names = {m["business_name"]: m["merchant_id"] for m in merchants}

selected_name = st.sidebar.selectbox("Choose a merchant persona", list(merchant_names.keys()))
selected_id = merchant_names[selected_name]

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
profile = merchant_store.get_profile(selected_id)
features = merchant_store.get_regression_features(selected_id)

# --- Merchant Profile Card ---
st.subheader(f"🏪 {profile['business_name']}")
col1, col2, col3 = st.columns(3)
col1.metric("Business Type", profile["business_type"])
col2.metric("Sales Trend", profile["sales_trend"].capitalize())
col3.metric("Years in Business", profile["years_in_business"])

st.markdown(f"**Location:** {profile['location']}")
st.markdown(f"**Seasonal Notes:** {profile['seasonal_notes']}")

# --- Transaction Pattern Chart ---
st.subheader("📊 Business Profile")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=["Digital Adoption", "Retention Rate", "Loyalty Program"],
    y=[features["digital_payment_adoption"], features["customer_retention_rate"],
       features["has_loyalty_program"] * 100],
    marker_color=["#4CAF50", "#2196F3", "#FF9800"],
    text=[f"{features['digital_payment_adoption']}%",
          f"{features['customer_retention_rate']}%",
          "Yes" if features["has_loyalty_program"] else "No"],
    textposition="auto",
))
fig.update_layout(
    yaxis_title="Value",
    height=300,
    margin=dict(t=20),
)
st.plotly_chart(fig, use_container_width=True)

# Key metrics in columns
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Transaction", f"₹{features['avg_transaction_value']:.0f}")
c2.metric("Daily Footfall", f"~{features['footfall_per_day']}")
c3.metric("Inventory Turnover", f"{features['inventory_turnover']}x/mo")
c4.metric("Marketing Spend", f"₹{features['marketing_spend_monthly']:.0f}/mo")

# --- Generate Recommendation ---
st.divider()
st.subheader("🎯 Growth Recommendation")

if st.button("Generate Recommendation", type="primary", key="recommend_btn"):
    with st.spinner("Analyzing merchant data and generating advice..."):
        # Run regression
        regression_signal = regression_engine.predict(features)

        # Build keywords from profile
        keywords = []
        if profile["sales_trend"] == "declining":
            keywords.extend(["retention", "customers", "churn"])
        if features["digital_payment_adoption"] < 50:
            keywords.extend(["digital", "payment", "upi"])
        if features["has_loyalty_program"] == 0:
            keywords.extend(["loyalty", "retention"])
        if features["footfall_per_day"] < 50:
            keywords.extend(["marketing", "footfall", "traffic"])
        if features["inventory_turnover"] < 4:
            keywords.extend(["inventory", "stock"])

        # Retrieve roadmap
        roadmap_chunks = roadmap_retriever.retrieve(
            regression_signal["feature_contributions"],
            keywords=keywords,
            top_n=2,
        )

        # Generate recommendation
        profile_summary = merchant_store.get_profile_summary(selected_id)
        llm_result = generate_recommendation(
            regression_signal=regression_signal,
            merchant_profile=profile_summary,
            roadmap_chunks=roadmap_chunks,
        )

    # --- Display recommendation ---
    st.success(llm_result["recommendation"])

    # Reasoning chain
    st.subheader("🔍 Reasoning Chain")

    chain_col1, chain_col2, chain_col3 = st.columns(3)

    with chain_col1:
        st.markdown("**📊 Data Signal**")
        st.info(llm_result["cited_signal"])

    with chain_col2:
        stage = roadmap_retriever.get_stage_by_id(llm_result["cited_roadmap_stage"])
        stage_name = stage["name"] if stage else llm_result["cited_roadmap_stage"]
        st.markdown(f"**🗺️ Roadmap Stage**")
        st.info(f"**{stage_name}**")

    with chain_col3:
        st.markdown(f"**💡 Provider**")
        st.info(llm_result.get("_provider", "unknown"))

    # Explanation
    st.markdown("**Explanation:**")
    st.write(llm_result["explanation"])

    # Regression details
    with st.expander("📈 Regression Model Details"):
        st.json({
            "predicted_growth_percent": regression_signal["predicted_growth_percent"],
            "confidence": regression_signal["confidence"],
            "feature_contributions": regression_signal["feature_contributions"],
        })

else:
    st.info("Click the button above to generate a personalized growth recommendation for this merchant.")
