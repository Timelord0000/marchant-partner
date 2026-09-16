"""
Phase 0: Regression Insight Engine — Training Script

Generates synthetic merchant survey data, trains a linear regression model,
and saves the trained model + metadata as static artifacts.

Features are designed to map to roadmap categories:
  - digital_payment_adoption  → Stage 4: Digital Adoption
  - marketing_spend_monthly   → Stage 3: Marketing
  - customer_retention_rate   → Stage 2: Retention
  - avg_transaction_value     → Stage 1: Pricing
  - footfall_per_day          → Stage 3: Marketing / Location
  - inventory_turnover        → Operations
  - years_in_business         → General business maturity
  - has_loyalty_program       → Stage 2: Retention
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_SAMPLES = 1000
RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Feature columns (must match what the backend sends at inference time)
NUMERIC_FEATURES = [
    "digital_payment_adoption",   # 0-100 (% of transactions digital)
    "marketing_spend_monthly",    # INR per month
    "customer_retention_rate",    # 0-100 (% repeat customers)
    "avg_transaction_value",      # INR
    "footfall_per_day",           # avg customers per day
    "inventory_turnover",         # times per month
    "years_in_business",          # years
]

BINARY_FEATURES = [
    "has_loyalty_program",        # 0 or 1
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES


# ---------------------------------------------------------------------------
# 1. Generate Synthetic Merchant Survey Data
# ---------------------------------------------------------------------------
def generate_synthetic_data(n: int = NUM_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Creates a synthetic dataset mimicking a small-business merchant survey.
    Target: revenue_growth_percent (annual revenue growth %).
    The data encodes realistic-ish correlations so the model learns meaningful
    feature weights (e.g. higher digital adoption → higher growth).
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "digital_payment_adoption": rng.uniform(10, 100, n).round(1),
        "marketing_spend_monthly": rng.uniform(0, 50000, n).round(0),
        "customer_retention_rate": rng.uniform(10, 95, n).round(1),
        "avg_transaction_value": rng.uniform(50, 2000, n).round(0),
        "footfall_per_day": rng.integers(10, 500, n),
        "inventory_turnover": rng.uniform(1, 12, n).round(1),
        "years_in_business": rng.integers(1, 25, n),
        "has_loyalty_program": rng.choice([0, 1], n, p=[0.6, 0.4]),
    })

    # --- Generate target with known signal ---
    # Each feature contributes a weighted amount to revenue growth,
    # plus noise. This ensures the model can learn real coefficients.
    noise = rng.normal(0, 3, n)
    growth = (
        0.08 * df["digital_payment_adoption"]        # +0.08% per % digital
        + 0.0003 * df["marketing_spend_monthly"]     # small positive from marketing
        + 0.10 * df["customer_retention_rate"]       # retention is strong
        + 0.005 * df["avg_transaction_value"]        # higher basket → growth
        + 0.015 * df["footfall_per_day"]             # footfall matters
        + 0.80 * df["inventory_turnover"]            # turnover efficiency
        + 0.30 * df["years_in_business"]             # slight maturity bonus
        + 2.50 * df["has_loyalty_program"]           # loyalty program boost
        + noise
    )
    df["revenue_growth_percent"] = growth.clip(-20, 80).round(2)

    return df


# ---------------------------------------------------------------------------
# 2. Train Model
# ---------------------------------------------------------------------------
def train_model(df: pd.DataFrame) -> Pipeline:
    """Trains a linear regression pipeline with standard scaling."""
    X = df[ALL_FEATURES]
    y = df["revenue_growth_percent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    metrics = {
        "r2_score": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    }

    # Feature importances (coefficients after scaling)
    regressor = pipeline.named_steps["regressor"]
    feature_importance = dict(zip(ALL_FEATURES, regressor.coef_.round(4).tolist()))

    return pipeline, metrics, feature_importance, X_test, y_test


# ---------------------------------------------------------------------------
# 3. Save Artifacts
# ---------------------------------------------------------------------------
def save_artifacts(
    pipeline: Pipeline,
    metrics: dict,
    feature_importance: dict,
    df: pd.DataFrame,
):
    """Saves model, metadata, and a sample of the training data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = OUTPUT_DIR / "regression_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved → {model_path}")

    # Metadata (metrics + feature importance + feature schema)
    metadata = {
        "model_type": "LinearRegression",
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "binary_features": BINARY_FEATURES,
        "target": "revenue_growth_percent",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "training_samples": len(df),
    }
    meta_path = OUTPUT_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved → {meta_path}")

    # Sample data for reference / frontend charts
    sample_path = DATA_DIR / "training_sample.csv"
    df.sample(min(50, len(df)), random_state=RANDOM_STATE).to_csv(sample_path, index=False)
    print(f"Sample data saved → {sample_path}")

    return metadata


# ---------------------------------------------------------------------------
# 4. Demo: Predict for a sample merchant
# ---------------------------------------------------------------------------
def demo_predict(pipeline: Pipeline, metadata: dict):
    """Runs a prediction on a sample merchant to verify the pipeline works."""
    sample_merchant = {
        "digital_payment_adoption": 72.0,
        "marketing_spend_monthly": 15000.0,
        "customer_retention_rate": 55.0,
        "avg_transaction_value": 450.0,
        "footfall_per_day": 120,
        "inventory_turnover": 6.5,
        "years_in_business": 5,
        "has_loyalty_program": 1,
    }

    X_sample = pd.DataFrame([sample_merchant])
    prediction = pipeline.predict(X_sample)[0]

    print("\n--- Demo Prediction ---")
    print(f"Sample merchant features: {sample_merchant}")
    print(f"Predicted revenue growth: {prediction:.2f}%")
    print(f"Model R²: {metadata['metrics']['r2_score']}")
    print(f"Feature importances:")
    for feat, coeff in sorted(metadata["feature_importance"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:35s} → {coeff:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Merchant Growth AI — Regression Model Training")
    print("=" * 60)

    print("\n[1/4] Generating synthetic merchant survey data...")
    df = generate_synthetic_data()
    print(f"  → {len(df)} samples, {len(ALL_FEATURES)} features")

    print("\n[2/4] Training linear regression model...")
    pipeline, metrics, feature_importance, X_test, y_test = train_model(df)
    print(f"  → R²: {metrics['r2_score']}  MAE: {metrics['mae']}  RMSE: {metrics['rmse']}")

    print("\n[3/4] Saving artifacts...")
    metadata = save_artifacts(pipeline, metrics, feature_importance, df)

    print("\n[4/4] Running demo prediction...")
    demo_predict(pipeline, metadata)

    print("\n" + "=" * 60)
    print("Training complete. Model ready for Phase 3 (LLM Orchestrator).")
    print("=" * 60)
