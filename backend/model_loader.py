"""
Model loader — loads the trained regression model and metadata at backend startup.
"""

import json
from pathlib import Path
from typing import Any, Optional

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class RegressionEngine:
    """Loads and runs predictions using the pre-trained regression model."""

    def __init__(self):
        self.pipeline = None
        self.metadata: Optional[dict] = None
        self._loaded = False

    def load(self) -> None:
        import joblib

        model_path = MODELS_DIR / "regression_model.joblib"
        meta_path = MODELS_DIR / "model_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Run scripts/train_regression.py first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {meta_path}. "
                "Run scripts/train_regression.py first."
            )

        self.pipeline = joblib.load(model_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        self._loaded = True

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Run a single merchant's features through the model.

        Args:
            features: dict with keys matching ALL_FEATURES from training script.

        Returns:
            dict with prediction, confidence signal, and feature contributions.
        """
        if not self._loaded:
            self.load()

        import pandas as pd

        df = pd.DataFrame([features])
        prediction = float(self.pipeline.predict(df)[0])

        # Feature contributions (scaled feature * coefficient)
        regressor = self.pipeline.named_steps["regressor"]
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_scaled = preprocessor.transform(df)
        contributions = dict(zip(
            self.metadata["features"],
            (X_scaled[0] * regressor.coef_).round(4).tolist(),
        ))

        # Confidence signal: based on model R² and how far this prediction
        # is from the training mean (simple heuristic)
        r2 = self.metadata["metrics"]["r2_score"]
        confidence = min(max(r2, 0.0), 1.0)  # clamp to [0, 1]

        return {
            "predicted_growth_percent": round(prediction, 2),
            "confidence": round(confidence, 4),
            "feature_contributions": contributions,
            "model_r2": r2,
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Returns global feature importance (coefficients) from the model."""
        if not self._loaded:
            self.load()
        return self.metadata["feature_importance"]

    def get_metadata(self) -> dict[str, Any]:
        """Returns full model metadata."""
        if not self._loaded:
            self.load()
        return self.metadata
