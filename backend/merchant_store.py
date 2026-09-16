"""
Merchant Profile Store — reads seeded merchant data from SQLite.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "merchants.db"


class MerchantStore:
    """Read-only access to seeded merchant profiles."""

    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(str(DB_PATH))
            self.conn.row_factory = sqlite3.Row
        return self.conn

    def list_merchants(self) -> list[dict[str, Any]]:
        """Returns all merchant profiles (summary view)."""
        conn = self._ensure_conn()
        cursor = conn.execute(
            "SELECT merchant_id, business_name, business_type, sales_trend FROM merchant_profiles"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_profile(self, merchant_id: str) -> Optional[dict[str, Any]]:
        """Returns full profile for a single merchant."""
        conn = self._ensure_conn()
        cursor = conn.execute(
            "SELECT * FROM merchant_profiles WHERE merchant_id = ?", (merchant_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        profile = dict(row)
        # Parse JSON fields
        for field in ["peak_days", "low_days", "customer_segment_tags"]:
            profile[field] = json.loads(profile[field])
        return profile

    def get_regression_features(self, merchant_id: str) -> Optional[dict[str, Any]]:
        """Returns just the regression-model features for a merchant."""
        profile = self.get_profile(merchant_id)
        if profile is None:
            return None
        return {
            "digital_payment_adoption": profile["digital_payment_adoption"],
            "marketing_spend_monthly": profile["marketing_spend_monthly"],
            "customer_retention_rate": profile["customer_retention_rate"],
            "avg_transaction_value": profile["avg_transaction_value"],
            "footfall_per_day": profile["footfall_per_day"],
            "inventory_turnover": profile["inventory_turnover"],
            "years_in_business": profile["years_in_business"],
            "has_loyalty_program": profile["has_loyalty_program"],
        }

    def get_profile_summary(self, merchant_id: str) -> str:
        """Returns a compact text summary of the merchant for LLM context."""
        profile = self.get_profile(merchant_id)
        if profile is None:
            return "Merchant not found."
        segments = ", ".join(profile["customer_segment_tags"])
        peak = ", ".join(profile["peak_days"])
        low = ", ".join(profile["low_days"])
        return (
            f"Business: {profile['business_name']} ({profile['business_type']})\n"
            f"Location: {profile['location']}\n"
            f"Sales trend: {profile['sales_trend']}\n"
            f"Peak days: {peak}\n"
            f"Low days: {low}\n"
            f"Customer segments: {segments}\n"
            f"Seasonal notes: {profile['seasonal_notes']}\n"
            f"Digital payment adoption: {profile['digital_payment_adoption']}%\n"
            f"Marketing spend: ₹{profile['marketing_spend_monthly']:.0f}/month\n"
            f"Customer retention: {profile['customer_retention_rate']}%\n"
            f"Avg transaction value: ₹{profile['avg_transaction_value']:.0f}\n"
            f"Footfall: ~{profile['footfall_per_day']}/day\n"
            f"Inventory turnover: {profile['inventory_turnover']}x/month\n"
            f"Years in business: {profile['years_in_business']}\n"
            f"Loyalty program: {'Yes' if profile['has_loyalty_program'] else 'No'}"
        )

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
