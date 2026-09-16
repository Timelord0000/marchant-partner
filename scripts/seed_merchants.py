"""
Database setup and seeding for merchant personas.
Creates SQLite DB with merchant_profiles and past_suggestions tables,
then seeds 3 distinct demo personas.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "merchants.db"


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS merchant_profiles (
            merchant_id TEXT PRIMARY KEY,
            business_name TEXT NOT NULL,
            business_type TEXT NOT NULL,
            location TEXT NOT NULL,
            sales_trend TEXT NOT NULL,
            peak_days TEXT NOT NULL,
            low_days TEXT NOT NULL,
            customer_segment_tags TEXT NOT NULL,
            seasonal_notes TEXT NOT NULL,
            -- Regression-relevant features (stored for easy access)
            digital_payment_adoption REAL NOT NULL,
            marketing_spend_monthly REAL NOT NULL,
            customer_retention_rate REAL NOT NULL,
            avg_transaction_value REAL NOT NULL,
            footfall_per_day INTEGER NOT NULL,
            inventory_turnover REAL NOT NULL,
            years_in_business INTEGER NOT NULL,
            has_loyalty_program INTEGER NOT NULL,
            last_updated TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS past_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            merchant_id TEXT NOT NULL,
            date TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            cited_signal TEXT NOT NULL,
            cited_roadmap_stage TEXT NOT NULL,
            outcome_simulated TEXT,
            FOREIGN KEY (merchant_id) REFERENCES merchant_profiles(merchant_id)
        );
    """)
    conn.commit()


PERSONAS = [
    {
        "merchant_id": "kirana_001",
        "business_name": "Gupta General Store",
        "business_type": "kirana store",
        "location": "Residential neighborhood, Tier-2 city",
        "sales_trend": "growing",
        "peak_days": '["Saturday", "Sunday"]',
        "low_days": '["Monday", "Tuesday"]',
        "customer_segment_tags": '["repeat local customers", "families", "daily essentials buyers"]',
        "seasonal_notes": "Dips slightly in monsoon (Jul-Aug) due to reduced foot traffic; spikes during Diwali and festival season.",
        "digital_payment_adoption": 65.0,
        "marketing_spend_monthly": 8000.0,
        "customer_retention_rate": 72.0,
        "avg_transaction_value": 380.0,
        "footfall_per_day": 95,
        "inventory_turnover": 5.5,
        "years_in_business": 8,
        "has_loyalty_program": 1,
    },
    {
        "merchant_id": "food_002",
        "business_name": "Spice Corner Food Stall",
        "business_type": "food stall",
        "location": "Near office district, Tier-1 city",
        "sales_trend": "declining",
        "peak_days": '["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]',
        "low_days": '["Saturday", "Sunday"]',
        "customer_segment_tags": '["one-off foot traffic", "office workers", "lunch crowd"]',
        "seasonal_notes": "Weekday-heavy due to office crowd. Summers are strong; monsoon and winter see 30% drop in footfall.",
        "digital_payment_adoption": 45.0,
        "marketing_spend_monthly": 3000.0,
        "customer_retention_rate": 28.0,
        "avg_transaction_value": 180.0,
        "footfall_per_day": 140,
        "inventory_turnover": 8.0,
        "years_in_business": 2,
        "has_loyalty_program": 0,
    },
    {
        "merchant_id": "service_003",
        "business_name": "QuickFix Repair Shop",
        "business_type": "service provider",
        "location": "Market street, Tier-2 city",
        "sales_trend": "flat",
        "peak_days": '["Wednesday", "Saturday"]',
        "low_days": '["Monday", "Friday"]',
        "customer_segment_tags": '["local residents", "walk-in repairs", "referral customers"]',
        "seasonal_notes": "Steady year-round except holiday weeks. Slight spike before school/office reopenings.",
        "digital_payment_adoption": 82.0,
        "marketing_spend_monthly": 2000.0,
        "customer_retention_rate": 55.0,
        "avg_transaction_value": 650.0,
        "footfall_per_day": 25,
        "inventory_turnover": 2.5,
        "years_in_business": 12,
        "has_loyalty_program": 0,
    },
]


def seed_personas(conn: sqlite3.Connection) -> None:
    now = datetime.now().isoformat()
    for p in PERSONAS:
        conn.execute(
            """INSERT OR REPLACE INTO merchant_profiles
               (merchant_id, business_name, business_type, location, sales_trend,
                peak_days, low_days, customer_segment_tags, seasonal_notes,
                digital_payment_adoption, marketing_spend_monthly,
                customer_retention_rate, avg_transaction_value, footfall_per_day,
                inventory_turnover, years_in_business, has_loyalty_program,
                last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                p["merchant_id"], p["business_name"], p["business_type"],
                p["location"], p["sales_trend"], p["peak_days"], p["low_days"],
                p["customer_segment_tags"], p["seasonal_notes"],
                p["digital_payment_adoption"], p["marketing_spend_monthly"],
                p["customer_retention_rate"], p["avg_transaction_value"],
                p["footfall_per_day"], p["inventory_turnover"],
                p["years_in_business"], p["has_loyalty_program"], now,
            ),
        )
    conn.commit()


def init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    create_tables(conn)
    seed_personas(conn)
    return conn


if __name__ == "__main__":
    conn = init_db()
    print(f"Database created → {DB_PATH}")

    # Verify
    cursor = conn.execute("SELECT merchant_id, business_name, sales_trend FROM merchant_profiles")
    for row in cursor.fetchall():
        print(f"  {row[0]:15s} | {row[1]:30s} | trend: {row[2]}")

    conn.close()
