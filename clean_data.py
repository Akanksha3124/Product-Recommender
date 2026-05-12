"""
Step 2: Clean interaction + product data and save to SQLite.
"""

import os
import sqlite3
import pandas as pd

os.makedirs("data/cleaned", exist_ok=True)

EVENT_WEIGHTS = {"view": 1, "add_to_cart": 3, "purchase": 5}


def clean_interactions():
    path = "data/raw/interactions.csv"
    if not os.path.exists(path):
        print("Interactions file not found. Run generate_data.py first.")
        return None

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month"]       = df["timestamp"].dt.month_name()

    # Implicit score: weighted sum of events per (user, product)
    df["implicit_score"] = df["event_type"].map(EVENT_WEIGHTS)
    agg = df.groupby(["user_id", "product_id"]).agg(
        implicit_score=("implicit_score", "sum"),
        n_events=("event_type", "count"),
        last_event=("timestamp", "max"),
        has_purchase=("event_type", lambda x: int("purchase" in x.values)),
    ).reset_index()

    # Explicit score: use rating when available, else NaN
    ratings = (
        df[df["rating"].notna()]
        .groupby(["user_id", "product_id"])["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "explicit_score"})
    )
    agg = agg.merge(ratings, on=["user_id", "product_id"], how="left")

    agg.to_csv("data/cleaned/interactions_clean.csv", index=False)
    print(f"Interactions: cleaned {len(agg)} (user, product) pairs → data/cleaned/interactions_clean.csv")
    return agg


def clean_products():
    path = "data/raw/products.csv"
    if not os.path.exists(path):
        print("Products file not found. Run generate_data.py first.")
        return None

    df = pd.read_csv(path)
    df["price_bucket"] = pd.cut(
        df["price"],
        bins=[0, 20, 50, 100, 250, float("inf")],
        labels=["<$20", "$20-50", "$50-100", "$100-250", "$250+"]
    )
    df.to_csv("data/cleaned/products_clean.csv", index=False)
    print(f"Products: cleaned {len(df)} rows → data/cleaned/products_clean.csv")
    return df


def save_to_database(interactions_df, products_df):
    conn = sqlite3.connect("data/recommendations.db")
    if interactions_df is not None:
        interactions_df.to_sql("interactions", conn, if_exists="replace", index=False)
        print("Interactions saved to SQLite.")
    if products_df is not None:
        products_df.to_sql("products", conn, if_exists="replace", index=False)
        print("Products saved to SQLite.")
    conn.close()
    print("Database ready at: data/recommendations.db")


if __name__ == "__main__":
    interactions = clean_interactions()
    products     = clean_products()
    save_to_database(interactions, products)
