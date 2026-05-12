import os
import numpy as np
import pandas as pd

np.random.seed(42)

os.makedirs("data/raw", exist_ok=True)

# Config
N_USERS    = 500
N_PRODUCTS = 200
N_INTERACTIONS = 8000  

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden",
              "Sports", "Beauty", "Toys", "Automotive"]

#  Products 
product_ids = [f"P{str(i).zfill(4)}" for i in range(1, N_PRODUCTS + 1)]
products = pd.DataFrame({
    "product_id":   product_ids,
    "product_name": [f"Product {i}" for i in range(1, N_PRODUCTS + 1)],
    "category":     np.random.choice(CATEGORIES, N_PRODUCTS),
    "price":        np.round(np.random.lognormal(mean=3.5, sigma=1.0, size=N_PRODUCTS), 2),
    "avg_rating":   np.round(np.random.uniform(2.5, 5.0, N_PRODUCTS), 1),
    "n_reviews":    np.random.randint(10, 2000, N_PRODUCTS),
})

# Users
user_ids = [f"U{str(i).zfill(4)}" for i in range(1, N_USERS + 1)]

# Interactions (implicit + explicit) 
popularity_weights = np.random.zipf(1.5, N_PRODUCTS).astype(float)
popularity_weights /= popularity_weights.sum()

sampled_users    = np.random.choice(user_ids, N_INTERACTIONS)
sampled_products = np.random.choice(product_ids, N_INTERACTIONS, p=popularity_weights)

interactions = pd.DataFrame({
    "user_id":    sampled_users,
    "product_id": sampled_products,
    "event_type": np.random.choice(
        ["view", "add_to_cart", "purchase"],
        N_INTERACTIONS,
        p=[0.65, 0.20, 0.15]
    ),
    "rating": np.where(
        np.random.random(N_INTERACTIONS) < 0.35,  
        np.random.randint(1, 6, N_INTERACTIONS),
        np.nan
    ),
    "timestamp": pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, N_INTERACTIONS), unit="D"
    ),
})

# Remove duplicate (user, product) purchase events
interactions = interactions.drop_duplicates(subset=["user_id", "product_id", "event_type"])

products.to_csv("data/raw/products.csv", index=False)
interactions.to_csv("data/raw/interactions.csv", index=False)

print(f"Generated {len(products)} products → data/raw/products.csv")
print(f"Generated {len(interactions)} interactions → data/raw/interactions.csv")
print(f"Unique users:    {interactions['user_id'].nunique()}")
print(f"Unique products: {interactions['product_id'].nunique()}")
density = len(interactions) / (N_USERS * N_PRODUCTS) * 100
print(f"Matrix density:  {density:.2f}%")
