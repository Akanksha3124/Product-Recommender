
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.makedirs("dashboard", exist_ok=True)

# ── Colour palette (mirrors social_dashboard) ────────────────────────────────
COLORS = {
    "blue":   "#378ADD",
    "orange": "#EF9F27",
    "teal":   "#1D9E75",
    "coral":  "#D85A30",
    "purple": "#7F77DD",
    "gray":   "#888780",
}

#  Load data 
interactions_path = "data/cleaned/interactions_clean.csv"
products_path     = "data/cleaned/products_clean.csv"

if not os.path.exists(interactions_path):
    print("Cleaned data not found. Run scripts/clean_data.py first.")
    exit()

interactions = pd.read_csv(interactions_path)
products     = pd.read_csv(products_path)

# Load models 
item_cf_model = svd_model = ncf_ckpt = None

if os.path.exists("models/item_cf.pkl"):
    with open("models/item_cf.pkl", "rb") as f:
        item_cf_model = pickle.load(f)

if os.path.exists("models/svd_model.pkl"):
    with open("models/svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

if os.path.exists("models/ncf_model.pt"):
    ncf_ckpt = torch.load("models/ncf_model.pt", map_location="cpu")

#  Build charts 
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [{"type": "xy"}, {"type": "domain"}],   # row 1
        [{"type": "xy"}, {"type": "xy"}]        # row 2
    ],
    subplot_titles=(
        "Top 15 Products by Total Interactions",
        "Events by Type",
        "User Activity Distribution",
        "Purchase Conversion Rate by Category"
    )
)

# Chart 1: Top 15 products
top15 = (
    interactions.groupby("product_id")["n_events"]
    .sum().sort_values(ascending=False).head(15)
    .reset_index()
    .merge(products[["product_id", "product_name"]], on="product_id")
)
fig.add_trace(go.Bar(
    x=top15["n_events"],
    y=top15["product_name"].str[:30],
    orientation="h",
    marker_color=COLORS["blue"],
    name="Interactions",
), row=1, col=1)

#Chart 2: Event type pie 
raw = pd.read_csv("data/raw/interactions.csv")
event_counts = raw["event_type"].value_counts()
fig.add_trace(go.Pie(
    labels=event_counts.index,
    values=event_counts.values,
    hole=0.5,  # donut style
    marker_colors=[COLORS["blue"], COLORS["orange"], COLORS["teal"]],
    textinfo="label+percent",   # 👈 shows View / Add to Cart / Purchase
), row=1, col=2)

# Chart 3: User activity histogram 
user_activity = interactions.groupby("user_id")["n_events"].sum().values
fig.add_trace(go.Histogram(
    x=user_activity,
    nbinsx=30,
    marker_color=COLORS["purple"],
    name="Users",
), row=2, col=1)

# Chart 4: Category purchase rate 
cat_data = interactions.merge(products[["product_id", "category"]], on="product_id")
cat_purchase = (
    cat_data.groupby("category")["has_purchase"]
    .mean().sort_values(ascending=False).reset_index()
)
best_cat = cat_purchase.iloc[0]["category"]
fig.add_trace(go.Bar(
    x=cat_purchase["category"],
    y=cat_purchase["has_purchase"],
    marker_color=[
        COLORS["coral"] if c == best_cat else COLORS["teal"]
        for c in cat_purchase["category"]
    ],
    name="Purchase rate",
), row=2, col=2)

#  Chart 5: Model comparison bar 
def quick_ndcg(rec_fn, test_dict, k=10, max_users=200):
    ndcg_vals = []
    for user, held_out in list(test_dict.items())[:max_users]:
        recs = rec_fn(user)
        for rank, item in enumerate(recs[:k], start=1):
            if item == held_out:
                ndcg_vals.append(1.0 / np.log2(rank + 1))
                break
        else:
            ndcg_vals.append(0.0)
    return float(np.mean(ndcg_vals))

interactions["last_event"] = pd.to_datetime(interactions["last_event"])
test = (
    interactions.sort_values("last_event")
    .groupby("user_id").tail(1)
    .set_index("user_id")["product_id"]
    .to_dict()
)

model_scores = {}

if item_cf_model:
    ue, ie, R, sim = (item_cf_model[k] for k in ["user_enc", "item_enc", "R", "item_similarity"])
    def rec_cf(user_id):
        if user_id not in ue.classes_: return []
        u = ue.transform([user_id])[0]
        v = R[u].toarray().flatten()
        s = v @ sim; s[v > 0] = -1
        return ie.inverse_transform(np.argsort(s)[::-1][:50]).tolist()
    model_scores["Item-CF"] = quick_ndcg(rec_cf, test)

if svd_model:
    U, V, ue_s, ie_s = (svd_model[k] for k in ["U", "V", "user_enc", "item_enc"])
    def rec_svd(user_id):
        if user_id not in ue_s.classes_: return []
        u = ue_s.transform([user_id])[0]
        s = U[u] @ V
        if item_cf_model and user_id in item_cf_model["user_enc"].classes_:
            u2 = item_cf_model["user_enc"].transform([user_id])[0]
            s[item_cf_model["R"][u2].toarray().flatten() > 0] = -1
        return ie_s.inverse_transform(np.argsort(s)[::-1][:50]).tolist()
    model_scores["SVD"] = quick_ndcg(rec_svd, test)

if ncf_ckpt:
    class NCF(nn.Module):
        def __init__(self, n_users, n_items, emb_dim=32):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, emb_dim)
            self.item_emb = nn.Embedding(n_items, emb_dim)
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        def forward(self, u, i):
            return self.mlp(torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)).squeeze()

    ncf_mdl = NCF(ncf_ckpt["n_users"], ncf_ckpt["n_items"], ncf_ckpt["emb_dim"])
    ncf_mdl.load_state_dict(ncf_ckpt["model_state"])
    ncf_mdl.eval()
    all_items = torch.arange(ncf_ckpt["n_items"])
    ue_nc, ie_nc = ncf_ckpt["user_enc"], ncf_ckpt["item_enc"]

    def rec_ncf(user_id):
        if user_id not in ue_nc.classes_: return []
        u = int(ue_nc.transform([user_id])[0])
        with torch.no_grad():
            s = ncf_mdl(torch.tensor([u] * ncf_ckpt["n_items"]), all_items).numpy()
        if item_cf_model and user_id in item_cf_model["user_enc"].classes_:
            u2 = item_cf_model["user_enc"].transform([user_id])[0]
            s[item_cf_model["R"][u2].toarray().flatten() > 0] = -1
        return ie_nc.inverse_transform(np.argsort(s)[::-1][:50]).tolist()

    model_scores["NCF"] = quick_ndcg(rec_ncf, test)

if model_scores:
    best_model = max(model_scores, key=model_scores.get)
    fig.add_trace(go.Bar(
        x=list(model_scores.keys()),
        y=list(model_scores.values()),
        marker_color=[
            COLORS["coral"] if m == best_model else COLORS["blue"]
            for m in model_scores
        ],
        text=[f"{v:.4f}" for v in model_scores.values()],
        textposition="outside",
        name="NDCG@10",
    ), row=2, col=1)
else:
    fig.add_annotation(
        text="Train models first (scripts/train_models.py)",
        xref="paper", yref="paper", x=0.25, y=0.1,
        showarrow=False, row=3, col=1
    )

# Chart 6: NCF loss curve 
if ncf_ckpt:
    losses = ncf_ckpt["epoch_losses"]
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses) + 1)),
        y=losses,
        mode="lines+markers",
        line=dict(color=COLORS["coral"], width=2),
        marker=dict(size=7),
        name="NCF loss",
    ), row=3, col=2)

#  Layout 
fig.update_layout(
    title=dict(
        text="E-Commerce Product Recommendation Dashboard",
        font=dict(size=22),
        x=0.5,
    ),
    height=1050,
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial, sans-serif", size=12, color="#333333"),
    margin=dict(t=100, b=60, l=60, r=80),
)

fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", linecolor="#cccccc")
fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", linecolor="#cccccc")

output_path = "dashboard/dashboard.html"
fig.write_html(output_path, include_plotlyjs="cdn")
print(f"Dashboard saved to: {output_path}")
print("Open dashboard/dashboard.html in your browser.")
