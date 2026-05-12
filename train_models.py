"""
Step 3: Train three recommendation models.
  1. Item-Based Collaborative Filtering (cosine similarity)
  2. SVD Matrix Factorization (scipy TruncatedSVD)
  3. Neural Collaborative Filtering (PyTorch, Embedding + MLP)

Outputs saved to models/
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.makedirs("models", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned/interactions_clean.csv")

# Encode user/item IDs to integers
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["user_idx"] = user_enc.fit_transform(df["user_id"])
df["item_idx"] = item_enc.fit_transform(df["product_id"])

N_USERS = df["user_idx"].nunique()
N_ITEMS = df["item_idx"].nunique()

# Interaction matrix (implicit scores)
R = csr_matrix(
    (df["implicit_score"].values, (df["user_idx"].values, df["item_idx"].values)),
    shape=(N_USERS, N_ITEMS)
)

print(f"Matrix shape: {R.shape}, density: {R.nnz / (N_USERS * N_ITEMS) * 100:.2f}%")


# ── 1. Item-Based Collaborative Filtering ────────────────────────────────────
print("\nTraining Item-Based CF...")
from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(R.T)   # shape: (N_ITEMS, N_ITEMS)

item_cf_model = {
    "item_similarity": item_sim,
    "user_enc": user_enc,
    "item_enc": item_enc,
    "R": R,
}

with open("models/item_cf.pkl", "wb") as f:
    pickle.dump(item_cf_model, f)
print("Saved models/item_cf.pkl")


# ── 2. SVD Matrix Factorization ──────────────────────────────────────────────
print("\nTraining SVD (k=50)...")
N_FACTORS = 50
svd = TruncatedSVD(n_components=N_FACTORS, random_state=42)
U = svd.fit_transform(R)          # user latent factors (N_USERS × k)
V = svd.components_               # item latent factors (k × N_ITEMS)

svd_model = {
    "svd":      svd,
    "U":        U,
    "V":        V,
    "user_enc": user_enc,
    "item_enc": item_enc,
}

with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)
print("Saved models/svd_model.pkl")


# ── 3. Neural Collaborative Filtering ────────────────────────────────────────
print("\nTraining Neural CF...")

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users  = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items  = torch.tensor(df["item_idx"].values, dtype=torch.long)
        # Normalise implicit scores to [0,1]
        scores = df["implicit_score"].values.astype(float)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        self.scores = torch.tensor(scores, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.scores[idx]


class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, hidden=[64, 32]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        layers = []
        in_dim = emb_dim * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        x = torch.cat([u, i], dim=1)
        return self.mlp(x).squeeze()


DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM    = 32
EPOCHS     = 10
BATCH_SIZE = 512
LR         = 1e-3

dataset    = InteractionDataset(df)
loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model      = NCF(N_USERS, N_ITEMS, emb_dim=EMB_DIM).to(DEVICE)
optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
criterion  = nn.MSELoss()

epoch_losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for users, items, scores in loader:
        users, items, scores = users.to(DEVICE), items.to(DEVICE), scores.to(DEVICE)
        optimizer.zero_grad()
        preds = model(users, items)
        loss  = criterion(preds, scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    epoch_losses.append(avg)
    print(f"  Epoch {epoch:02d}/{EPOCHS} — loss: {avg:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "n_users":     N_USERS,
    "n_items":     N_ITEMS,
    "emb_dim":     EMB_DIM,
    "user_enc":    user_enc,
    "item_enc":    item_enc,
    "epoch_losses": epoch_losses,
}, "models/ncf_model.pt")
print("Saved models/ncf_model.pt")

print("\nAll models trained successfully.")
