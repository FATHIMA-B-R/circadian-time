import pandas as pd
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pycombat import Combat

# ==========================================================
# SEED
# ==========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================================
# LOAD DATA
# ==========================================================

gse137732_ld = pd.read_csv("GSE137732_LD_final.csv", index_col=0)
gse137732_ll = pd.read_csv("GSE137732_LL_final.csv", index_col=0)
gse115583 = pd.read_csv("GSE115583_TPM_final_with_label (1).csv", index_col=0)

tair_df = pd.read_csv("gene_results_2026-06-03.tsv", sep="\t")

circadian_genes = set(tair_df["Locus"].astype(str).str.strip())

common_genes = sorted(
    set(gse115583.columns)
    & set(gse137732_ld.columns)
    & set(gse137732_ll.columns)
)

gene_cols = [g for g in common_genes if g in circadian_genes]

gse115583 = gse115583[gene_cols].copy()
gse137732_ld = gse137732_ld[gene_cols].copy()
gse137732_ll = gse137732_ll[gene_cols].copy()

# ==========================================================
# LABELS
# ==========================================================

gse115583["CT"] = np.repeat([2,4,6,8,10,12,14,16,18,20,22,24], 14)

ct_137732 = [2,5,8,11,14,17,20,23] * 2
gse137732_ld["CT"] = ct_137732
gse137732_ll["CT"] = ct_137732

all_df = pd.concat([
    gse115583,
    gse137732_ld,
    gse137732_ll
])

all_df["condition"] = (
    ["GSE115583"] * len(gse115583) +
    ["LD"] * len(gse137732_ld) +
    ["LL"] * len(gse137732_ll)
)

# ==========================================================
# ENCODE CONDITION
# ==========================================================

le = LabelEncoder()
all_df["condition_enc"] = le.fit_transform(all_df["condition"])

# ==========================================================
# LOG TRANSFORM
# ==========================================================

all_df[gene_cols] = np.log2(all_df[gene_cols] + 1)

gene_cols = list(all_df[gene_cols].columns[
    all_df[gene_cols].nunique() > 1
])

all_df = all_df[gene_cols + ["CT", "condition_enc"]]

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    all_df,
    test_size=0.2,
    random_state=42,
    stratify=all_df["CT"]
)

# ==========================================================
# COMBAT OUTSIDE CV
# ==========================================================

combat = Combat()

X_train = combat.fit_transform(
    train_df[gene_cols].values,
    train_df["condition_enc"].values
)

X_test = combat.transform(
    test_df[gene_cols].values,
    test_df["condition_enc"].values
)

y_train = train_df["CT"].values
y_test = test_df["CT"].values


# ==========================================================
# MODEL
# ==========================================================

class ChronoNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(self.net(x))

# ==========================================================
# LOSS HELPERS
# ==========================================================

def circular_error(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    return np.minimum(err, 24 - err)

# ==========================================================
# CV FUNCTION
# ==========================================================

def train_fold(X_tr, y_tr, X_val, y_val):

    # -------------------------
    # MI FEATURE SELECTION
    # -------------------------
    mi = mutual_info_regression(X_tr, y_tr, random_state=SEED)
    top_idx = np.argsort(mi)[-30:]

    X_tr = X_tr[:, top_idx]
    X_val = X_val[:, top_idx]

    # -------------------------
    # SCALING
    # -------------------------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # -------------------------
    # SIN/COS TARGET
    # -------------------------
    y_tr_sin = np.sin(2*np.pi*y_tr/24)
    y_tr_cos = np.cos(2*np.pi*y_tr/24)
    Y_tr = np.stack([y_tr_sin, y_tr_cos], axis=1)

    # -------------------------
    # TORCH DATA
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(Y_tr, dtype=torch.float32)
    )

    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = ChronoNet(X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # -------------------------
    # TRAIN
    # -------------------------
    for _ in range(30):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    # -------------------------
    # PREDICT
    # -------------------------
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.tensor(X_val, dtype=torch.float32).to(device)
        ).cpu().numpy()

    sin_p, cos_p = pred[:,0], pred[:,1]
    norm = np.sqrt(sin_p**2 + cos_p**2) + 1e-8
    sin_p /= norm
    cos_p /= norm

    ct_pred = (np.arctan2(sin_p, cos_p) * 24 / (2*np.pi)) % 24

    # Circular errors
    errors = circular_error(y_val, ct_pred)

    mae = np.mean(errors)
    medae = np.median(errors)
    rmse = np.sqrt(np.mean(errors**2))
    std_error = np.std(errors, ddof=1)

    return {
    "MAE": mae,
    "MedAE": medae,
    "RMSE": rmse,
    "STD": std_error
}

# ==========================================================
# CROSS VALIDATION
# ==========================================================

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):

    print(f"\nFold {fold+1}")
    metrics = train_fold(
        X_train[tr_idx],
        y_train[tr_idx],
        X_train[val_idx],
        y_train[val_idx]
    )

    print(
        f"Fold MAE={metrics['MAE']:.3f}, "
        f"MedAE={metrics['MedAE']:.3f}, "
        f"RMSE={metrics['RMSE']:.3f}"
    )

    cv_scores.append(metrics["MAE"])

print("\nCV Mean MAE:", np.mean(cv_scores))
print("CV Std:", np.std(cv_scores))

# ==========================================================
# FINAL TRAIN ON FULL TRAIN SET
# ==========================================================

final_metrics = train_fold(
    X_train,
    y_train,
    X_test,
    y_test
)

print("\n===== FINAL TEST =====")
print(f"MAE   : {final_metrics['MAE']:.3f}")
print(f"MedAE : {final_metrics['MedAE']:.3f}")
print(f"RMSE  : {final_metrics['RMSE']:.3f}")
print(f"STD   : {final_metrics['STD']:.3f}")
