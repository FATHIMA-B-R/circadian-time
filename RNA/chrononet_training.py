import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pycombat import Combat

import random
import numpy as np
import torch
from scipy.stats import pearsonr


SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

# ==========================================================
# LOAD DATA
# ==========================================================

gse137732_ld = pd.read_csv(
    "GSE137732_LD_final.csv",
    index_col=0
)

gse137732_ll = pd.read_csv(
    "GSE137732_LL_final.csv",
    index_col=0
)

gse115583 = pd.read_csv(
    "GSE115583_TPM_final_with_label (1).csv",
    index_col=0
)

tair_df = pd.read_csv(
    "gene_results_2026-06-03.tsv",
    sep="\t"
)

# ==========================================================
# LOAD TAIR CIRCADIAN GENES
# ==========================================================

circadian_genes = set(
    tair_df["Locus"]
    .astype(str)
    .str.strip()
)

# ==========================================================
# COMMON GENES
# ==========================================================

common_genes = sorted(
    set(gse115583.columns)
    & set(gse137732_ld.columns)
    & set(gse137732_ll.columns)
)

# ==========================================================
# KEEP ONLY TAIR CIRCADIAN GENES
# ==========================================================

gene_cols = [
    g for g in common_genes
    if g in circadian_genes
]

print("Circadian genes found:", len(gene_cols))

gse115583 = gse115583[gene_cols].copy()
gse137732_ld = gse137732_ld[gene_cols].copy()
gse137732_ll = gse137732_ll[gene_cols].copy()

# ==========================================================
# ADD CT LABELS
# ==========================================================

gse115583["CT"] = np.repeat(
    [2,4,6,8,10,12,14,16,18,20,22,24],
    14
)

ct_137732 = [2,5,8,11,14,17,20,23] * 2

gse137732_ld["CT"] = ct_137732
gse137732_ll["CT"] = ct_137732

# ==========================================================
# MERGE DATASETS
# ==========================================================

all_df = pd.concat([
    gse115583,
    gse137732_ld,
    gse137732_ll
])

all_df["condition"] = (
    ["GSE115583"] * len(gse115583)
    + ["LD"] * len(gse137732_ld)
    + ["LL"] * len(gse137732_ll)
)

# ==========================================================
# LOG2 TRANSFORM
# ==========================================================

all_df[gene_cols] = np.log2(
    all_df[gene_cols] + 1
)

# ==========================================================
# REMOVE ALL-ZERO GENES
# ==========================================================

expr = all_df[gene_cols]

keep = expr.columns[
    (expr != 0).any(axis=0)
]

gene_cols = list(keep)

# ==========================================================
# REMOVE CONSTANT GENES
# ==========================================================

expr = all_df[gene_cols]

keep = expr.columns[
    expr.nunique() > 1
]

gene_cols = list(keep)

all_df = all_df[
    gene_cols + ["CT", "condition"]
]

print("Genes after filtering:", len(gene_cols))

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_df, test_df = train_test_split(
    all_df,
    test_size=0.20,
    random_state=42,
    stratify=all_df["CT"]
)

# ==========================================================
# EXTRACT DATA
# ==========================================================

X_train_raw = train_df[gene_cols]
X_test_raw = test_df[gene_cols]

y_train = train_df["CT"].values
y_test = test_df["CT"].values

# ==========================================================
# COMBAT
# ==========================================================

combat = Combat()

X_train = combat.fit_transform(
    X_train_raw.values,
    train_df["condition"].values
)

try:
    X_test = combat.transform(
        X_test_raw.values,
        test_df["condition"].values
    )
except:
    print(
        "Combat transform unavailable. "
        "Using uncorrected test set."
    )
    X_test = X_test_raw.values

# X_train1 = combat.fit_transform(
#     X_train_raw.values,
#     train_df["condition"].values
# )

# X_train2 = combat.fit_transform(
#     X_train_raw.values,
#     train_df["condition"].values
# )

# print(np.max(np.abs(X_train1 - X_train2)))

# ==========================================================
# MUTUAL INFORMATION
# ==========================================================

mi = mutual_info_regression(
    X_train,
    y_train,
    random_state=42
)

mi_series = pd.Series(
    mi,
    index=gene_cols
)

top_genes = (
    mi_series
    .sort_values(ascending=False)
    .head(30)
    .index
)

print("\nSelected genes:")
print(list(top_genes))

gene_index = [
    gene_cols.index(g)
    for g in top_genes
]

X_train = X_train[:, gene_index]
X_test = X_test[:, gene_index]

# ==========================================================
# SCALE
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# SIN/COS TARGETS
# ==========================================================

y_train_sin = np.sin(
    2*np.pi*y_train/24
)

y_train_cos = np.cos(
    2*np.pi*y_train/24
)

Y_train = np.stack(
    [y_train_sin, y_train_cos],
    axis=1
)

# ==========================================================
# DATASET
# ==========================================================

g = torch.Generator()
g.manual_seed(SEED)


train_ds = TensorDataset(
    torch.tensor(
        X_train,
        dtype=torch.float32
    ),
    torch.tensor(
        Y_train,
        dtype=torch.float32
    )
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    generator=g
)

test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

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

        z = self.net(x)

        return self.head(z)

# ==========================================================
# TRAIN
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(SEED)

model = ChronoNet(
    X_train.shape[1]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()

for epoch in range(50):

    model.train()

    running_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        pred = model(xb)

        loss = loss_fn(
            pred,
            yb
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss {running_loss:.4f}"
    )

# ==========================================================
# PREDICT
# ==========================================================

model.eval()

with torch.no_grad():

    pred = model(
        test_tensor.to(device)
    ).cpu().numpy()

sin_pred = pred[:,0]
cos_pred = pred[:,1]

norm = np.sqrt(
    sin_pred**2 +
    cos_pred**2
) + 1e-8

sin_pred /= norm
cos_pred /= norm

ct_pred = (
    np.arctan2(
        sin_pred,
        cos_pred
    )
    * 24
    / (2*np.pi)
) % 24

# ==========================================================
# CIRCULAR ERROR
# ==========================================================

errors = np.abs(
    y_test - ct_pred
)

errors = np.minimum(
    errors,
    24 - errors
)

mae = np.mean(errors)
medae = np.median(errors)

# Circular RMSE
rmse = np.sqrt(np.mean(errors**2))

# Standard deviation of errors themselves
std_error = np.std(errors, ddof=1)

# pearson_r = pearsonr(y_test, ct_pred)[0]
true_angle = 2*np.pi*y_test/24
pred_angle = 2*np.pi*ct_pred/24

pearson_r = pearsonr(
    np.cos(true_angle),
    np.cos(pred_angle)
)[0]

print("\n========== TEST SET ==========")

print(f"Circular MAE  : {mae:.4f} h")

print(f"Circular MedAE: {medae:.4f} h")

print(f"Circular RMSE : {rmse:.4f} h")

print(f"Error Std     : {std_error:.4f} h")

print(f"Pearson r      : {pearson_r:.4f}")

# ==========================================================
# SAMPLE PREDICTIONS
# ==========================================================

print("\nPredictions")

for i in range(min(10, len(y_test))):

    print(
        f"True={y_test[i]:.1f}  "
        f"Pred={ct_pred[i]:.2f}  "
        f"Err={errors[i]:.2f}"
    )

# 64-32-2
# Circular MAE: 0.3151000699028373
# Circular MedAE: 0.15246212482452393
# with dropout:0.2
# Circular MAE: 0.33773322980850934
# Circular MedAE: 0.18640518188476562

# 32-16-2
# Circular MAE: 0.3303252661600709
# Circular MedAE: 0.1715688705444336
# with dropout:0.2
# Circular MAE: 0.3426683079858776
# Circular MedAE: 0.15397155284881592
# with dropout:0.3
# Circular MAE: 0.37800354361534116
# Circular MedAE: 0.17047762870788574
