import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# ==========================================================
# LOAD DATA
# ==========================================================

gse = pd.read_csv(
    "full_expression_matrix.csv",
    index_col=0
)

# ==========================================================
# FIX GENE IDS
# ==========================================================

gse.columns = (
    gse.columns
    .astype(str)
    .str.replace(r"\.\d+$", "", regex=True)
)

# Merge transcript isoforms
gse = gse.T.groupby(level=0).mean().T

print("Shape after gene correction:", gse.shape)

# ==========================================================
# KEEP PURE COL SAMPLES
# ==========================================================

# col_mask = (
#     gse.index.str.contains("Col_", na=False)
#     & ~gse.index.str.contains("x", na=False)
# )

# gse = gse[col_mask].copy()

# print("After Col-only filter:", gse.shape)

# ==========================================================
# LOG2 TRANSFORM
# ==========================================================

gse = np.log2(gse + 1)

# ==========================================================
# CHECK GENE AVAILABILITY
# ==========================================================

present_genes = [
    g for g in top_genes
    if g in gse.columns
]

missing_genes = [
    g for g in top_genes
    if g not in gse.columns
]

print("\nTop genes expected :", len(top_genes))
print("Top genes present  :", len(present_genes))
print("Top genes missing  :", len(missing_genes))

if len(missing_genes) > 0:
    print("\nMissing genes:")
    print(missing_genes)

# ==========================================================
# ALIGN TO MODEL FEATURES
# ==========================================================

gse = gse.reindex(
    columns=top_genes
)

assert list(gse.columns) == list(top_genes)

print("Final model input shape:", gse.shape)

# ==========================================================
# TRUE LABELS
# ==========================================================

y_true = np.array([
    0,0,0,0,
    6,6,6,6,
    15,15,15,15
])

X_test = scaler.transform(
    gse.values
)

# ==========================================================
# PREDICT
# ==========================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

model.eval()

with torch.no_grad():

    pred = model(
        X_test_tensor
    ).cpu().numpy()

print("\nRaw network output:")
print(pred)

# ==========================================================
# SIN/COS -> CT
# ==========================================================

sin_pred = pred[:, 0]
cos_pred = pred[:, 1]

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

print("\nPredicted CT:")
print(np.round(ct_pred, 2))

# ==========================================================
# CIRCULAR ERROR
# ==========================================================

errors = np.abs(
    y_true - ct_pred
)

errors = np.minimum(
    errors,
    24 - errors
)

mae = np.mean(errors)
medae = np.median(errors)
rmse = np.sqrt(np.mean(errors**2))
std_error = np.std(errors, ddof=1)

# ==========================================================
# CIRCULAR PEARSON
# ==========================================================

true_angle = 2*np.pi*y_true/24
pred_angle = 2*np.pi*ct_pred/24

pearson_r = pearsonr(
    np.cos(true_angle),
    np.cos(pred_angle)
)[0]

# ==========================================================
# RESULTS TABLE
# ==========================================================

results = pd.DataFrame({
    "Sample": gse.index,
    "True_CT": y_true,
    "Pred_CT": np.round(ct_pred, 2),
    "Error": np.round(errors, 2)
})

print("\n========== RESULTS ==========")
print(results)

print("\n========== METRICS ==========")
print(f"Circular MAE   : {mae:.3f} h")
print(f"Circular MedAE : {medae:.3f} h")
print(f"Circular RMSE  : {rmse:.3f} h")
print(f"Error Std      : {std_error:.3f} h")
print(f"Pearson r      : {pearson_r:.3f}")