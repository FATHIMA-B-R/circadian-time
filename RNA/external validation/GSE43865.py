import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr


# ==========================================================
# LOAD EXTERNAL DATA
# ==========================================================

gse43865 = pd.read_csv(
    "GSE43865_Col_only.csv", # removed denti samples from GSE43865
    index_col=0
)

print("Raw shape:", gse43865.shape)

# ==========================================================
# LOG2 TRANSFORM
# ==========================================================

gse43865 = np.log2(
    gse43865 + 1
)

# ==========================================================
# ALIGN TO TRAINING GENES
# ==========================================================

gse43865 = gse43865.reindex(
    columns=top_genes,
    fill_value=0
)

print("Aligned shape:", gse43865.shape)

# ==========================================================
# CHECK MISSING GENES
# ==========================================================

missing = [
    g for g in top_genes
    if g not in gse43865.columns
]

print("Missing genes:", len(missing))

# ==========================================================
# TRUE CT LABELS
# ==========================================================

y_true = np.array([
    2,2,2,
    6,6,6,
    10,10,10,
    14,14,14,
    18,18,18,
    22,22,22
])

# ==========================================================
# SCALE USING TRAINING SCALER
# ==========================================================

X_test = scaler.transform(
    gse43865.values
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
pearson_r = pearsonr(y_true, ct_pred)[0]

print("\n========== GSE43865 ==========")
print("Circular MAE  :", round(mae,3))
print("Circular MedAE:", round(medae,3))
print("Pearson r:", round(pearson_r,3))
print("================================")


# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame({
    "Sample": gse43865.index,
    "True_CT": y_true,
    "Pred_CT": np.round(ct_pred,2),
    "Error": np.round(errors,2)
})

print(results)

results.to_csv(
    "GSE43865_predictions.csv",
    index=False
)

print("\nSaved: GSE43865_predictions.csv")

# ========== GSE43865 ==========
# Circular MAE  : 0.924
# Circular MedAE: 0.738
# ================================


# ========== GSE43865 ==========
# Circular MAE  : 0.501
# Circular MedAE: 0.365
# ================================0.2
# Circular MAE  : 0.451
# Circular MedAE: 0.377
# ========== GSE43865 ==========0.3
# Circular MAE  : 0.352
# Circular MedAE: 0.236
# ================================