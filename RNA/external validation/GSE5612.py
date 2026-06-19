import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# ==========================================================
# LOAD MICROARRAY DATA
# ==========================================================

gse = pd.read_csv(
    "GSE5612_AGI_only.csv",
    index_col=0
)

print("Raw shape:", gse.shape)

# ==========================================================
# LOG2 TRANSFORM
# ==========================================================

gse = np.log2(gse + 1)

# ==========================================================
# GENE AVAILABILITY CHECK
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
# RECOVER TRAINING GENE MEANS
# ==========================================================

train_top = pd.DataFrame(
    scaler.inverse_transform(X_train),
    columns=top_genes
)

train_gene_means = train_top.mean()

# ==========================================================
# IMPUTE MISSING GENES
# ==========================================================

gse_proc = gse.copy()

for g in top_genes:

    if g not in gse_proc.columns:

        gse_proc[g] = train_gene_means[g]

# ==========================================================
# GENE-WISE CENTERING
# ==========================================================

for g in top_genes:

    gse_proc[g] = (
        gse_proc[g]
        - gse_proc[g].mean()
        + train_gene_means[g]
    )

# ==========================================================
# FINAL FEATURE ORDER
# ==========================================================

gse_proc = gse_proc[top_genes]

print(
    "\nFinal model input shape:",
    gse_proc.shape
)

# ==========================================================
# TRUE CT LABELS
# ==========================================================

hours = [
    26,30,34,38,42,46,50,
    54,58,62,66,70,74
]

y_true = np.array(
    [h % 24 for h in hours]
)

print("\nTrue CT:")
print(y_true)

# ==========================================================
# SCALE USING TRAINING SCALER
# ==========================================================

X_test = scaler.transform(
    gse_proc.values
)

print("\nScaled mean :", X_test.mean())
print("Scaled std  :", X_test.std())

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

# print("\nRaw network output:")
# print(pred)

# ==========================================================
# SIN/COS -> CT
# ==========================================================

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

print("\nPredicted CT:")
print(np.round(ct_pred,2))

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

true_angle = (
    2*np.pi*y_true/24
)

pred_angle = (
    2*np.pi*ct_pred/24
)

pearson_r = pearsonr(
    np.cos(true_angle),
    np.cos(pred_angle)
)[0]

# ==========================================================
# RESULTS TABLE
# ==========================================================

results = pd.DataFrame({
    "Sample": gse_proc.index,
    "Hour": hours,
    "True_CT": y_true,
    "Pred_CT": np.round(ct_pred,2),
    "Error": np.round(errors,2)
})

print("\n========== RESULTS ==========")
print(results)

print("\n========== METRICS ==========")

print(f"Circular MAE   : {mae:.3f} h")
print(f"Circular MedAE : {medae:.3f} h")
print(f"Circular RMSE  : {rmse:.3f} h")
print(f"Error Std      : {std_error:.3f} h")
print(f"Pearson r      : {pearson_r:.3f}")