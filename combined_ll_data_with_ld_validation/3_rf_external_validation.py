import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler

# =====================================================
# LOAD LD DATA
# =====================================================

ld = pd.read_csv("GSE3416_AGI_only.csv")

# =====================================================
# SAME 15 GENES USED BY RF MODEL
# =====================================================

# top_genes = [
#     'AT2G46830', 'AT5G51200', 'AT2G21070', 'AT5G64170', 'AT5G60100',
#     'AT2G23070', 'AT1G26800', 'AT3G20810', 'AT3G04910', 'AT5G62430',
#     'AT1G12910', 'AT1G22770', 'AT1G09340', 'AT4G39260', 'AT2G21320'
# ]
top_genes = joblib.load("rf_top_genes.pkl")

# =====================================================
# FEATURE ALIGNMENT
# =====================================================

missing = [g for g in top_genes if g not in ld.columns]

if len(missing) > 0:
    print("Missing genes:", missing)

X_ld = ld[top_genes].copy()

print("LD shape:", X_ld.shape)

# =====================================================
# SAME PREPROCESSING AS TRAINING
# =====================================================

# log2 transform
X_ld = np.log2(X_ld + 1)

# quantile normalization
qt = QuantileTransformer(
    output_distribution="normal",
    n_quantiles=min(len(X_ld), 18),
    random_state=42
)

X_ld = qt.fit_transform(X_ld)

# standardization
scaler = StandardScaler()

X_ld = scaler.fit_transform(X_ld)

print("\nProcessed LD data")
print("Min :", np.min(X_ld))
print("Max :", np.max(X_ld))
print("Mean:", np.mean(X_ld))
print("Std :", np.std(X_ld))

# =====================================================
# RF PREDICTION
# =====================================================

pred_xy = rf_final.predict(X_ld)

# =====================================================
# NORMALIZE TO UNIT CIRCLE
# =====================================================

norm = np.linalg.norm(pred_xy, axis=1, keepdims=True)

pred_xy = np.divide(
    pred_xy,
    norm,
    where=norm != 0
)

# =====================================================
# XY -> CT
# =====================================================

def xy_to_ct(xy):
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return (angle * 24 / (2 * np.pi)) % 24

pred_ct = xy_to_ct(pred_xy)

# =====================================================
# TRUE CT FROM GSE3416 METADATA
# 00h,04h,08h,12h,16h,20h
# each with 3 replicates
# =====================================================

true_ct = np.array(
    [0]*3 +
    [4]*3 +
    [8]*3 +
    [12]*3 +
    [16]*3 +
    [20]*3
)

# =====================================================
# CIRCULAR METRICS
# =====================================================

def circular_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 24 - diff)
    return np.mean(diff)

def circular_rmse(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 24 - diff)
    return np.sqrt(np.mean(diff**2))

print("\n===== GSE3416 RESULTS =====")

print("Circular MAE :", circular_mae(true_ct, pred_ct))
print("Circular RMSE:", circular_rmse(true_ct, pred_ct))

# Pearson only if predictions vary
if np.std(pred_ct) > 0:
    from scipy.stats import pearsonr
    print("Pearson R    :", pearsonr(true_ct, pred_ct)[0])
else:
    print("Pearson R    : constant predictions")

# =====================================================
# SAVE
# =====================================================

results = pd.DataFrame({
    "True_CT": true_ct,
    "Predicted_CT": pred_ct
})

results.to_csv("GSE3416_RF_results.csv", index=False)

print("\nSaved: GSE3416_RF_results.csv")
print(results)