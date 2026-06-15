import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

nn_final = load_model(
    "final_circadian_model.h5",
    compile=False
)
# =====================================================
# LOAD LD DATA
# =====================================================

ld = pd.read_csv("GSE3416_AGI_only.csv")

# =====================================================
# SAME GENES USED DURING TRAINING
# =====================================================


top_genes = joblib.load("top_genes.pkl")
print("Selected genes:", list(top_genes))

# top_genes = [
#     'AT2G46830', 'AT5G51200', 'AT2G21070', 'AT5G64170',
#     'AT5G60100', 'AT2G23070', 'AT1G26800', 'AT3G20810',
#     'AT3G04910', 'AT5G62430', 'AT1G12910', 'AT1G22770',
#     'AT1G01060', 'AT5G61380', 'AT5G57360'
# ]

# =====================================================
# FEATURE ALIGNMENT
# =====================================================
missing = [g for g in top_genes if g not in ld.columns]

if len(missing) > 0:
    print("Missing genes:", missing)
X_ld = ld[top_genes].copy()

print("LD shape:", X_ld.shape)

# =====================================================
# SAME PREPROCESSING FOR RF
# =====================================================

X_ld = np.log2(X_ld + 1)

qt = QuantileTransformer(
    output_distribution="normal",
    n_quantiles=min(len(X_ld), 18),
    random_state=42
)

X_ld = qt.fit_transform(X_ld)

scaler = StandardScaler()

X_ld = scaler.fit_transform(X_ld)

print("\nProcessed LD data")
print("Min :", np.min(X_ld))
print("Max :", np.max(X_ld))
print("Mean:", np.mean(X_ld))
print("Std :", np.std(X_ld))

# =====================================================
# NN PREDICTION
# =====================================================

pred_xy = nn_final.predict(X_ld)

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
# TRUE CT FROM GSE3416 DESIGN
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
# METRICS
# =====================================================

def circular_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 24 - diff)
    return np.mean(diff)

def circular_rmse(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 24 - diff)
    return np.sqrt(np.mean(diff**2))

print("\n===== GSE3416 NN RESULTS =====")

print("Circular MAE :", circular_mae(true_ct, pred_ct))
print("Circular RMSE:", circular_rmse(true_ct, pred_ct))
print("Pearson R    :", pearsonr(true_ct, pred_ct)[0])

# =====================================================
# SAVE RESULTS
# =====================================================

results = pd.DataFrame({
    "True_CT": true_ct,
    "Predicted_CT": pred_ct
})

results.to_csv("GSE3416_NN_results.csv", index=False)

print("\nSaved: GSE3416_NN_results.csv")
print(results)
