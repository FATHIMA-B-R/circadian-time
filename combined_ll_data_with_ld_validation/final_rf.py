import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("combined_circadian_core_plus_tair.csv")

# =====================================
# CIRCADIAN LABELS
# =====================================
ct_5612 = [2,6,10,14,18,22,2,6,10,14,18,22,2]
ct_8365 = [0,4,8,12,16,20,0,4,8,12,16,20]

ct = np.array(ct_5612 + ct_8365)
df["CT"] = ct

X = df.drop(columns=["SampleID", "CT"])
y = df["CT"].values

# =====================================
# FEATURE SELECTION (same as LOOCV)
# =====================================
variances = X.var()
top_genes = variances.sort_values(ascending=False).head(15).index
X = X[top_genes]

print("Selected genes:", list(top_genes))

# =====================================
# CIRCULAR ENCODING (TRAIN TARGET)
# =====================================
y_xy = np.column_stack([
    np.cos(2 * np.pi * y / 24),
    np.sin(2 * np.pi * y / 24)
])

# =====================================
# SCALER (FIT ON FULL DATA)
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# FINAL RANDOM FOREST MODEL
# =====================================
rf_final = RandomForestRegressor(
    n_estimators=150,
    max_depth=3,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

rf_final.fit(X_scaled, y_xy)

print("Final RF model trained on full dataset")

joblib.dump(rf_final, "rf_final_model.pkl")
joblib.dump(scaler, "rf_scaler.pkl")
joblib.dump(top_genes.tolist(), "rf_top_genes.pkl")

print("Saved RF model, scaler, and gene list")