import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# =====================================
# Load data
# =====================================
df = pd.read_csv("combined_circadian_core_plus_tair.csv")

# =====================================
# Circadian labels
# =====================================

ct_5612 = [2,6,10,14,18,22,2,6,10,14,18,22,2]

ct_8365 = [0,4,8,12,16,20,0,4,8,12,16,20]

ct = np.array(ct_5612 + ct_8365)

print(len(ct))

df["CT"] = ct

X = df.drop(columns=["SampleID", "CT"])
y = df["CT"].values

print("Original shape:", X.shape)
# print("sample targets", y[:15])
# print("Sample data", X.head())


# =====================================
# Variance filtering
# =====================================
variances = X.var()

top_genes = variances.sort_values(ascending=False).head(15).index
X = X[top_genes]

print("\nSelected genes:", list(top_genes))
print("Shape after filtering:", X.shape)

# =====================================
# Circular encoding
# =====================================
y_xy = np.column_stack([
    np.cos(2 * np.pi * y / 24),
    np.sin(2 * np.pi * y / 24)
])

# =====================================
# Model
# =====================================
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=3,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

loo = LeaveOneOut()

# =====================================
# helpers
# =====================================
def xy_to_ct(xy):
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return (angle * 24 / (2 * np.pi)) % 24

def circular_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 24 - diff)
    return np.mean(diff)
def circular_rmse(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 24 - diff)
    return np.sqrt(np.mean(diff**2))

# print("Circular MAE :", circular_mae(true_ct, pred_ct))
# print("Circular RMSE:", circular_rmse(true_ct, pred_ct))

# =====================================
# storage
# =====================================
true_ct = []
pred_ct = []

train_mae_list = []
test_mae_list = []
gap_list = []

# =====================================
# LOOCV LOOP
# =====================================
for train_idx, test_idx in loo.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train on circular space
    rf.fit(X_train, y_xy[train_idx])

    # predictions in XY
    train_pred_xy = rf.predict(X_train)
    test_pred_xy = rf.predict(X_test)

    # normalize
    train_pred_xy = train_pred_xy / np.linalg.norm(train_pred_xy, axis=1, keepdims=True)
    test_pred_xy = test_pred_xy / np.linalg.norm(test_pred_xy, axis=1, keepdims=True)

    # convert to CT
    train_pred_ct = xy_to_ct(train_pred_xy)
    test_pred_ct = xy_to_ct(test_pred_xy)

    # store final predictions
    true_ct.append(y_test[0])
    pred_ct.append(test_pred_ct[0])
    results = pd.DataFrame({
    "True_CT": true_ct,
    "Pred_CT": pred_ct
})

    # print(results.sort_values("True_CT"))

    # circular MAE per fold
    train_mae = circular_mae(y_train, train_pred_ct)
    test_mae = circular_mae(y_test, test_pred_ct)

    train_mae_list.append(train_mae)
    test_mae_list.append(test_mae)
    gap_list.append(test_mae - train_mae)

# =====================================
# FINAL METRICS
# =====================================
true_ct = np.array(true_ct)
pred_ct = np.array(pred_ct)

print("\n===== TEST RESULTS =====")

print("Circular MAE :", circular_mae(true_ct, pred_ct))
print("Circular RMSE:", circular_rmse(true_ct, pred_ct))
print("Pearson R    :", pearsonr(true_ct, pred_ct)[0])

print("\n===== OVERFITTING GAP =====")
print("Train MAE:", np.mean(train_mae_list), "±", np.std(train_mae_list))
print("Test MAE :", np.mean(test_mae_list), "±", np.std(test_mae_list))
print("Gap      :", np.mean(gap_list), "±", np.std(gap_list))
