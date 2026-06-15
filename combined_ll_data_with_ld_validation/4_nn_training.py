import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from scipy.stats import pearsonr

# =====================================
# Reproducibility
# =====================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("combined_circadian_core_plus_tair.csv")

ct_5612 = [2,6,10,14,18,22,2,6,10,14,18,22,2]
ct_8365 = [0,4,8,12,16,20,0,4,8,12,16,20]

ct = np.array(ct_5612 + ct_8365)
df["CT"] = ct

X = df.drop(columns=["SampleID", "CT"])
y = df["CT"].values

print("Original shape:", X.shape)

# =====================================
# FEATURE SELECTION
# =====================================
variances = X.var()
top_genes = variances.sort_values(ascending=False).head(15).index
X = X[top_genes]

print("\nSelected genes:", list(top_genes))
print("Shape after filtering:", X.shape)

# =====================================
# CYCLICAL TRANSFORMATION
# =====================================
def ct_to_xy(ct):
    angle = 2 * np.pi * ct / 24
    return np.column_stack([np.cos(angle), np.sin(angle)])

def xy_to_ct(xy):
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return (angle * 24 / (2 * np.pi)) % 24

def circular_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred)
    return np.mean(np.minimum(diff, 24 - diff))

y_xy = ct_to_xy(y)

# =====================================
# MODEL
# =====================================
def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(16, activation="relu"),
        Dropout(0.3),

        Dense(8, activation="relu"),
        Dropout(0.2),

        Dense(2)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    return model

# =====================================
# NORMALIZATION
# =====================================
def safe_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, where=norm != 0)

# =====================================
# LOOCV
# =====================================
loo = LeaveOneOut()

true_ct = []
pred_ct = []

train_mae_list = []
test_mae_list = []

for train_idx, test_idx in loo.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_xy, y_test_ct = y_xy[train_idx], y[test_idx]

    # scaling inside fold
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tf.keras.backend.clear_session()

    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="loss",
        patience=30,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train_xy,
        epochs=200,
        batch_size=2,
        verbose=0,
        callbacks=[early_stop]
    )

    # =====================================
    # PREDICTIONS
    # =====================================
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_pred = safe_normalize(train_pred)
    test_pred = safe_normalize(test_pred)

    train_ct_pred = xy_to_ct(train_pred)
    test_ct_pred = xy_to_ct(test_pred)

    # store final LOOCV prediction
    true_ct.append(y_test_ct[0])
    pred_ct.append(test_ct_pred[0])

    # circular MAE (IMPORTANT FIX)
    train_mae_list.append(circular_mae(y[train_idx], train_ct_pred))
    test_mae_list.append(circular_mae(y[test_idx], test_ct_pred))

    print("Train MAE:", train_mae_list[-1])
    print("Test MAE :", test_mae_list[-1])

# =====================================
# FINAL METRICS
# =====================================
true_ct = np.array(true_ct)
pred_ct = np.array(pred_ct)

print("\n===== FINAL RESULTS (MLP) =====")

print("Circular MAE :", circular_mae(true_ct, pred_ct))
print("Circular RMSE:", circular_rmse(true_ct, pred_ct))
print("Pearson R    :", pearsonr(true_ct, pred_ct)[0])

print("\n===== OVERFITTING =====")
print("Train MAE:", np.mean(train_mae_list), "±", np.std(train_mae_list))
print("Test MAE :", np.mean(test_mae_list), "±", np.std(test_mae_list))
print("Gap      :", np.mean(test_mae_list) - np.mean(train_mae_list))