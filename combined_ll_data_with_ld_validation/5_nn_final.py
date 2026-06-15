import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

# =====================================
# REPRODUCIBILITY
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

# =====================================
# FEATURE SELECTION
# =====================================
variances = X.var()
top_genes = variances.sort_values(ascending=False).head(15).index
X = X[top_genes]

print("Selected genes:", list(top_genes))

# =====================================
# CYCLICAL ENCODING
# =====================================
def ct_to_xy(ct):
    angle = 2 * np.pi * ct / 24
    return np.column_stack([np.cos(angle), np.sin(angle)])

def xy_to_ct(xy):
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return (angle * 24 / (2 * np.pi)) % 24

y_xy = ct_to_xy(y)

# =====================================
# SCALER
# =====================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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

model = build_model(X_scaled.shape[1])

# =====================================
# TRAIN FINAL MODEL
# =====================================
early_stop = EarlyStopping(
    monitor="loss",
    patience=30,
    restore_best_weights=True
)

model.fit(
    X_scaled,
    y_xy,
    epochs=200,
    batch_size=2,
    verbose=1,
    callbacks=[early_stop]
)

# =====================================
# =====================================
model.save("final_circadian_model.h5")

import joblib
# joblib.dump(model, "final_circadian_model.pkl")
joblib.dump(top_genes.tolist(), "top_genes.pkl")

print("Final model saved")