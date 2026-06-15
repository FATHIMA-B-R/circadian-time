import pandas as pd
import numpy as np

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from pycombat import Combat

# LOAD DATA

gse5612 = pd.read_csv("GSE5612.csv")
gse8365 = pd.read_csv("GSE8365.csv")

print("GSE5612:", gse5612.shape)
print("GSE8365:", gse8365.shape)

# PRESERVE SAMPLE IDS
sample_col_5612 = gse5612.columns[0]
sample_col_8365 = gse8365.columns[0]

sample_ids_5612 = gse5612[sample_col_5612].copy()
sample_ids_8365 = gse8365[sample_col_8365].copy()

# remove sample-name column from expression matrix
gse5612 = gse5612.drop(columns=[sample_col_5612])
gse8365 = gse8365.drop(columns=[sample_col_8365])

# ADD CT LABELS
ct_5612 = [2,6,10,14,18,22,2,6,10,14,18,22,2]

ct_8365 = [0,4,8,12,16,20,0,4,8,12,16,20]

gse5612["CT"] = ct_5612
gse8365["CT"] = ct_8365

gse5612["Batch"] = "GSE5612"
gse8365["Batch"] = "GSE8365"

# COMMON GENES
common_genes = sorted(
    list(
        set(gse5612.columns)
        .intersection(set(gse8365.columns))
    )
)

common_genes = [
    g for g in common_genes
    if g not in ["CT", "Batch"]
]

print("Common genes:", len(common_genes))

gse5612 = gse5612[
    common_genes + ["CT", "Batch"]
]

gse8365 = gse8365[
    common_genes + ["CT", "Batch"]
]

# COMBINE

df = pd.concat(
    [gse5612, gse8365],
    ignore_index=True
)

sample_ids = pd.concat(
    [sample_ids_5612, sample_ids_8365],
    ignore_index=True
)

print("Combined shape:", df.shape)

# SPLIT

X = df.drop(columns=["CT", "Batch"])

y = df["CT"]

batch = df["Batch"]


# NUMERIC CONVERSION

X = X.apply(pd.to_numeric, errors="coerce")

X = X.dropna(axis=1, how="all")

print("Expression matrix:", X.shape)

# LOG2

max_val = X.max().max()

print("Max expression:", max_val)

if max_val > 100:
    print("Applying log2(x+1)")
    X = np.log2(X + 1)

# QUANTILE NORMALIZATION

qt = QuantileTransformer(
    output_distribution="normal",
    n_quantiles=min(25, len(X)),
    random_state=42
)

X_qn = pd.DataFrame(
    qt.fit_transform(X),
    columns=X.columns
)

# COMBAT

combat = Combat()

X_combat = combat.fit_transform(
    X_qn.values,
    batch.values
)

X_combat = pd.DataFrame(
    X_combat,
    columns=X.columns
)

# STANDARDIZATION

# scaler = StandardScaler()

# X_final = pd.DataFrame(
#     scaler.fit_transform(X_combat),
#     columns=X.columns
# )

# FINAL DATASET

final_df = X_final.copy()

final_df.insert(
    0,
    "SampleID",
    sample_ids.values
)

final_df["CT"] = y.values

final_df["Batch"] = batch.values

print("\nFinal shape:", final_df.shape)

print(final_df.iloc[:5, :8])


final_df.to_csv(
    "combined_circadian_dataset.csv",
    index=False
)

print("\nSaved: combined_circadian_dataset.csv")