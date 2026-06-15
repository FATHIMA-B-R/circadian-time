import pandas as pd
import numpy as np

# LOAD EXPRESSION DATA

print("Loading expression matrix...")

expr = pd.read_csv(
    "combined_circadian_dataset.csv"
)

print("Original shape:", expr.shape)

# SEPARATE METADATA

meta = expr[
    ["SampleID", "CT", "Batch"]
].copy()

expr = expr.drop(
    columns=["SampleID", "CT", "Batch"]
)

print("Expression only:", expr.shape)

# FORCE NUMERIC

expr = expr.apply(
    pd.to_numeric,
    errors="coerce"
)

# remove completely empty columns
expr = expr.dropna(
    axis=1,
    how="all"
)

print("After numeric conversion:", expr.shape)

# LOAD GPL198 ANNOTATION

print("\nLoading annotation...")

annot = pd.read_csv(
    "GPL198-17390.txt",
    sep="\t",
    comment="#",
    low_memory=False
)

print("Annotation shape:", annot.shape)

# KEEP PROBE + AGI

annot = annot[
    ["ID", "AGI"]
]

annot["AGI"] = (
    annot["AGI"]
    .fillna("")
    .astype(str)
    .str.split(" /// ")
    .str[0]
    .str.strip()
)

annot = annot[
    annot["AGI"] != ""
]

print("Annotated probes:", len(annot))

# PROBE -> GENE MAP

probe_to_gene = dict(
    zip(
        annot["ID"],
        annot["AGI"]
    )
)

# RENAME PROBES TO AGI

print("\nMapping probes to genes...")

new_cols = []

for col in expr.columns:

    if col in probe_to_gene:
        new_cols.append(
            probe_to_gene[col]
        )
    else:
        new_cols.append(col)

expr.columns = new_cols

# TRANSPOSE
# rows = genes
# cols = samples

expr = expr.T

print("After transpose:", expr.shape)

# KEEP ONLY AGI GENES

expr = expr[
    expr.index.str.startswith(
        "AT",
        na=False
    )
]

print("AGI rows:", expr.shape)

# COLLAPSE DUPLICATE PROBES

print("\nCollapsing duplicate probes...")

expr = expr.groupby(
    expr.index
).mean(
    numeric_only=True
)

print(
    "Unique AGI genes:",
    expr.shape
)

# TRANSPOSE BACK
# rows = samples
# cols = genes

expr = expr.T

print(
    "Final expression matrix:",
    expr.shape
)


final_df = pd.concat(
    [
        meta.reset_index(drop=True),
        expr.reset_index(drop=True)
    ],
    axis=1
)

print(
    "\nFinal dataset shape:",
    final_df.shape
)

# saving
outfile = "combined_AGI_dataset.csv"

final_df.to_csv(
    outfile,
    index=False
)

print("\nSaved:", outfile)

test = pd.read_csv(outfile)

gene_cols = [
    c for c in test.columns
    if c.startswith("AT")
]

print("\nVerification")
print("Shape:", test.shape)
print("AGI genes:", len(gene_cols))

print("\nFirst 20 genes:")
print(gene_cols[:20])

print("\nMetadata columns:")
print(
    test[
        ["SampleID", "CT", "Batch"]
    ].head()
)