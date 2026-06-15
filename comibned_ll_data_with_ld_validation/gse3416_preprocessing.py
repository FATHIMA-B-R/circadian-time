import pandas as pd

# =====================================================
# 1. LOAD EXPRESSION MATRIX
# =====================================================

print("Loading expression matrix...")

expr = pd.read_csv("GSE3416_series_matrix (1).csv")

print("Expression shape:", expr.shape)

# =====================================================
# 2. LOAD ANNOTATION FILE
# =====================================================

print("\nLoading annotation...")

annot = pd.read_csv(
    "GPL198-17390.txt",
    sep="\t",
    comment="#",
    low_memory=False
)

print("Annotation shape:", annot.shape)

# =====================================================
# 3. KEEP ONLY ID + AGI
# =====================================================

annot = annot[["ID", "AGI"]]

# Some probes map to multiple genes:
# AT1G01010 /// AT1G01020
# Keep first gene

annot["AGI"] = (
    annot["AGI"]
    .fillna("")
    .astype(str)
    .str.split(" /// ")
    .str[0]
    .str.strip()
)

# Remove empty AGI entries
annot = annot[annot["AGI"] != ""]

print("Annotated probes:", len(annot))

# =====================================================
# 4. CREATE PROBE -> GENE MAP
# =====================================================

probe_to_gene = dict(zip(annot["ID"], annot["AGI"]))

# =====================================================
# 5. RENAME PROBE COLUMNS TO AGI
# =====================================================

print("\nMapping probes to genes...")

new_cols = []

for col in expr.columns:

    if col in probe_to_gene:
        new_cols.append(probe_to_gene[col])
    else:
        new_cols.append(col)

expr.columns = new_cols

# =====================================================
# 6. SET SAMPLE COLUMN AS INDEX
# =====================================================

sample_col = expr.columns[0]

expr = expr.set_index(sample_col)

print("Shape after indexing:", expr.shape)

# =====================================================
# 7. TRANSPOSE
#    rows = genes
#    cols = samples
# =====================================================

expr = expr.T

print("Shape after transpose:", expr.shape)

# =====================================================
# 8. COLLAPSE DUPLICATE GENES
# =====================================================

print("\nCollapsing duplicate probes...")

expr = expr.groupby(expr.index).mean()

print("After collapsing:", expr.shape)

# =====================================================
# 9. KEEP ONLY AGI GENES
# =====================================================

expr = expr[
    expr.index.str.startswith("AT", na=False)
]

print("AGI genes only:", expr.shape)

# =====================================================
# 10. TRANSPOSE BACK
#     rows = samples
#     cols = genes
# =====================================================

expr = expr.T

print("Final shape:", expr.shape)

# =====================================================
# 11. SAVE
# =====================================================

outfile = "GSE3416_AGI_only.csv"

expr.to_csv(outfile)

print("\nSaved:", outfile)

# =====================================================
# 12. VERIFY
# =====================================================

test = pd.read_csv(outfile)

print("\nVerification")
print("Shape:", test.shape)

gene_cols = [c for c in test.columns if c.startswith("AT")]

print("Number of AGI genes:", len(gene_cols))

print("\nFirst 20 AGI genes:")
print(gene_cols[:20])