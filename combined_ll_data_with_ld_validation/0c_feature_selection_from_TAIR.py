import pandas as pd

# Load files

tair_file = "gene_results_2026-06-03.tsv"
expr_file = "combined_AGI_dataset.csv"

tair_df = pd.read_csv(tair_file, sep="\t")
expr_df = pd.read_csv(expr_file)

# 10 core circadian clock genes - from research papers to cross verify with TAIR list and ensure they are included in final dataset

core_clock_genes = [
    "AT1G01060",  # LHY
    "AT2G46830",  # CCA1
    "AT5G61380",  # TOC1
    "AT5G02810",  # PRR7
    "AT5G24470",  # PRR5
    "AT1G22770",  # GI
    "AT2G25930",  # ELF3
    "AT2G40080",  # ELF4
    "AT3G46640",  # LUX
    "AT3G09600"   # RVE8
]

# TAIR circadian genes

tair_genes = tair_df["Locus"].astype(str).str.strip().tolist()

# Combine both lists without duplicates

combined_genes = list(set(tair_genes + core_clock_genes))

print("Total combined genes:", len(combined_genes))

# Keeping only genes present in dataset

dataset_genes = expr_df.columns.tolist()
print("Total genes in dataset:", len(dataset_genes))

final_genes = [g for g in combined_genes if g in dataset_genes]

print("Genes found in dataset:", len(final_genes))


# keep first sample column also
filtered_df = expr_df[["SampleID"] + final_genes]

print(filtered_df.shape)

filtered_df.to_csv("combined_circadian_core_plus_tair.csv", index=False)

print("Saved: combined_circadian_core_plus_tair.csv")