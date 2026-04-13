# Step 2 — Data Loading and Initial Audit
# ============================================================
# Public PCI dataset source:
# CSV download from Data.gov / Montgomery County of Maryland
# PCI is described as a 0–100 pavement condition score.
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# Optional: make notebook output a bit nicer
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 120)

# ------------------------------------------------------------
# 1) Load the dataset
# ------------------------------------------------------------

CSV_URL = "https://data.montgomerycountymd.gov/api/views/g3x8-hk6f/rows.csv?accessType=DOWNLOAD"

# Option A: load directly from the public URL
# df_raw = pd.read_csv(CSV_URL)

# Option B: load from a local file instead
LOCAL_FILE = Path("pavement_pci_dataset.csv")
df_raw = pd.read_csv(LOCAL_FILE)

# Work on a copy
df = df_raw.copy()

# Standardize column names for easier handling
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
    .str.replace("/", "_", regex=False)
)

print("Dataset loaded successfully.")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------------------------------------
# 2) Quick inspection
# ------------------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nData types and non-null counts:")
df.info()

print("\nDescriptive statistics for numeric columns:")
print(df.describe())

print("\nColumn names:")
print(list(df.columns))

# ------------------------------------------------------------
# 3) Missing values check
# ------------------------------------------------------------

missing_count = df.isna().sum()
missing_pct = (missing_count / len(df) * 100).round(2)

missing_summary = (
    pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_percent": missing_pct.values,
        "dtype": df.dtypes.astype(str).values,
    })
    .sort_values(["missing_count", "column"], ascending=[False, True])
    .reset_index(drop=True)
)

print("\nMissing value summary:")
print(missing_summary)

# Save missing summary if needed
missing_summary.to_csv("step2_missing_summary.csv", index=False)

# ------------------------------------------------------------
# 4) Duplicate row check
# ------------------------------------------------------------

duplicate_rows = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_rows}")

# ------------------------------------------------------------
# 5) Identify candidate PCI column
# ------------------------------------------------------------

pci_candidates = [col for col in df.columns if "pci" in col.lower()]
print("\nPCI-like column candidates:")
print(pci_candidates)

# If exactly one PCI column exists, inspect its range
if len(pci_candidates) == 1:
    pci_col = pci_candidates[0]
    print(f"\nUsing '{pci_col}' as PCI target candidate.")
    print(df[pci_col].describe())

    invalid_pci_mask = df[pci_col].notna() & ((df[pci_col] < 0) | (df[pci_col] > 100))
    invalid_pci_count = invalid_pci_mask.sum()
    print(f"Invalid PCI values (<0 or >100): {invalid_pci_count}")

# ------------------------------------------------------------
# 6) Detect columns that may be numeric but stored as text
# ------------------------------------------------------------

possible_numeric_text_cols = []

for col in df.select_dtypes(include="object").columns:
    # Try to coerce to numeric after removing commas and extra spaces
    coerced = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )
    if coerced.notna().mean() >= 0.90:
        possible_numeric_text_cols.append(col)

print("\nPossible numeric columns stored as text:")
print(possible_numeric_text_cols)

# ------------------------------------------------------------
# 7) Check for constant or near-constant columns
# ------------------------------------------------------------

constant_cols = []
for col in df.columns:
    if df[col].nunique(dropna=True) <= 1:
        constant_cols.append(col)

print("\nConstant columns:")
print(constant_cols)

# ------------------------------------------------------------
# 8) Build an initial data audit table
# ------------------------------------------------------------

audit_rows = []

# Missing values
for col in df.columns:
    if missing_count[col] > 0:
        audit_rows.append({
            "issue_type": "missing_values",
            "column": col,
            "details": f"{missing_count[col]} missing values ({missing_pct[col]}%)",
            "recommended_action": "Decide between imputation or row removal"
        })

# Duplicate rows
if duplicate_rows > 0:
    audit_rows.append({
        "issue_type": "duplicate_rows",
        "column": "entire_dataset",
        "details": f"{duplicate_rows} duplicated rows found",
        "recommended_action": "Remove exact duplicates"
    })

# Text columns that look numeric
for col in possible_numeric_text_cols:
    audit_rows.append({
        "issue_type": "type_issue",
        "column": col,
        "details": "Column appears numeric but is stored as object/text",
        "recommended_action": "Convert to numeric type"
    })

# Constant columns
for col in constant_cols:
    audit_rows.append({
        "issue_type": "low_variance",
        "column": col,
        "details": "Column has only one unique value",
        "recommended_action": "Remove if not meaningful"
    })

# PCI range validation if a PCI column exists
if len(pci_candidates) == 1:
    pci_col = pci_candidates[0]
    invalid_pci_count = ((df[pci_col] < 0) | (df[pci_col] > 100)).sum()
    if invalid_pci_count > 0:
        audit_rows.append({
            "issue_type": "invalid_target_range",
            "column": pci_col,
            "details": f"{invalid_pci_count} values outside valid PCI range 0–100",
            "recommended_action": "Inspect and correct or remove invalid records"
        })

audit_df = pd.DataFrame(audit_rows)

print("\nInitial audit table:")
print(audit_df if not audit_df.empty else pd.DataFrame([{
    "issue_type": "none",
    "column": "-",
    "details": "No major issues detected in the initial audit",
    "recommended_action": "Proceed to EDA"
}]))

# Save audit report
audit_df.to_csv("step2_data_audit_report.csv", index=False)

# ------------------------------------------------------------
# 9) Short text summary for the notebook
# ------------------------------------------------------------

summary_text = f"""
Dataset loading completed successfully.

Rows: {df.shape[0]}
Columns: {df.shape[1]}
Duplicate rows: {duplicate_rows}
Columns with missing values: {(missing_count > 0).sum()}
Possible numeric text columns: {len(possible_numeric_text_cols)}
Constant columns: {len(constant_cols)}
PCI column candidates: {pci_candidates}
"""

print(summary_text)

# Optional: create a compact overview table
overview = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "duplicate_rows",
        "columns_with_missing_values",
        "possible_numeric_text_columns",
        "constant_columns"
    ],
    "value": [
        df.shape[0],
        df.shape[1],
        duplicate_rows,
        int((missing_count > 0).sum()),
        len(possible_numeric_text_cols),
        len(constant_cols)
    ]
})

print("\nCompact overview:")
print(overview)