# Step 3 — Data Cleaning
# ============================================================
# Goal:
# - Clean the synthetic PCI dataset
# - Handle missing values
# - Remove duplicates
# - Convert numeric-like text columns
# - Detect and cap outliers using IQR
# - Save a cleaned CSV for later modeling
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = "pavement_pci_dataset.csv"      
OUTPUT_FILE = "pci_cleaned_data.csv"
AUDIT_FILE = "step3_cleaning_audit.csv"

TARGET_COLUMN = "pci"
ID_COLUMN = "section_id"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------
# 1) Load raw data
# ------------------------------------------------------------

print_header("1) Loading data")

df = pd.read_csv(INPUT_FILE)
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

print(f"Loaded dataset with shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nColumn types:")
print(df.dtypes)


# ------------------------------------------------------------
# 2) Remove duplicate rows
# ------------------------------------------------------------

print_header("2) Removing duplicate rows")

duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)

print(f"Shape after duplicate removal: {df.shape[0]} rows, {df.shape[1]} columns")


# ------------------------------------------------------------
# 3) Convert numeric-like text columns to numeric
# ------------------------------------------------------------

print_header("3) Converting numeric-like text columns")

audit_rows = []

object_cols = df.select_dtypes(include="object").columns.tolist()
converted_columns = []

for col in object_cols:
    # Try to convert text numbers safely
    cleaned = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )

    numeric_version = pd.to_numeric(cleaned, errors="coerce")

    # If at least 90% of the non-missing values become numeric, convert it
    conversion_rate = numeric_version.notna().mean()

    if conversion_rate >= 0.90:
        df[col] = numeric_version
        converted_columns.append(col)
        audit_rows.append({
            "step": "type_conversion",
            "column": col,
            "details": f"Converted text column to numeric (success rate: {conversion_rate:.2f})",
            "action": "Converted to numeric"
        })

print(f"Converted columns: {converted_columns if converted_columns else 'None'}")
print("\nCurrent column types:")
print(df.dtypes)


# ------------------------------------------------------------
# 4) Basic validation of PCI target
# ------------------------------------------------------------

print_header("4) Validating target column")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

invalid_target_mask = df[TARGET_COLUMN].notna() & ((df[TARGET_COLUMN] < 0) | (df[TARGET_COLUMN] > 100))
invalid_target_count = invalid_target_mask.sum()

print(f"Invalid PCI values outside 0–100: {invalid_target_count}")

if invalid_target_count > 0:
    # For this project, clipping is acceptable because PCI is bounded.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 100)
    audit_rows.append({
        "step": "target_validation",
        "column": TARGET_COLUMN,
        "details": f"Clipped {invalid_target_count} invalid PCI values to the 0–100 range",
        "action": "Clipped target values"
    })

print(df[TARGET_COLUMN].describe())


# ------------------------------------------------------------
# 5) Handle missing values
# ------------------------------------------------------------

print_header("5) Handling missing values")

missing_before = df.isna().sum()
print("Missing values before cleaning:")
print(missing_before[missing_before > 0].to_string())

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Fill numeric columns with median
for col in numeric_cols:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        audit_rows.append({
            "step": "missing_value_imputation",
            "column": col,
            "details": f"Filled {missing_count} missing values with median ({median_value})",
            "action": "Median imputation"
        })

# Fill categorical columns with mode if any exist
for col in categorical_cols:
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        mode_value = df[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        df[col] = df[col].fillna(fill_value)
        audit_rows.append({
            "step": "missing_value_imputation",
            "column": col,
            "details": f"Filled {missing_count} missing values with mode/Unknown",
            "action": "Mode imputation"
        })

missing_after = df.isna().sum()
print("\nMissing values after cleaning:")
print(missing_after[missing_after > 0].to_string() if (missing_after > 0).any() else "No missing values remain.")


# ------------------------------------------------------------
# 6) IQR outlier detection and capping
# ------------------------------------------------------------

print_header("6) Detecting and capping outliers with IQR")

# Do not treat ID as a feature for outlier processing
exclude_from_outliers = {ID_COLUMN}

outlier_summary = []

for col in numeric_cols:
    if col in exclude_from_outliers:
        continue

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isna(iqr):
        continue

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    below_count = (df[col] < lower_bound).sum()
    above_count = (df[col] > upper_bound).sum()
    total_outliers = int(below_count + above_count)

    if total_outliers > 0:
        # Cap values instead of dropping rows
        df[col] = df[col].clip(lower_bound, upper_bound)

        outlier_summary.append({
            "step": "outlier_capping",
            "column": col,
            "details": f"Capped {total_outliers} values using IQR bounds [{lower_bound:.3f}, {upper_bound:.3f}]",
            "action": "IQR capping"
        })

print("Outlier handling summary:")
if outlier_summary:
    for item in outlier_summary:
        print(f"- {item['column']}: {item['details']}")
else:
    print("No significant outliers found for capping.")


# ------------------------------------------------------------
# 7) Final consistency check
# ------------------------------------------------------------

print_header("7) Final consistency check")

print(f"Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFinal data types:")
print(df.dtypes)

print("\nFinal missing values:")
print(df.isna().sum().to_string())

print("\nFinal target summary:")
print(df[TARGET_COLUMN].describe().to_string())


# ------------------------------------------------------------
# 8) Save cleaned dataset
# ------------------------------------------------------------

print_header("8) Saving cleaned data")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned dataset saved to: {OUTPUT_FILE}")


# ------------------------------------------------------------
# 9) Save audit log
# ------------------------------------------------------------

print_header("9) Saving cleaning audit report")

audit_df = pd.DataFrame(audit_rows)

if audit_df.empty:
    audit_df = pd.DataFrame([{
        "step": "cleaning",
        "column": "-",
        "details": "No major cleaning actions were required",
        "action": "No action"
    }])

audit_df.to_csv(AUDIT_FILE, index=False)
print(f"Cleaning audit saved to: {AUDIT_FILE}")

print("\nAudit preview:")
print(audit_df.to_string(index=False))


# ------------------------------------------------------------
# 10) Compact final summary
# ------------------------------------------------------------

print_header("10) Cleaning summary")

summary = {
    "rows": df.shape[0],
    "columns": df.shape[1],
    "duplicate_rows_removed": int(duplicate_count),
    "converted_columns": len(converted_columns),
    "numeric_columns": len(numeric_cols),
    "categorical_columns": len(categorical_cols),
    "outlier_capped_columns": len(outlier_summary)
}

for k, v in summary.items():
    print(f"{k}: {v}")