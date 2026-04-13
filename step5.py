# Step 5 — Feature Engineering
# ============================================================
# Goal:
# - Create meaningful engineered features
# - Improve representation of pavement behavior
# - Prepare dataset for modeling (X and y)
# - Save enhanced dataset
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = "pci_cleaned_data.csv"
OUTPUT_FILE = "pci_feature_engineered_data.csv"

TARGET_COLUMN = "pci"
ID_COLUMN = "section_id"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ------------------------------------------------------------
# 1) Load cleaned dataset
# ------------------------------------------------------------

print_header("1) Loading cleaned dataset")

df = pd.read_csv(INPUT_FILE)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumns:")
print(list(df.columns))


# ------------------------------------------------------------
# 2) Create Traffic Load Index
# ------------------------------------------------------------

print_header("2) Creating Traffic Load Index")

if "aadt" in df.columns and "heavy_vehicle_percentage" in df.columns:
    df["traffic_load_index"] = df["aadt"] * (df["heavy_vehicle_percentage"] / 100.0)
    print("Feature 'traffic_load_index' created successfully.")
else:
    print("Warning: Required columns for Traffic Load Index not found.")

print(df[["aadt", "heavy_vehicle_percentage", "traffic_load_index"]].head().to_string(index=False)
      if "traffic_load_index" in df.columns else "Feature not created.")


# ------------------------------------------------------------
# 3) Maintenance Gap (already exists logically)
# ------------------------------------------------------------

print_header("3) Validating Maintenance Gap")

if "years_since_last_maintenance" in df.columns:
    df["maintenance_gap"] = df["years_since_last_maintenance"]
    print("Feature 'maintenance_gap' created (copied from years_since_last_maintenance).")
else:
    print("Warning: 'years_since_last_maintenance' not found.")

if "maintenance_gap" in df.columns:
    print(df[["years_since_last_maintenance", "maintenance_gap"]].head().to_string(index=False))


# ------------------------------------------------------------
# 4) Optional log transformation (for skewed data)
# ------------------------------------------------------------

print_header("4) Optional transformation for skewed features")

skewed_features = ["aadt", "annual_rainfall_mm"]

for col in skewed_features:
    if col in df.columns:
        # Check skewness
        skewness = df[col].skew()
        print(f"{col} skewness: {skewness:.3f}")

        if abs(skewness) > 1:
            df[f"log_{col}"] = np.log1p(df[col])
            print(f"Applied log transformation: log_{col}")
        else:
            print(f"No transformation applied for {col}")


# ------------------------------------------------------------
# 5) Correlation check after feature engineering
# ------------------------------------------------------------

print_header("5) Correlation check with target")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET_COLUMN in numeric_cols:
    corr = df[numeric_cols].corr()[TARGET_COLUMN].sort_values(ascending=False)
    print("Correlation with PCI:")
    print(corr.to_string())


# ------------------------------------------------------------
# 6) Feature selection preparation
# ------------------------------------------------------------

print_header("6) Preparing feature set (X) and target (y)")

# Drop ID column if exists
feature_cols = [col for col in df.columns if col not in [TARGET_COLUMN, ID_COLUMN]]

X = df[feature_cols]
y = df[TARGET_COLUMN]

print(f"Number of features: {len(feature_cols)}")
print("Feature columns:")
print(feature_cols)

print("\nTarget preview:")
print(y.head().to_string(index=False))


# ------------------------------------------------------------
# 7) Final dataset preview
# ------------------------------------------------------------

print_header("7) Final dataset preview")

print(df.head().to_string(index=False))
print(f"\nFinal dataset shape: {df.shape}")


# ------------------------------------------------------------
# 8) Save engineered dataset
# ------------------------------------------------------------

print_header("8) Saving feature engineered dataset")

df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved dataset: {OUTPUT_FILE}")


# ------------------------------------------------------------
# 9) Summary
# ------------------------------------------------------------

print_header("9) Feature Engineering Summary")

created_features = []

if "traffic_load_index" in df.columns:
    created_features.append("traffic_load_index")

if "maintenance_gap" in df.columns:
    created_features.append("maintenance_gap")

log_features = [col for col in df.columns if col.startswith("log_")]

print(f"Engineered features: {created_features}")
print(f"Log-transformed features: {log_features if log_features else 'None'}")
print(f"Total features ready for modeling: {len(feature_cols)}")

print("\nNext Step: Step 6 — Data Preparation (Train-Test Split & Scaling)")