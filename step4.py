# Step 4 — Exploratory Data Analysis (EDA)
# ============================================================
# Goal:
# - Load the cleaned dataset
# - Inspect target distribution
# - Visualize feature distributions
# - Examine relationships with PCI
# - Compute correlations
# - Save plots for later reporting
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INPUT_FILE = "pci_cleaned_data.csv"
OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_COLUMN = "pci"
ID_COLUMN = "section_id"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

sns.set_style("whitegrid")


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_current_plot(filename):
    filepath = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filepath}")
    plt.close()


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

print(f"Loaded cleaned dataset with shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn names:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nData types:")
print(df.dtypes)

print("\nSummary statistics:")
print(df.describe().to_string())


# ------------------------------------------------------------
# 2) Basic checks
# ------------------------------------------------------------

print_header("2) Basic quality checks")

print("Missing values per column:")
missing = df.isna().sum()
print(missing.to_string())

duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

print(f"\nTarget variable '{TARGET_COLUMN}' summary:")
print(df[TARGET_COLUMN].describe().to_string())


# ------------------------------------------------------------
# 3) Target distribution
# ------------------------------------------------------------

print_header("3) Target distribution analysis")

plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET_COLUMN], bins=30, kde=True)
plt.title("Distribution of PCI")
plt.xlabel("PCI")
plt.ylabel("Frequency")
save_current_plot("pci_distribution.png")

print("Target distribution plot created.")
print("Interpretation note: check whether PCI values are spread across the full range or concentrated in one region.")


# ------------------------------------------------------------
# 4) Feature distribution plots
# ------------------------------------------------------------

print_header("4) Feature distribution plots")

candidate_features = [
    col for col in df.columns
    if col not in [TARGET_COLUMN, ID_COLUMN]
    and pd.api.types.is_numeric_dtype(df[col])
]

print("Numeric feature columns identified:")
print(candidate_features)

for col in candidate_features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    save_current_plot(f"distribution_{col}.png")

print("\nAll feature distribution plots saved.")


# ------------------------------------------------------------
# 5) Feature vs PCI scatter plots
# ------------------------------------------------------------

print_header("5) Relationship between each feature and PCI")

for col in candidate_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[col], y=df[TARGET_COLUMN], alpha=0.7)
    plt.title(f"{col} vs PCI")
    plt.xlabel(col)
    plt.ylabel("PCI")
    save_current_plot(f"scatter_{col}_vs_pci.png")

print("\nAll scatter plots saved.")


# ------------------------------------------------------------
# 6) Correlation matrix
# ------------------------------------------------------------

print_header("6) Correlation analysis")

numeric_df = df[[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]].copy()
corr_matrix = numeric_df.corr()

print("Correlation with PCI:")
pci_corr = corr_matrix[TARGET_COLUMN].sort_values(ascending=False)
print(pci_corr.to_string())

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=False, linewidths=0.5)
plt.title("Correlation Heatmap")
save_current_plot("correlation_heatmap.png")

print("\nCorrelation heatmap saved.")


# ------------------------------------------------------------
# 7) Strongly correlated feature pairs
# ------------------------------------------------------------

print_header("7) Strong feature-to-feature correlation check")

threshold = 0.75
strong_pairs = []

cols = corr_matrix.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        value = corr_matrix.iloc[i, j]
        if abs(value) >= threshold and cols[i] != TARGET_COLUMN and cols[j] != TARGET_COLUMN:
            strong_pairs.append((cols[i], cols[j], value))

if strong_pairs:
    print("Highly correlated feature pairs (|corr| >= 0.75):")
    for a, b, v in strong_pairs:
        print(f"- {a} vs {b}: {v:.3f}")
else:
    print("No highly correlated feature pairs found above the threshold.")


# ------------------------------------------------------------
# 8) Simple engineering interpretation helper
# ------------------------------------------------------------

print_header("8) Quick engineering interpretation guide")

interpretation_rules = {
    "pavement_age_years": "Older pavement is expected to have lower PCI.",
    "aadt": "Higher traffic volume may accelerate deterioration.",
    "heavy_vehicle_percentage": "More heavy vehicles usually reduce PCI faster.",
    "pavement_thickness_mm": "Thicker pavement should generally support better performance.",
    "subgrade_cbr": "Stronger subgrade usually supports higher PCI.",
    "annual_rainfall_mm": "Higher rainfall may contribute to faster deterioration.",
    "mean_temperature_c": "Temperature may affect pavement distress depending on the region.",
    "years_since_last_maintenance": "Longer maintenance gaps usually reduce PCI.",
    "distress_density": "More distress should correspond to lower PCI."
}

for feature, note in interpretation_rules.items():
    if feature in df.columns:
        corr_value = corr_matrix.loc[feature, TARGET_COLUMN] if feature in corr_matrix.index else np.nan
        print(f"- {feature}: correlation with PCI = {corr_value:.3f} | {note}")


# ------------------------------------------------------------
# 9) Save correlation table
# ------------------------------------------------------------

print_header("9) Saving correlation table")

corr_output = OUTPUT_DIR / "correlation_matrix.csv"
corr_matrix.to_csv(corr_output)
print(f"Correlation matrix saved to: {corr_output}")


# ------------------------------------------------------------
# 10) Final EDA summary
# ------------------------------------------------------------

print_header("10) EDA summary")

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Numeric feature columns analyzed: {len(candidate_features)}")
print(f"Strong feature-feature correlation pairs found: {len(strong_pairs)}")
print(f"Plots saved in folder: {OUTPUT_DIR.resolve()}")
