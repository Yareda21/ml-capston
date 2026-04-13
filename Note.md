## Step 4 — Exploratory Data Analysis (EDA)

### Objective

Understand the cleaned pavement dataset before building machine learning models. The goal of EDA is to discover patterns, relationships, trends, and possible data issues that may affect PCI prediction.

---

### Why EDA Matters

EDA helps us answer important questions such as:

- Which variables appear to influence PCI?
- Are the relationships linear or nonlinear?
- Are there any unusual patterns or hidden issues in the data?
- Which features are likely to be useful for regression modeling?

For civil engineering, this step is important because the data must make engineering sense before any model is trusted.

---

### 4.1 Load the Cleaned Dataset

Use the cleaned file `pci_cleaned_data.csv` created in Step 3.

**Purpose:**

- Confirm that the cleaned dataset is ready for analysis
- Inspect the shape, column names, and summary statistics again

---

### 4.2 Check the Distribution of the Target Variable

The target variable is **PCI**.

**What to look for:**

- Is PCI concentrated in one range?
- Are there too many high or low values?
- Is the distribution balanced enough for regression?

**Engineering Insight:**
If most PCI values are very high or very low, the model may learn poorly and produce biased predictions.

---

### 4.3 Explore Feature Distributions

Examine the distribution of all main input features such as:

- pavement_age_years
- aadt
- heavy_vehicle_percentage
- pavement_thickness_mm
- subgrade_cbr
- annual_rainfall_mm
- mean_temperature_c
- years_since_last_maintenance
- distress_density

**What to look for:**

- Skewness
- Spread
- Extreme values
- Whether features look realistic for civil engineering data

---

### 4.4 Relationship Between Each Feature and PCI

Use scatter plots to examine how each variable relates to PCI.

**Expected patterns:**

- PCI should decrease as pavement age increases
- PCI should decrease as traffic and heavy vehicle percentage increase
- PCI should increase with greater pavement thickness and better subgrade strength
- PCI should decrease with larger maintenance gaps and higher distress density

**Engineering Insight:**
These trends should match pavement deterioration logic. If they do not, the data or feature engineering may need review.

---

### 4.5 Correlation Analysis

Compute a correlation matrix to identify:

- Strong positive relationships
- Strong negative relationships
- Possible multicollinearity among predictors

**Why this matters:**

- Linear and Ridge/Lasso models are sensitive to correlated predictors
- Correlation helps identify which features are likely important
- It helps avoid redundant variables

---

### 4.6 Detect Possible Multicollinearity

Look for pairs of features that are strongly correlated with each other.

**Examples:**

- aadt and heavy_vehicle_percentage may be related
- pavement_age_years and years_since_last_maintenance may also be related

**Engineering Insight:**
Some variables may represent similar physical behavior. This matters when interpreting regression coefficients.

---

### 4.7 Feature Engineering Check

Before modeling, confirm whether any derived features should be created later, such as:

- Traffic Load Index
- Maintenance Gap

This EDA step helps decide whether the original variables are enough or whether additional engineered features will improve model performance.

---

### 4.8 EDA Summary

At the end of this step, the student should be able to explain:

- Which variables appear most influential
- Whether the data behaves realistically
- Whether regression is appropriate
- Which features deserve special attention during modeling

---

### Expected Output from This Step

- Summary statistics
- Distribution plots
- Scatter plots
- Correlation heatmap
- Short written interpretation of the main patterns

This step prepares the dataset for feature engineering and model training.
