## Step 3 — Data Cleaning

### Objective

Prepare the dataset for analysis and modeling by addressing data quality issues such as missing values, incorrect data types, duplicates, and outliers. The goal is to ensure that the dataset is **consistent, reliable, and suitable for machine learning**.

---

### Why Data Cleaning Matters

Machine learning models are highly sensitive to data quality. Poor data leads to:

- Biased or inaccurate predictions
- Misleading model evaluation
- Incorrect engineering conclusions

In civil engineering applications, this is critical because decisions (e.g., maintenance planning) depend directly on the results.

---

### 3.1 Removing Duplicate Records

Duplicate rows can occur due to repeated data entry or system errors.

**Action:**

- Detect duplicates using `.duplicated()`
- Remove them to avoid bias in the model

**Engineering Insight:**
Duplicate road segments would artificially increase the importance of certain conditions, leading to misleading predictions.

---

### 3.2 Converting Data Types

Some columns may be stored as text even though they represent numbers (e.g., "10000", "25%").

**Action:**

- Identify object (text) columns
- Convert numeric-like columns to proper numeric types

**Engineering Insight:**
Machine learning models require numerical input. Incorrect data types can silently break model performance.

---

### 3.3 Validating the Target Variable (PCI)

The Pavement Condition Index (PCI) must lie within a valid engineering range.

**Expected Range:**

- PCI ∈ [0, 100]

**Action:**

- Detect values outside this range
- Clip or correct invalid values

**Engineering Insight:**
PCI is a standardized index. Values outside 0–100 indicate data errors and must be corrected before modeling.

---

### 3.4 Handling Missing Values

Missing data is common in real-world engineering datasets.

**Action:**

- For numerical columns → fill with **median**
- For categorical columns → fill with **mode** or "Unknown"

**Why Median?**

- Robust to outliers
- Preserves distribution better than mean

**Engineering Insight:**
Removing rows with missing values may lead to loss of important data, especially when datasets are small.

---

### 3.5 Outlier Detection and Treatment (IQR Method)

Outliers are extreme values that may result from:

- Measurement errors
- Data entry mistakes
- Rare but real engineering events

**Method Used: Interquartile Range (IQR)**

- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 − Q1

**Outlier Bounds:**

- Lower = Q1 − 1.5 × IQR
- Upper = Q3 + 1.5 × IQR

**Action:**

- Detect outliers using IQR
- Cap values within bounds (instead of removing rows)

**Engineering Insight:**
In civil engineering, extreme values can represent real failures (e.g., very high traffic or severe distress).  
Therefore, **capping is preferred over deletion** to retain information.

---

### 3.6 Final Consistency Check

After cleaning, verify that:

- No missing values remain
- Data types are correct
- Target variable is valid
- Dataset shape is consistent

**Engineering Insight:**
This step ensures the dataset is ready for analysis and prevents errors in later stages.

---

### 3.7 Saving Cleaned Data

Save the processed dataset for reuse in later steps.

**Outputs:**

- Cleaned dataset (`.csv`)
- Cleaning audit report

**Engineering Insight:**
Reproducibility is essential in research. Saving intermediate results ensures that experiments can be repeated and verified.

---

### Summary

At the end of this step, the dataset should be:

- Clean and consistent
- Free of duplicates and missing values
- Properly formatted
- Ready for exploratory data analysis and modeling

This step forms the foundation for all subsequent machine learning work.
