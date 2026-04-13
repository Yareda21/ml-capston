## Step 5 — Feature Engineering

### Objective

Enhance the dataset by creating new features that better represent real-world pavement behavior. The goal is to improve model performance by incorporating **engineering knowledge** into the data.

---

### Why Feature Engineering Matters

Raw data often does not fully capture the relationships needed for accurate prediction. Feature engineering allows us to:

- Combine variables to reflect real physical behavior
- Improve model accuracy
- Reduce noise and ambiguity
- Introduce domain-specific intelligence into the model

In civil engineering, this step is critical because many important factors (like traffic load or maintenance impact) are not directly given but must be derived.

---

### 5.1 Traffic Load Index

**Definition:**
Traffic Load Index = AADT × Heavy Vehicle Percentage

**Purpose:**

- Captures the combined effect of traffic volume and heavy vehicles
- Heavy vehicles contribute significantly more to pavement damage

**Engineering Insight:**
Two roads with the same AADT may deteriorate differently if one has more trucks. This feature captures that difference.

---

### 5.2 Maintenance Gap

**Definition:**
Maintenance Gap = Years Since Last Maintenance

**Purpose:**

- Represents how long the pavement has been left without intervention

**Engineering Insight:**
Longer maintenance gaps typically lead to lower PCI due to accumulated damage.

---

### 5.3 Optional Feature Transformations (Advanced Insight)

Depending on the dataset, students may optionally:

- Apply log transformation to highly skewed variables (e.g., AADT)
- Normalize variables if distributions are extreme
- Combine environmental variables into a climate severity index

These are optional and should only be applied if justified by EDA.

---

### 5.4 Feature Selection Preparation

After feature engineering:

- Identify all usable input features
- Exclude non-informative columns such as IDs
- Separate features (X) and target (y)

---

### 5.5 Engineering Validation

Students must verify:

- Do new features make engineering sense?
- Do they improve correlation with PCI?
- Are there redundant or highly correlated variables?

---

### 5.6 Output of This Step

At the end of this step:

- The dataset includes both original and engineered features
- Features are ready for model training
- A new dataset is saved for reproducibility

---

### Summary

Feature engineering bridges the gap between raw data and real-world engineering behavior. This step ensures that the machine learning model learns from **meaningful, physically relevant inputs**, not just raw numbers.
