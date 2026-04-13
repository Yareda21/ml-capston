# Predicting Pavement Condition Index (PCI) Using Machine Learning

## Project Scenario
A municipal road authority is struggling to efficiently allocate maintenance budgets due to limited inspection resources. Manual PCI assessment is time-consuming and expensive.

Your task is to develop a machine learning model that predicts **Pavement Condition Index (PCI)** based on historical road data such as traffic, structural properties, environmental factors, and maintenance history.

### Objectives
The final system should help engineers:
- Estimate pavement condition without full inspection
- Identify high-risk road segments
- Support maintenance prioritization decisions

## Dataset Structure
*Keep it strictly tabular (important for time constraints).*

### Input Features (X)
- **Pavement Age**: Age of the pavement in years.
- **Traffic Volume (AADT)**: Average Annual Daily Traffic.
- **Heavy Vehicle Percentage**: Percentage of heavy vehicles (%).
- **Pavement Thickness**: Thickness of the pavement layers (mm).
- **Number of Lanes**: Number of traffic lanes.
- **Subgrade Strength**: Soil strength (CBR or encoded Soil Type).
- **Climate Factors**: Rainfall and Temperature patterns.
- **Maintenance Gap**: Years since last maintenance.

### Target Variable (y)
- **Pavement Condition Index (PCI)**: A score ranging from 0–100.

---

## End-to-End Tasks

### 1. Data Loading & Inspection
- Load CSV using `pandas`.
- Understand structure and datatypes.
- Detect missing values.

### 2. Data Cleaning
- Handle missing values.
- Remove or treat outliers using the **IQR method**.
- Fix incorrect data types.

### 3. Exploratory Data Analysis (EDA)
- Scatter plots (e.g., Age vs PCI).
- Correlation heatmap.
- Identify strongest predictors.
- **Engineering interpretation**: Explaining the physical meaning of the data (critical for grading).

### 4. Feature Engineering
- Create a **Traffic Load Index** (AADT × heavy %).
- Calculate **Age since last maintenance**.
- Optional encoding for soil type.

### 5. Data Preparation
- Train-test split.
- Feature scaling using `StandardScaler`.

### 6. Model Building
*Strictly using models from the course:*
- **Linear Regression**
- **Ridge or Lasso Regression**
- **Random Forest Regression** (or Decision Tree if needed)

### 7. Model Evaluation
Compare models using:
- **R² Score**
- **Mean Squared Error (MSE)**

| Model | R² | MSE | Comment |
| :--- | :--- | :--- | :--- |
| Linear Regression | | | |
| Ridge/Lasso | | | |
| Random Forest | | | |

### 8. Model Selection
Choose the best model and justify your choice based on:
- Performance
- Stability
- Engineering interpretability

### 9. Model Saving
- Save the final model using `joblib` or `pickle`.
- Demonstrate reuse with a simple prediction script.

### 10. Technical Conclusion (CRITICAL)
Students must answer:
- What factors most affect PCI?
- How reliable is the model?
- Can this replace field inspection? (Critical thinking)
- Practical recommendations for engineers.

---

## Deliverables
1. **Jupyter Notebook / Python Script**: Containing the full pipeline.
2. **Cleaned Dataset**: The version used for modeling.
3. **Model File**: The exported `.pkl` file.
4. **Short Technical Report (2–3 pages)**:
   - Problem Statement
   - Methodology
   - Results
   - Engineering Interpretation

