## Step 0 — Problem Framing

### Project title:

Predicting Pavement Condition Index (PCI) Using Machine Learning

### Problem statement:

Road agencies need a practical way to estimate pavement condition so they can prioritize maintenance before roads deteriorate too far. Manual PCI assessment is expensive, slow, and not always available for every road segment. In this project, students will build a regression model that predicts PCI from pavement, traffic, structural, environmental, and maintenance-related variables.

This fits your module exactly because the capstone is supposed to be an end-to-end machine learning project with data loading, cleaning, feature engineering, model training, evaluation, comparison, saving the model, and technical conclusions.

### Target variable:

PCI score, usually on a 0–100 scale.

### Practical meaning of the output:

The model should help answer:
“Given the current road conditions and history, what PCI is likely for this pavement section?”

### Engineering decision framing:

To make the project more meaningful, students should interpret PCI as:

PCI < 40 → poor condition, urgent intervention
PCI 40–70 → moderate condition, maintenance needed
PCI > 70 → good condition, low immediate risk

That turns the project from a pure prediction task into a maintenance support tool.

## Step 1 — Dataset Understanding and Hypothesis Formation

Before any coding, students should first understand what the dataset represents and what engineering behavior they expect to see.

### What students must identify

They should answer:

What does each row represent?
For example, one road section, one pavement sample, or one inspection record.
What does each feature mean physically?
Which variable is the target?
Which variables are likely to influence PCI most?
Example feature groups

### A good PCI dataset may include:

Pavement age
Traffic volume
Heavy vehicle percentage
Pavement thickness
Number of lanes
Subgrade strength / CBR
Rainfall or temperature
Years since last maintenance
Required hypotheses

Students should write 3 to 5 simple but meaningful hypotheses before looking too deeply at the data.

Example hypotheses:

#### Older pavements will have lower PCI.

#### Road sections with higher heavy-vehicle traffic will have lower PCI.

#### Thicker pavement layers will be associated with higher PCI.

#### Roads with recent maintenance will show higher PCI.

#### Poor subgrade strength will reduce PCI over time.

Why this step matters

This step forces students to think like engineers, not just coders. They are not merely fitting a model; they are testing whether the data behaves in a way that matches real pavement deterioration logic.

A short section like this:

The objective of this project is to predict Pavement Condition Index (PCI) using regression-based machine learning models. We assume that pavement age, traffic loading, structural properties, environmental exposure, and maintenance history influence PCI. The following hypotheses will be tested through exploratory data analysis and model building.
