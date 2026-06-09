"""
01_explore_data.py — Load and explore the stroke risk dataset.

This is your first look at the data. We load it, inspect the columns,
check the class balance (how many at-risk vs not-at-risk patients),
and understand what each feature means.

Dataset: Stroke Risk Prediction Dataset v2
Source: https://www.kaggle.com/datasets/mahatiratakandh/stroke-risk-prediction-dataset
License: CC BY 4.0

Run: python 01_explore_data.py
"""

import pandas as pd

# ─── Load the CSV ──────────────────────────────────────────────────────────────
df = pd.read_csv("./data/stroke_risk_dataset_v2.csv")

# ─── Basic info ────────────────────────────────────────────────────────────────
print(f"Dataset shape: {df.shape}")           # → (35000, 18)
print(f"Number of patients: {len(df)}")       # → 35000
print(f"Columns: {list(df.columns)}")

# ─── First 5 rows ─────────────────────────────────────────────────────────────
print("\nFirst 5 rows:")
print(df.head())

# ─── Column types ─────────────────────────────────────────────────────────────
# age: continuous integer (patient age)
# gender: categorical (Male/Female) — needs encoding
# 14 symptom columns: binary (0 = absent, 1 = present)
# stroke_risk_percentage: continuous float 0–100 (regression target)
# at_risk: binary 0/1 (classification target)
print("\nData types:")
print(df.dtypes)

# ─── Class balance ─────────────────────────────────────────────────────────────
# Important: if classes are very imbalanced, accuracy can be misleading.
at_risk_counts = df["at_risk"].value_counts()
print(f"\nClass distribution:")
print(f"  Not at risk (0): {at_risk_counts[0]} ({at_risk_counts[0]/len(df)*100:.1f}%)")
print(f"  At risk     (1): {at_risk_counts[1]} ({at_risk_counts[1]/len(df)*100:.1f}%)")

# ─── Age statistics ────────────────────────────────────────────────────────────
print(f"\nAge — min: {df['age'].min()}, max: {df['age'].max()}, mean: {df['age'].mean():.1f}")

# ─── Risk percentage statistics ────────────────────────────────────────────────
print(f"Risk % — min: {df['stroke_risk_percentage'].min()}, max: {df['stroke_risk_percentage'].max()}, mean: {df['stroke_risk_percentage'].mean():.1f}")

# ─── Gender distribution ──────────────────────────────────────────────────────
print(f"\nGender distribution:")
print(df["gender"].value_counts())
