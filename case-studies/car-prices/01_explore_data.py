"""
01_explore_data.py — Load and explore the used car dataset.

This is the first step: understand what the data looks like before
doing anything with it. We use Pandas to load the CSV file and
inspect its shape, columns, and sample rows.

Run: python 01_explore_data.py
"""

import pandas as pd

# ─── Load the CSV file into a Pandas DataFrame ─────────────────────────────────
# A DataFrame is like a spreadsheet: rows are individual cars, columns are
# properties (brand, model_year, milage, price, etc.)
df = pd.read_csv("./data/used_cars.csv")

# ─── Basic info ────────────────────────────────────────────────────────────────
# .shape returns (number_of_rows, number_of_columns)
print(f"Dataset shape: {df.shape}")          # → (4009, 12)
print(f"Number of cars: {len(df)}")          # → 4009
print(f"Columns: {list(df.columns)}")        # → list of 12 column names

# ─── Show the first 5 rows ────────────────────────────────────────────────────
# .head() is the quickest way to see what the data looks like
print("\nFirst 5 rows:")
print(df.head())

# ─── Quick statistics on model_year ────────────────────────────────────────────
print(f"\nModel year — mean: {df['model_year'].mean():.1f}")
print(f"Model year — min:  {df['model_year'].min()}")
print(f"Model year — max:  {df['model_year'].max()}")
