"""
Seasonal Analysis Script
Focuses on identifying seasonal patterns in medication sales
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

print("=" * 80)
print("SEASONAL MEDICATION DEMAND ANALYSIS")
print("=" * 80)

# Load the dataset
df = pd.read_csv('synthetic_pharma_sales.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Clean data
df = df.dropna(subset=['units_sold', 'Season'])
print(f"After cleaning: {df.shape[0]} rows")

# Analyze seasonal patterns
print("\nANALYZING SEASONAL PATTERNS")
seasonal_sales = df.groupby('Season')['units_sold'].agg(['mean', 'count']).reset_index()
seasonal_sales = seasonal_sales.sort_values('mean', ascending=False)
print("\nAverage sales by season:")
print(seasonal_sales)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(x='Season', y='mean', data=seasonal_sales)
plt.title('Average Medication Sales by Season')
plt.ylabel('Average Units Sold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/seasonal_sales.png')
print("✅ Saved seasonal analysis to figures/seasonal_sales.png")

# Analyze top drugs in each season
print("\nTop performing drugs by season:")
season_drug = df.groupby(['Season', 'Drug_ID'])['units_sold'].mean().reset_index()

for season in df['Season'].unique():
    top_drugs = season_drug[season_drug['Season'] == season].sort_values('units_sold', ascending=False).head(3)
    print(f"\n{season}:")
    print(top_drugs)

# Create a simple seasonal heatmap
pivot_data = season_drug.pivot(index='Drug_ID', columns='Season', values='units_sold')
top_drugs_overall = df.groupby('Drug_ID')['units_sold'].sum().nlargest(10).index
pivot_subset = pivot_data.loc[pivot_data.index.isin(top_drugs_overall)]

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_subset, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Seasonal Sales Patterns for Top 10 Medications')
plt.tight_layout()
plt.savefig('figures/seasonal_heatmap.png')
print("✅ Saved seasonal heatmap to figures/seasonal_heatmap.png")

print("\n" + "=" * 80)
print("SEASONAL ANALYSIS COMPLETED")
print("=" * 80)
