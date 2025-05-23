"""
Restock Recommendation Generator
Analyzes current inventory levels and sales patterns to recommend restock quantities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Ensure output directories exist
os.makedirs('reports', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("=" * 80)
print("MEDICATION RESTOCK RECOMMENDATION SYSTEM")
print("=" * 80)

# Load the dataset
df = pd.read_csv('synthetic_pharma_sales.csv')
print(f"Dataset loaded with {df.shape[0]} rows")

# Convert date columns to datetime
df['Date'] = pd.to_datetime(df['Date'])
df['expiration_date'] = pd.to_datetime(df['expiration_date'])

# Basic data cleaning
df.dropna(subset=['units_sold', 'available_stock', 'Drug_ID', 'Health_Center'], inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

# Get the latest date in the dataset
latest_date = df['Date'].max()
print(f"Latest data date: {latest_date.strftime('%Y-%m-%d')}")

# Calculate average daily sales (last 30 days if available)
thirty_days_ago = latest_date - timedelta(days=30)
recent_data = df[df['Date'] >= thirty_days_ago]

# Group by drug, health center, and province
print("\nCalculating restock requirements...")
stock_analysis = recent_data.groupby(['Drug_ID', 'Health_Center', 'Province']).agg(
    avg_daily_sales=('units_sold', 'mean'),
    max_daily_sales=('units_sold', 'max'),
    total_sales=('units_sold', 'sum'),
    current_stock=('available_stock', 'last'),
    avg_price=('Price_Per_Unit', 'mean')
).reset_index()

# Calculate metrics for restock decisions
stock_analysis['days_until_stockout'] = stock_analysis['current_stock'] / stock_analysis['avg_daily_sales']
stock_analysis['days_until_stockout'] = stock_analysis['days_until_stockout'].fillna(100)  # Handle divide by zero
stock_analysis['days_until_stockout'] = stock_analysis['days_until_stockout'].round(1)

# Predicted demand for next 30 days
stock_analysis['demand_30days'] = stock_analysis['avg_daily_sales'] * 30
stock_analysis['demand_with_buffer'] = stock_analysis['demand_30days'] * 1.2  # 20% safety buffer

# Calculate restock quantity
stock_analysis['restock_quantity'] = stock_analysis['demand_with_buffer'] - stock_analysis['current_stock']
stock_analysis['restock_quantity'] = stock_analysis['restock_quantity'].apply(lambda x: max(0, round(x)))

# Assign priority based on days of inventory left
def get_priority(days):
    if days < 7:
        return "URGENT"
    elif days < 14:
        return "HIGH"
    elif days < 30:
        return "MEDIUM"
    else:
        return "LOW"

stock_analysis['restock_priority'] = stock_analysis['days_until_stockout'].apply(get_priority)

# Create a priority value for sorting
priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
stock_analysis['priority_value'] = stock_analysis['restock_priority'].map(priority_order)

# Sort by priority and restock quantity
restock_recommendations = stock_analysis.sort_values(['priority_value', 'restock_quantity'], ascending=[True, False])

# Estimated restock cost
restock_recommendations['estimated_cost'] = restock_recommendations['restock_quantity'] * restock_recommendations['avg_price']

# Save to CSV for operational use
restock_recommendations.to_csv('reports/restock_recommendations.csv', index=False)
print(f"✅ Saved detailed restock recommendations to reports/restock_recommendations.csv")

# Also create a simplified version for quick reference
simple_recommendations = restock_recommendations[['Drug_ID', 'Health_Center', 
                                               'Province', 'current_stock', 
                                               'days_until_stockout', 'restock_quantity',
                                               'restock_priority', 'estimated_cost']]
simple_recommendations.to_csv('reports/simple_restock_plan.csv', index=False)
print(f"✅ Saved simplified restock plan to reports/simple_restock_plan.csv")

# Generate summary statistics
total_drugs = restock_recommendations['Drug_ID'].nunique()
total_centers = restock_recommendations['Health_Center'].nunique()
total_provinces = restock_recommendations['Province'].nunique()
total_restock_needed = restock_recommendations['restock_quantity'].sum()
total_cost = restock_recommendations['estimated_cost'].sum()

print("\nRESTOCK SUMMARY STATISTICS")
print(f"Total unique drugs: {total_drugs}")
print(f"Total health centers: {total_centers}")
print(f"Total provinces: {total_provinces}")
print(f"Total units to restock: {total_restock_needed:.0f}")
print(f"Estimated restock cost: {total_cost:.2f}")

# Count items by priority
priority_counts = restock_recommendations['restock_priority'].value_counts().reset_index()
priority_counts.columns = ['Priority', 'Count']
print("\nItems by priority level:")
for _, row in priority_counts.sort_values('Priority', key=lambda x: x.map(priority_order)).iterrows():
    print(f"{row['Priority']}: {row['Count']} items")

# Display the most urgent items
print("\nMost urgent restock items:")
urgent_items = restock_recommendations[restock_recommendations['restock_priority'] == 'URGENT']
if len(urgent_items) > 0:
    print(urgent_items[['Drug_ID', 'Health_Center', 'current_stock', 
                      'days_until_stockout', 'restock_quantity']].head(10))
else:
    print("No urgent items found!")

# Generate visualization of restock quantities by priority
plt.figure(figsize=(10, 6))
priority_totals = restock_recommendations.groupby('restock_priority')['restock_quantity'].sum().reset_index()
# Sort by priority
priority_totals['sort_order'] = priority_totals['restock_priority'].map(priority_order)
priority_totals = priority_totals.sort_values('sort_order')

# Plot
sns.barplot(x='restock_priority', y='restock_quantity', data=priority_totals)
plt.title('Total Restock Quantity by Priority Level')
plt.ylabel('Total Units to Restock')
plt.xlabel('Priority Level')
plt.tight_layout()
plt.savefig('figures/restock_by_priority.png')
print("✅ Saved restock priority visualization to figures/restock_by_priority.png")

# Generate visualization of inventory status by province
plt.figure(figsize=(12, 8))
province_status = restock_recommendations.groupby(['Province', 'restock_priority']).size().unstack().fillna(0)
province_status.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Restock Needs by Province and Priority')
plt.ylabel('Number of Items')
plt.xlabel('Province')
plt.legend(title='Priority')
plt.tight_layout()
plt.savefig('figures/restock_by_province.png')
print("✅ Saved provincial restock visualization to figures/restock_by_province.png")

print("\n" + "=" * 80)
print("RESTOCK RECOMMENDATION SYSTEM COMPLETED")
print("=" * 80)
