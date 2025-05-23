"""
Simple Restock Calculator
Generates basic restock recommendations based on recent sales data
"""
import pandas as pd
import os

# Create reports directory if it doesn't exist
os.makedirs('reports', exist_ok=True)

print("Generating restock recommendations...")

# Load data
df = pd.read_csv('synthetic_pharma_sales.csv')

# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Get the latest date in the data
latest_date = df['Date'].max()
print(f"Latest data date: {latest_date}")

# Group by drug and health center
latest_stock = df[df['Date'] == latest_date].groupby(['Drug_ID', 'Health_Center']).agg({
    'available_stock': 'first',
    'units_sold': 'first'
}).reset_index()

# Calculate 30-day average sales (simplistic approach)
all_sales = df.groupby(['Drug_ID', 'Health_Center']).agg({
    'units_sold': 'mean'
}).reset_index()
all_sales.rename(columns={'units_sold': 'avg_daily_sales'}, inplace=True)

# Merge the data
restock_data = pd.merge(latest_stock, all_sales, on=['Drug_ID', 'Health_Center'])

# Calculate days of supply left
restock_data['days_supply'] = restock_data['available_stock'] / restock_data['avg_daily_sales']
restock_data['days_supply'] = restock_data['days_supply'].round(1)

# Calculate 30-day forecast
restock_data['forecast_30day'] = restock_data['avg_daily_sales'] * 30
restock_data['forecast_30day'] = restock_data['forecast_30day'].round(0)

# Calculate restock amount (30-day supply minus current stock)
restock_data['restock_amount'] = restock_data['forecast_30day'] - restock_data['available_stock']
restock_data['restock_amount'] = restock_data['restock_amount'].apply(lambda x: max(0, round(x)))

# Add priority
def get_priority(days):
    if days < 7:
        return "URGENT"
    elif days < 14:
        return "HIGH"
    elif days < 30:
        return "MEDIUM"
    else:
        return "LOW"

restock_data['priority'] = restock_data['days_supply'].apply(get_priority)

# Sort by priority
priority_map = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
restock_data['priority_value'] = restock_data['priority'].map(priority_map)
restock_data = restock_data.sort_values('priority_value')

# Save the recommendations
restock_data.to_csv('reports/restock_recommendations.csv', index=False)
print(f"Saved restock recommendations to reports/restock_recommendations.csv")

# Report summary statistics
urgent_count = len(restock_data[restock_data['priority'] == 'URGENT'])
high_count = len(restock_data[restock_data['priority'] == 'HIGH'])
total_restock = restock_data['restock_amount'].sum()

print(f"\nSUMMARY:")
print(f"Total items requiring restock: {len(restock_data)}")
print(f"Urgent items: {urgent_count}")
print(f"High priority items: {high_count}")
print(f"Total units to restock: {total_restock:,.0f}")

# Display top urgent items
if urgent_count > 0:
    print("\nTop urgent items:")
    urgent = restock_data[restock_data['priority'] == 'URGENT']
    print(urgent[['Drug_ID', 'Health_Center', 'available_stock', 'days_supply', 'restock_amount']].head(5))

print("\nRestock analysis complete!")
