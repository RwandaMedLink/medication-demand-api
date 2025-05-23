"""
Quick Medication Demand Analysis Script
Focuses on identifying key sales drivers and generating visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression

# Ensure output directories exist
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("=" * 80)
print("RWANDA MEDICATION DEMAND ANALYSIS")
print("=" * 80)

# Load the dataset
df = pd.read_csv('synthetic_pharma_sales.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Convert date columns
df['Date'] = pd.to_datetime(df['Date'])
df['expiration_date'] = pd.to_datetime(df['expiration_date'])
df['stock_entry_timestamp'] = pd.to_datetime(df['stock_entry_timestamp'])

# Data cleaning
df.dropna(inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

# 1. ANALYZE SEASONAL PATTERNS
print("\n1. SEASONAL SALES PATTERNS")
if 'Season' in df.columns:
    seasonal_sales = df.groupby('Season')['units_sold'].agg(['mean', 'median', 'count']).reset_index()
    seasonal_sales = seasonal_sales.sort_values('mean', ascending=False)
    print(seasonal_sales)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Season', y='mean', data=seasonal_sales)
    plt.title('Average Medication Sales by Season')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/seasonal_sales.png')
    print("✅ Saved seasonal analysis visualization")

# 2. PROMOTION EFFECTIVENESS
print("\n2. PROMOTION EFFECTIVENESS")
if 'Promotion' in df.columns:
    promotion_effect = df.groupby(['Drug_ID', 'Promotion'])['units_sold'].mean().reset_index()
    promotion_pivot = promotion_effect.pivot(index='Drug_ID', columns='Promotion', values='units_sold')
    
    if promotion_pivot.shape[1] >= 2:  # If we have both 0 and 1 for Promotion
        promotion_pivot.columns = ['No_Promo', 'With_Promo']
        promotion_pivot['Lift_Percent'] = (promotion_pivot['With_Promo'] / promotion_pivot['No_Promo'] - 1) * 100
        top_responsive = promotion_pivot.sort_values('Lift_Percent', ascending=False).head(10)
        
        print("Top 10 drugs with highest promotion response:")
        print(top_responsive[['Lift_Percent']].round(1))
        
        # Visualize top 5
        plt.figure(figsize=(12, 6))
        top5_drugs = top_responsive.head(5).index
        plot_data = df[df['Drug_ID'].isin(top5_drugs)]
        
        sns.barplot(x='Drug_ID', y='units_sold', hue='Promotion', data=plot_data)
        plt.title('Impact of Promotions on Top 5 Responsive Drugs')
        plt.ylabel('Average Units Sold')
        plt.tight_layout()
        plt.savefig('figures/promotion_impact.png')
        print("✅ Saved promotion analysis visualization")

# 3. RESTOCK RECOMMENDATIONS
print("\n3. GENERATING RESTOCK RECOMMENDATIONS")

# Get latest data for each drug/location
latest_date = df['Date'].max()
latest_data = df[df['Date'] == latest_date]

# Calculate average daily sales (from last 30 days if available)
thirty_days_ago = latest_date - pd.Timedelta(days=30)
recent_data = df[df['Date'] >= thirty_days_ago]

# Group by drug and health center
sales_by_location = recent_data.groupby(['Drug_ID', 'Health_Center']).agg(
    avg_daily_sales=('units_sold', 'mean'),
    total_sales=('units_sold', 'sum'),
    current_stock=('available_stock', 'last')
).reset_index()

# Calculate days of supply left and restock needs
sales_by_location['days_supply_remaining'] = sales_by_location['current_stock'] / sales_by_location['avg_daily_sales']
sales_by_location['30_day_forecast'] = sales_by_location['avg_daily_sales'] * 30
sales_by_location['restock_needed'] = sales_by_location['30_day_forecast'] - sales_by_location['current_stock']
sales_by_location['restock_needed'] = sales_by_location['restock_needed'].apply(lambda x: max(0, round(x)))

# Add priority levels
def get_priority(days):
    if days <= 7:
        return "URGENT"
    elif days <= 14:
        return "HIGH"
    elif days <= 30:
        return "MEDIUM"
    else:
        return "LOW"

sales_by_location['priority'] = sales_by_location['days_supply_remaining'].apply(get_priority)

# Sort by priority and save
priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
sales_by_location['priority_value'] = sales_by_location['priority'].map(priority_order)
restock_recommendations = sales_by_location.sort_values('priority_value')

# Save to CSV
restock_recommendations.to_csv('reports/restock_recommendations.csv', index=False)

# Display urgent items
urgent_items = restock_recommendations[restock_recommendations['priority'] == 'URGENT']
print(f"\nUrgent items requiring immediate restock: {len(urgent_items)}")
if not urgent_items.empty:
    print(urgent_items[['Drug_ID', 'Health_Center', 'current_stock', 'days_supply_remaining', 'restock_needed']].head())

# 4. KEY DRIVERS CORRELATION ANALYSIS
print("\n4. KEY SALES DRIVERS CORRELATION ANALYSIS")

# Select numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [col for col in numerical_cols if col != 'units_sold']

# Calculate correlations
correlations = []
for col in numerical_cols:
    corr = df[['units_sold', col]].corr().iloc[0, 1]
    correlations.append({'Feature': col, 'Correlation': corr})

corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)

# Display top correlations
print("\nTop factors correlated with medication sales:")
print(corr_df.head(10))

# Visualize
plt.figure(figsize=(12, 8))
top_features = corr_df.head(10)
sns.barplot(x='Correlation', y='Feature', data=top_features)
plt.title('Top 10 Features Correlated with Medication Sales')
plt.tight_layout()
plt.savefig('figures/correlation_analysis.png')
print("✅ Saved correlation analysis visualization")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print(f"- {len(restock_recommendations)} drug-location combinations analyzed")
print(f"- {len(urgent_items)} items identified for urgent restock")
print("Generated files:")
print("- figures/seasonal_sales.png")
print("- figures/promotion_impact.png") 
print("- figures/correlation_analysis.png")
print("- reports/restock_recommendations.csv")
print("=" * 80)
