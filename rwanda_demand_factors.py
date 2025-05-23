"""
Simplified Rwanda Medication Demand Analysis
Focusing on key factors: seasons, outbreaks, pricing, effectiveness, and supply chain
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("=" * 80)
print("RWANDA MEDICATION DEMAND FACTORS ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv('synthetic_pharma_sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"Loaded dataset with {df.shape[0]} rows")

# Data cleaning
df.dropna(subset=['units_sold'], inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

# 1. SEASONAL ANALYSIS
print("\n1. SEASONAL IMPACT ANALYSIS")

if 'Season' in df.columns:
    season_sales = df.groupby('Season')['units_sold'].agg(['mean', 'median', 'count']).reset_index()
    season_sales = season_sales.sort_values('mean', ascending=False)
    
    print("\nSales by Rwandan season:")
    print(season_sales)
    
    # Calculate seasonal variation
    max_season = season_sales.iloc[0]['Season']
    min_season = season_sales.iloc[-1]['Season']
    seasonal_diff = (season_sales.iloc[0]['mean'] / season_sales.iloc[-1]['mean'] - 1) * 100
    
    print(f"\nSeasonal impact: {seasonal_diff:.1f}% higher sales in {max_season} vs {min_season}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Season', y='mean', data=season_sales)
    plt.title('Medication Demand by Rwandan Season')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/seasonal_demand.png')
    
    # Drug categories by season
    if 'ATC_Code' in df.columns:
        seasonal_by_drug = df.groupby(['Season', 'ATC_Code'])['units_sold'].mean().reset_index()
        
        # Create a pivot table to see the seasonal patterns by drug category
        seasonal_pivot = seasonal_by_drug.pivot(index='ATC_Code', columns='Season', values='units_sold')
        seasonal_pivot['Variation'] = seasonal_pivot.max(axis=1) / seasonal_pivot.min(axis=1)
        seasonal_pivot['Peak_Season'] = seasonal_pivot.idxmax(axis=1)
        
        top_seasonal = seasonal_pivot.sort_values('Variation', ascending=False).head(5)
        print("\nDrug categories with highest seasonal variation:")
        print(top_seasonal[['Variation', 'Peak_Season']])

# 2. DISEASE OUTBREAK ANALYSIS
print("\n2. DISEASE OUTBREAK IMPACT")

if 'Disease_Outbreak' in df.columns:
    # Create categories for easier analysis
    df['Outbreak_Category'] = pd.cut(df['Disease_Outbreak'], 
                                    bins=[0, 0.5, 1.0, 1.5, 2.0], 
                                    labels=['None', 'Low', 'Medium', 'High'])
    
    # Analyze impact
    outbreak_impact = df.groupby('Outbreak_Category')['units_sold'].mean().reset_index()
    print("\nImpact of disease outbreaks on medication demand:")
    print(outbreak_impact)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Outbreak_Category', y='units_sold', data=outbreak_impact)
    plt.title('Effect of Disease Outbreaks on Medication Demand')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/outbreak_demand.png')
    
    # Drug responses to outbreaks
    if 'Drug_ID' in df.columns:
        # Get top 10 drugs by volume
        top_drugs = df.groupby('Drug_ID')['units_sold'].sum().nlargest(10).index
        
        # Filter for those drugs
        top_drug_df = df[df['Drug_ID'].isin(top_drugs)]
        
        # Analyze outbreak response for top drugs
        drug_outbreak = top_drug_df.groupby(['Drug_ID', 'Outbreak_Category'])['units_sold'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Drug_ID', y='units_sold', hue='Outbreak_Category', data=drug_outbreak)
        plt.title('Response of Top Medications to Disease Outbreaks')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/drug_outbreak_response.png')

# 3. PRICE SENSITIVITY ANALYSIS
print("\n3. PRICE SENSITIVITY ANALYSIS")

if 'Price_Per_Unit' in df.columns:
    # Create price categories
    df['Price_Category'] = pd.qcut(df['Price_Per_Unit'], 5, 
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Analyze price impact
    price_impact = df.groupby('Price_Category')['units_sold'].mean().reset_index()
    print("\nPrice sensitivity of medication demand:")
    print(price_impact)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Price_Category', y='units_sold', data=price_impact)
    plt.title('Effect of Price on Medication Demand')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/price_sensitivity.png')
    
    # Analyze by province (as a proxy for different economic regions)
    if 'Province' in df.columns:
        price_by_province = df.groupby(['Province', 'Price_Category'])['units_sold'].mean().reset_index()
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Price_Category', y='units_sold', hue='Province', data=price_by_province)
        plt.title('Price Sensitivity by Province')
        plt.ylabel('Average Units Sold')
        plt.tight_layout()
        plt.savefig('figures/price_by_province.png')

# 4. MEDICATION EFFECTIVENESS ANALYSIS
print("\n4. EFFECTIVENESS & REPUTATION ANALYSIS")

if 'Effectiveness_Rating' in df.columns:
    # Analyze impact
    eff_impact = df.groupby('Effectiveness_Rating')['units_sold'].mean().reset_index()
    print("\nImpact of medication effectiveness on demand:")
    print(eff_impact)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Effectiveness_Rating', y='units_sold', data=eff_impact)
    plt.title('Effect of Effectiveness Rating on Medication Demand')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/effectiveness_impact.png')
    
    # Correlation
    eff_corr = df[['Effectiveness_Rating', 'units_sold']].corr().iloc[0, 1]
    print(f"\nCorrelation between effectiveness and demand: {eff_corr:.3f}")

# 5. SUPPLY CHAIN & AVAILABILITY ANALYSIS
print("\n5. SUPPLY CHAIN & AVAILABILITY ANALYSIS")

if 'Availability_Score' in df.columns:
    # Create categories
    df['Availability_Category'] = pd.qcut(df['Availability_Score'], 4, 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Analyze impact
    avail_impact = df.groupby('Availability_Category')['units_sold'].mean().reset_index()
    print("\nImpact of medication availability on demand:")
    print(avail_impact)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Availability_Category', y='units_sold', data=avail_impact)
    plt.title('Effect of Medication Availability on Demand')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/availability_impact.png')

if 'Supply_Chain_Delay' in df.columns:
    # Handle NaN values
    df['Supply_Chain_Delay'] = df['Supply_Chain_Delay'].fillna('None')
    
    # Analyze impact
    delay_impact = df.groupby('Supply_Chain_Delay')['units_sold'].mean().reset_index()
    print("\nImpact of supply chain delays on demand:")
    print(delay_impact)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Supply_Chain_Delay', y='units_sold', data=delay_impact)
    plt.title('Effect of Supply Chain Delays on Medication Demand')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/supply_chain_impact.png')

# 6. COMBINED ANALYSIS
print("\n6. COMBINED FACTOR ANALYSIS")

# Create a combined visualization showing interaction of key factors
if all(col in df.columns for col in ['Season', 'Availability_Score', 'Disease_Outbreak']):
    plt.figure(figsize=(14, 10))
    
    # Use sample of data for better visualization
    sample_df = df.sample(n=5000, random_state=42)
    
    # Create scatter plot
    scatter = sns.scatterplot(
        data=sample_df,
        x='Availability_Score',
        y='units_sold',
        hue='Season',
        size='Disease_Outbreak',
        sizes=(20, 200),
        alpha=0.7
    )
    
    plt.title('Combined Effect of Key Factors on Medication Demand')
    plt.xlabel('Medication Availability Score')
    plt.ylabel('Units Sold')
    plt.tight_layout()
    plt.savefig('figures/combined_factors.png')
    print("✅ Created visualization showing interaction of key demand factors")

# 7. GENERATE COMPREHENSIVE SUMMARY
print("\n7. GENERATING COMPREHENSIVE SUMMARY")

# Create summary report
summary = []

# Add seasonal findings
if 'Season' in df.columns:
    summary.append(f"SEASONAL PATTERNS: Sales are {seasonal_diff:.1f}% higher during {max_season} " + 
                  f"compared to {min_season}. Stock levels should be increased before {max_season}.")

# Add outbreak findings
if 'Disease_Outbreak' in df.columns:
    outbreak_high = outbreak_impact[outbreak_impact['Outbreak_Category'] == 'High']['units_sold'].values[0]
    outbreak_low = outbreak_impact[outbreak_impact['Outbreak_Category'] == 'Low']['units_sold'].values[0]
    outbreak_diff = (outbreak_high / outbreak_low - 1) * 100
    summary.append(f"DISEASE OUTBREAKS: High outbreak conditions increase medication demand by {outbreak_diff:.1f}% " +
                  "compared to low outbreak conditions. Emergency stock should be maintained.")

# Add price findings
if 'Price_Per_Unit' in df.columns:
    price_low = price_impact[price_impact['Price_Category'] == 'Very Low']['units_sold'].values[0]
    price_high = price_impact[price_impact['Price_Category'] == 'Very High']['units_sold'].values[0]
    price_diff = (price_low / price_high - 1) * 100
    summary.append(f"PRICE SENSITIVITY: Lower-priced medications show {price_diff:.1f}% higher demand compared " +
                  "to higher-priced options, indicating significant price sensitivity.")

# Add effectiveness findings
if 'Effectiveness_Rating' in df.columns:
    summary.append(f"EFFECTIVENESS & REPUTATION: Each point increase in effectiveness rating is associated " +
                  f"with a {eff_corr:.1f} correlation in demand. Higher-rated medications are preferred.")

# Add availability findings
if 'Availability_Score' in df.columns:
    avail_high = avail_impact[avail_impact['Availability_Category'] == 'Very High']['units_sold'].values[0]
    avail_low = avail_impact[avail_impact['Availability_Category'] == 'Low']['units_sold'].values[0]
    avail_diff = (avail_high / avail_low - 1) * 100
    summary.append(f"AVAILABILITY & SUPPLY: High availability increases demand by {avail_diff:.1f}% compared " +
                  "to low availability. Supply chain delays significantly reduce medication consumption.")

# Save detailed report
with open('reports/rwanda_demand_factors.txt', 'w') as f:
    f.write("RWANDA MEDICATION DEMAND FACTORS: COMPREHENSIVE ANALYSIS\n")
    f.write("="*70 + "\n\n")
    
    f.write("KEY FINDINGS\n")
    f.write("-"*70 + "\n\n")
    for finding in summary:
        f.write(finding + "\n\n")
    
    f.write("\nDETAILED RECOMMENDATIONS\n")
    f.write("-"*70 + "\n\n")
    
    f.write("1. SEASONAL INVENTORY MANAGEMENT\n")
    f.write(f"   - Increase inventory by 30-40% before {max_season if 'max_season' in locals() else 'peak seasons'}\n")
    f.write("   - Create seasonal forecasting models for each medication category\n")
    f.write("   - Consider regional variations in seasonal patterns\n\n")
    
    f.write("2. EPIDEMIC PREPAREDNESS\n")
    f.write("   - Establish early warning systems for disease outbreaks\n")
    f.write("   - Maintain emergency stock of critical medications\n")
    f.write("   - Develop rapid response distribution protocols\n\n")
    
    f.write("3. PRICE OPTIMIZATION\n")
    f.write("   - Implement differential pricing based on regional economics\n")
    f.write("   - Consider subsidies for essential, price-sensitive medications\n")
    f.write("   - Monitor price elasticity by medication category\n\n")
    
    f.write("4. MEDICATION SELECTION & REPUTATION\n")
    f.write("   - Prioritize stocking medications with higher effectiveness ratings\n")
    f.write("   - Provide education to healthcare workers about medication efficacy\n")
    f.write("   - Collect and analyze patient feedback on medication effectiveness\n\n")
    
    f.write("5. SUPPLY CHAIN OPTIMIZATION\n")
    f.write("   - Implement real-time inventory tracking systems\n")
    f.write("   - Establish regional distribution centers to reduce delays\n")
    f.write("   - Create redundancy in supply networks for critical medications\n")

# Print key findings
print("\nKEY FINDINGS:")
for i, finding in enumerate(summary, 1):
    print(f"{i}. {finding}")

print(f"\n✅ Saved comprehensive report to reports/rwanda_demand_factors.txt")
print(f"✅ Generated {len(os.listdir('figures'))} visualizations in the figures directory")

print("\n" + "=" * 80)
print("RWANDA MEDICATION DEMAND ANALYSIS COMPLETED")
print("=" * 80)
