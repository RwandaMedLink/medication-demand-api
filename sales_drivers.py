"""
Sales Drivers Analysis
Identifies key factors that influence medication sales in Rwanda health centers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Ensure output directories exist
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("=" * 80)
print("MEDICATION SALES DRIVERS ANALYSIS")
print("=" * 80)

# Load the dataset
df = pd.read_csv('synthetic_pharma_sales.csv')
print(f"Dataset loaded with {df.shape[0]} rows, {df.shape[1]} columns")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Basic data cleaning
df.dropna(subset=['units_sold'], inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

# 1. CORRELATION ANALYSIS
print("\n1. IDENTIFYING NUMERICAL SALES DRIVERS")

# Select numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
numerical_cols = [col for col in numerical_cols if col != 'units_sold']

# Calculate correlations
correlations = []
for col in numerical_cols:
    corr = df[['units_sold', col]].corr().iloc[0, 1]
    correlations.append({'Feature': col, 'Correlation': corr})

# Create correlation dataframe
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Correlation', ascending=False)

# Display top correlations
print("\nTop factors correlated with medication sales:")
print(corr_df.head(10))

# Save correlation data
corr_df.to_csv('reports/sales_correlations.csv', index=False)

# Visualize
plt.figure(figsize=(12, 8))
sns.barplot(x='Correlation', y='Feature', data=corr_df.head(10))
plt.title('Top Factors Correlated with Medication Sales')
plt.axvline(x=0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig('figures/correlation_analysis.png')
print("✅ Saved correlation analysis to figures/correlation_analysis.png")

# 2. RWANDAN SEASONAL TRENDS ANALYSIS
print("\n2. ANALYZING RWANDAN SEASONAL TRENDS")

if 'Season' in df.columns:
    # Analyze seasonal patterns
    seasonal_sales = df.groupby('Season')['units_sold'].agg(['mean', 'median', 'count', 'std']).reset_index()
    seasonal_sales = seasonal_sales.sort_values('mean', ascending=False)
    
    print("\nSales by Rwandan season:")
    print(seasonal_sales)
    
    # Calculate seasonal impact percentage
    max_season = seasonal_sales.iloc[0]
    min_season = seasonal_sales.iloc[-1]
    impact_pct = (max_season['mean'] - min_season['mean']) / min_season['mean'] * 100
    
    print(f"\nSeasonal impact: {impact_pct:.1f}% higher sales in {max_season['Season']} vs {min_season['Season']}")
    
    # Analyze seasonal impact by drug category
    if 'ATC_Code' in df.columns:
        print("\nSeasonal impact by drug category:")
        seasonal_by_atc = df.groupby(['Season', 'ATC_Code'])['units_sold'].mean().reset_index()
        seasonal_by_atc_pivot = seasonal_by_atc.pivot(index='ATC_Code', columns='Season', values='units_sold')
        
        # Calculate seasonal variation for each drug category
        seasonal_by_atc_pivot['Variation'] = seasonal_by_atc_pivot.max(axis=1) / seasonal_by_atc_pivot.min(axis=1)
        seasonal_by_atc_pivot['Peak_Season'] = seasonal_by_atc_pivot.idxmax(axis=1)
        top_seasonal_categories = seasonal_by_atc_pivot.sort_values('Variation', ascending=False)
        
        print("Top drug categories by seasonal variation:")
        print(top_seasonal_categories[['Variation', 'Peak_Season']].head(5))
        
        # Visualization for seasonal trends
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=seasonal_by_atc, x='Season', y='units_sold', hue='ATC_Code', marker='o')
        plt.title('Medication Sales by Season and Drug Category')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/seasonal_trends.png')
        print("✅ Saved seasonal trends visualization")

# 3. EPIDEMIC/DISEASE OUTBREAK ANALYSIS
print("\n3. ANALYZING DISEASE OUTBREAK IMPACT")

if 'Disease_Outbreak' in df.columns:
    # Convert to categorical for better analysis
    df['Outbreak_Level'] = pd.cut(df['Disease_Outbreak'], 
                                  bins=[0, 0.5, 1.0, 1.5, 2.0], 
                                  labels=['None', 'Low', 'Medium', 'High'])
    
    # Analyze overall impact
    outbreak_impact = df.groupby('Outbreak_Level')['units_sold'].agg(['mean', 'count']).reset_index()
    print("\nImpact of disease outbreaks on medication sales:")
    print(outbreak_impact)
    
    # Calculate percentage increase during outbreaks
    if 'None' in outbreak_impact['Outbreak_Level'].values and 'High' in outbreak_impact['Outbreak_Level'].values:
        no_outbreak = outbreak_impact.loc[outbreak_impact['Outbreak_Level'] == 'None', 'mean'].values[0]
        high_outbreak = outbreak_impact.loc[outbreak_impact['Outbreak_Level'] == 'High', 'mean'].values[0]
        outbreak_lift = (high_outbreak - no_outbreak) / no_outbreak * 100
        print(f"High outbreak conditions increase sales by {outbreak_lift:.1f}% on average")
    
    # Analyze by drug category
    if 'ATC_Code' in df.columns:
        outbreak_by_atc = df.groupby(['Outbreak_Level', 'ATC_Code'])['units_sold'].mean().reset_index()
        
        # Visualize
        plt.figure(figsize=(14, 8))
        sns.barplot(x='ATC_Code', y='units_sold', hue='Outbreak_Level', data=outbreak_by_atc)
        plt.title('Impact of Disease Outbreaks by Drug Category')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/outbreak_impact.png')
        print("✅ Saved disease outbreak impact visualization")
        
        # Find drugs most responsive to outbreaks
        outbreak_pivot = outbreak_by_atc.pivot(index='ATC_Code', 
                                              columns='Outbreak_Level', 
                                              values='units_sold').reset_index()
        
        if 'None' in outbreak_pivot.columns and 'High' in outbreak_pivot.columns:
            outbreak_pivot['Outbreak_Response'] = outbreak_pivot['High'] / outbreak_pivot['None']
            top_responsive = outbreak_pivot.sort_values('Outbreak_Response', ascending=False)
            
            print("\nDrug categories most responsive to disease outbreaks:")
            print(top_responsive[['ATC_Code', 'Outbreak_Response']].head(5))
            
            # Save to reports
            top_responsive.to_csv('reports/outbreak_response.csv', index=False)

# 4. PRICE SENSITIVITY ANALYSIS
print("\n4. ANALYZING PRICE SENSITIVITY")

if 'Price_Per_Unit' in df.columns:
    # Create price bins for analysis
    df['Price_Category'] = pd.qcut(df['Price_Per_Unit'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Overall price impact
    price_impact = df.groupby('Price_Category')['units_sold'].agg(['mean', 'count']).reset_index()
    print("\nImpact of pricing on medication sales:")
    print(price_impact)
    
    # Calculate price elasticity (simplified)
    lowest_price_sales = price_impact.loc[price_impact['Price_Category'] == 'Very Low', 'mean'].values[0]
    highest_price_sales = price_impact.loc[price_impact['Price_Category'] == 'Very High', 'mean'].values[0]
    price_sensitivity = (lowest_price_sales - highest_price_sales) / highest_price_sales * 100
    print(f"Price sensitivity indicator: {price_sensitivity:.1f}% higher sales at lowest vs highest price point")
    
    # Visualize price impact
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Price_Category', y='mean', data=price_impact)
    plt.title('Impact of Price on Medication Sales')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/price_impact.png')
    print("✅ Saved price sensitivity visualization")
    
    # Analysis by drug category
    if 'ATC_Code' in df.columns:
        # For each drug category, compute correlation between price and sales
        atc_price_sensitivity = []
        for atc in df['ATC_Code'].unique():
            atc_df = df[df['ATC_Code'] == atc]
            corr = atc_df[['Price_Per_Unit', 'units_sold']].corr().iloc[0, 1]
            atc_price_sensitivity.append({
                'ATC_Code': atc,
                'Price_Correlation': corr,
                'Avg_Price': atc_df['Price_Per_Unit'].mean(),
                'Avg_Sales': atc_df['units_sold'].mean()
            })
        
        atc_price_df = pd.DataFrame(atc_price_sensitivity)
        print("\nPrice sensitivity by drug category:")
        print(atc_price_df.sort_values('Price_Correlation').head())
        
        # Save to reports
        atc_price_df.to_csv('reports/price_sensitivity.csv', index=False)

# 5. MEDICATION EFFECTIVENESS ANALYSIS
print("\n5. ANALYZING IMPACT OF MEDICATION EFFECTIVENESS")

if 'Effectiveness_Rating' in df.columns:
    # Analyze impact of effectiveness rating
    effectiveness_impact = df.groupby('Effectiveness_Rating')['units_sold'].agg(['mean', 'count']).reset_index()
    print("\nImpact of medication effectiveness on sales:")
    print(effectiveness_impact)
    
    # Visualize effectiveness impact
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Effectiveness_Rating', y='mean', data=effectiveness_impact)
    plt.title('Impact of Effectiveness Rating on Medication Sales')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/effectiveness_impact.png')
    print("✅ Saved effectiveness impact visualization")
    
    # Correlation between effectiveness and sales
    eff_corr = df[['Effectiveness_Rating', 'units_sold']].corr().iloc[0, 1]
    print(f"Correlation between effectiveness rating and sales: {eff_corr:.3f}")

# 6. SUPPLY CHAIN AND AVAILABILITY ANALYSIS
print("\n6. ANALYZING SUPPLY CHAIN AND AVAILABILITY FACTORS")

# Analyze availability score impact
if 'Availability_Score' in df.columns:
    # Create availability categories
    df['Availability_Category'] = pd.qcut(df['Availability_Score'], 4, 
                                         labels=['Low', 'Medium', 'High', 'Very High'])
    
    availability_impact = df.groupby('Availability_Category')['units_sold'].agg(['mean', 'count']).reset_index()
    print("\nImpact of medication availability on sales:")
    print(availability_impact)
    
    # Calculate availability impact percentage
    low_avail = availability_impact.loc[availability_impact['Availability_Category'] == 'Low', 'mean'].values[0]
    high_avail = availability_impact.loc[availability_impact['Availability_Category'] == 'Very High', 'mean'].values[0]
    avail_impact = (high_avail - low_avail) / low_avail * 100
    print(f"High availability increases sales by {avail_impact:.1f}% compared to low availability")

# Analyze supply chain delay impact
if 'Supply_Chain_Delay' in df.columns:
    # Handle NaN values in Supply_Chain_Delay
    df['Supply_Chain_Delay'] = df['Supply_Chain_Delay'].fillna('None')
    
    delay_impact = df.groupby('Supply_Chain_Delay')['units_sold'].agg(['mean', 'count']).reset_index()
    print("\nImpact of supply chain delays on sales:")
    print(delay_impact)
    
    # Visualize supply chain impact
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Supply_Chain_Delay', y='mean', data=delay_impact)
    plt.title('Impact of Supply Chain Delays on Medication Sales')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/supply_chain_impact.png')
    print("✅ Saved supply chain impact visualization")
    
    # Analyze combined effect of availability and supply chain
    combined_df = df.groupby(['Supply_Chain_Delay', 'Availability_Category'])['units_sold'].mean().reset_index()
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Supply_Chain_Delay', y='units_sold', hue='Availability_Category', data=combined_df)
    plt.title('Combined Effect of Supply Chain and Availability on Sales')
    plt.ylabel('Average Units Sold')
    plt.tight_layout()
    plt.savefig('figures/supply_availability_combined.png')
    print("✅ Saved combined supply chain and availability visualization")

# 7. CATEGORICAL VARIABLES IMPACT
print("\n7. ANALYZING OTHER CATEGORICAL SALES DRIVERS")

# Identify categorical columns (excluding ones we've already analyzed)
already_analyzed = ['Season', 'Outbreak_Level', 'Price_Category', 'Effectiveness_Rating', 
                    'Availability_Category', 'Supply_Chain_Delay']
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
remaining_categorical = [col for col in categorical_cols if col not in already_analyzed]
print(f"Remaining categorical variables to analyze: {remaining_categorical}")

# Analyze impact of each categorical variable
for i, col in enumerate(remaining_categorical[:3]):  # Analyze top 3 remaining categoricals
    if df[col].nunique() > 30:  # Skip if too many unique values
        print(f"Skipping {col} - too many unique values ({df[col].nunique()})")
        continue
        
    print(f"\nAnalyzing impact of {col}:")
    
    # Calculate average sales by category
    cat_impact = df.groupby(col)['units_sold'].agg(['mean', 'count']).reset_index()
    cat_impact = cat_impact.sort_values('mean', ascending=False)
    
    # Show results
    print(cat_impact.head(5))
    
    # Check statistical significance with ANOVA
    categories = df[col].unique()
    if len(categories) > 1:  # Only if we have multiple categories
        samples = [df[df[col] == category]['units_sold'] for category in categories]
        try:
            f_stat, p_value = stats.f_oneway(*samples)
            print(f"ANOVA: F-statistic={f_stat:.2f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print(f"✓ {col} has statistically significant impact on sales (p<0.05)")
            else:
                print(f"✗ {col} does not have statistically significant impact on sales (p≥0.05)")
        except:
            print(f"Could not perform ANOVA on {col}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(x=col, y='units_sold', data=df, estimator=np.mean, ci=None)
    plt.title(f'Impact of {col} on Medication Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'figures/impact_of_{col}.png')
    print(f"✅ Saved {col} impact visualization")

# 8. PROMOTION EFFECTIVENESS
print("\n8. ANALYZING PROMOTION EFFECTIVENESS")

if 'Promotion' in df.columns:
    # Overall promotion effect
    promotion_overall = df.groupby('Promotion')['units_sold'].agg(['mean', 'count']).reset_index()
    print("\nOverall promotion effect:")
    print(promotion_overall)
    
    # Calculate lift
    if promotion_overall.shape[0] >= 2:
        no_promo = promotion_overall.loc[promotion_overall['Promotion'] == 0, 'mean'].values[0]
        with_promo = promotion_overall.loc[promotion_overall['Promotion'] == 1, 'mean'].values[0]
        lift = (with_promo - no_promo) / no_promo * 100
        print(f"Promotion lift: {lift:.2f}% increase in average sales")
    
    # Drug-specific promotion effect
    drug_promo = df.groupby(['Drug_ID', 'Promotion'])['units_sold'].mean().reset_index()
    drug_promo_pivot = drug_promo.pivot(index='Drug_ID', columns='Promotion', values='units_sold')
    
    if drug_promo_pivot.shape[1] >= 2:
        drug_promo_pivot.columns = ['No_Promo', 'With_Promo']
        drug_promo_pivot['Lift_Percent'] = (drug_promo_pivot['With_Promo'] / drug_promo_pivot['No_Promo'] - 1) * 100
        top_responsive = drug_promo_pivot.sort_values('Lift_Percent', ascending=False)
        
        print("\nTop 10 drugs most responsive to promotions:")
        print(top_responsive[['Lift_Percent']].head(10).round(1))
        
        # Save promotion analysis
        top_responsive.reset_index().to_csv('reports/promotion_effectiveness.csv')
        
        # Visualize
        plt.figure(figsize=(12, 6))
        plt.bar(top_responsive.head(10).index, top_responsive['Lift_Percent'].head(10))
        plt.title('Promotion Response by Drug (% Lift)')
        plt.ylabel('Sales Increase %')
        plt.axhline(y=0, color='red', linestyle='-')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/promotion_lift.png')
        print("✅ Saved promotion effectiveness visualization")

# 9. COMBINED FACTOR ANALYSIS
print("\n9. COMPLEX DRIVER ANALYSIS")

# Combine top categorical and numerical drivers
top_cat = categorical_cols[:3]  # Top 3 categorical
top_num = corr_df['Feature'].head(3).tolist()  # Top 3 numerical

print(f"Analyzing interplay between top factors:")
print(f"- Categorical: {top_cat}")
print(f"- Numerical: {top_num}")

# Generate cross-tabulation of top factors
if len(top_cat) >= 2 and len(top_num) >= 1:
    # Create a pivot table with 2 categorical variables and 1 numerical
    cat1, cat2 = top_cat[0], top_cat[1]
    num1 = top_num[0]
    
    # Create interaction view
    pivot = pd.pivot_table(df, 
                         values='units_sold',
                         index=cat1,
                         columns=cat2,
                         aggfunc='mean')
    
    # Save the analysis
    pivot.to_csv(f'reports/interaction_{cat1}_{cat2}.csv')
    
    # Create a heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title(f'Interaction Effect of {cat1} and {cat2} on Sales')
    plt.tight_layout()
    plt.savefig(f'figures/interaction_{cat1}_{cat2}.png')
    print(f"✅ Saved interaction analysis between {cat1} and {cat2}")
    
    # Analysis of numerical + categorical
    if 'Season' in df.columns and 'Promotion' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='available_stock', y='units_sold', 
                       hue='Season', size='Promotion',
                       sizes=(50, 200), alpha=0.7, data=df)
        plt.title('Sales vs. Stock by Season and Promotion Status')
        plt.tight_layout()
        plt.savefig('figures/multi_factor_analysis.png')
        print("✅ Saved multi-factor analysis visualization")

# 10. KEY FINDINGS SUMMARY
print("\n10. SUMMARIZING KEY SALES DRIVERS")

# Create a summary of findings
summary = []

# Add correlation findings
if not corr_df.empty:
    top_pos = corr_df[corr_df['Correlation'] > 0].head(3)
    for _, row in top_pos.iterrows():
        summary.append(f"- Strong positive correlation between '{row['Feature']}' and sales (r={row['Correlation']:.2f})")

# Add promotion findings
if 'Promotion' in df.columns and 'promotion_effectiveness.csv' in os.listdir('reports'):
    promo_data = pd.read_csv('reports/promotion_effectiveness.csv')
    avg_lift = promo_data['Lift_Percent'].mean()
    summary.append(f"- Promotions drive an average {avg_lift:.1f}% increase in sales")
    if not promo_data.empty:
        top_drug = promo_data.iloc[0]['Drug_ID']
        top_lift = promo_data.iloc[0]['Lift_Percent']
        summary.append(f"- '{top_drug}' is most responsive to promotions (+{top_lift:.1f}%)")

# Add seasonal findings if available
if 'Season' in df.columns:
    seasonal = df.groupby('Season')['units_sold'].mean().sort_values(ascending=False)
    top_season = seasonal.index[0]
    low_season = seasonal.index[-1]
    diff = (seasonal.iloc[0] / seasonal.iloc[-1] - 1) * 100
    summary.append(f"- '{top_season}' is the peak sales season, with {diff:.1f}% higher sales than '{low_season}'")

# Save summary to file
with open('reports/sales_drivers_summary.txt', 'w') as f:
    f.write("KEY FINDINGS: MEDICATION SALES DRIVERS\n")
    f.write("="*50 + "\n\n")
    f.write("Based on analysis of sales data from Rwanda health centers, \n")
    f.write("the following factors significantly influence medication demand:\n\n")
    for finding in summary:
        f.write(finding + "\n")
    
    # Add recommendations
    f.write("\nRECOMMENDATIONS:\n")
    f.write("="*50 + "\n\n")
    f.write("1. Stock Planning: Increase inventory levels before peak seasons\n")
    f.write("2. Promotions: Target high-response medications for promotional campaigns\n")
    f.write("3. Inventory Management: Monitor available stock closely as it highly correlates with sales\n")
    f.write("4. Outbreak Monitoring: Track disease outbreaks to anticipate demand spikes\n")

# Print summary
print("\nKEY FINDINGS GENERATED:")
for finding in summary:
    print(finding)
print("\n✅ Saved detailed findings to reports/sales_drivers_summary.txt")

print("\n" + "=" * 80)
print("SALES DRIVERS ANALYSIS COMPLETED")
print("=" * 80)
