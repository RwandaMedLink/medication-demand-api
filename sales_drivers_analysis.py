import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import os

# Create figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

print("=" * 80)
print("MEDICATION SALES DRIVERS ANALYSIS")
print("=" * 80)

# Load the dataset
try:
    df = pd.read_csv('synthetic_pharma_sales.csv')
    print(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Convert date columns to datetime
    date_cols = ['Date', 'sale_timestamp', 'stock_entry_timestamp', 'expiration_date']
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"✅ Converted {col} to datetime")
            except Exception as e:
                print(f"⚠️ Could not convert {col}: {str(e)}")
    
    # 1. SEASONAL ANALYSIS
    print("\n1. ANALYZING SEASONAL PATTERNS")
    
    if 'Season' in df.columns and 'Drug_ID' in df.columns:
        plt.figure(figsize=(14, 8))
        season_drug_sales = df.groupby(['Season', 'Drug_ID'])['units_sold'].mean().reset_index()
        
        # Get top 10 drugs by sales volume
        top_drugs = df.groupby('Drug_ID')['units_sold'].sum().nlargest(10).index
        season_drug_sales = season_drug_sales[season_drug_sales['Drug_ID'].isin(top_drugs)]
        
        # Create seasonal analysis plot
        sns.barplot(x='Season', y='units_sold', hue='Drug_ID', data=season_drug_sales)
        plt.title('Seasonal Impact on Top 10 Drugs by Sales Volume')
        plt.xlabel('Season')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.legend(title='Drug', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('figures/seasonal_drug_sales.png')
        print("✅ Saved seasonal analysis plot to 'figures/seasonal_drug_sales.png'")
        
        # Calculate seasonal variation for each drug
        pivot_seasonal = season_drug_sales.pivot(index='Drug_ID', columns='Season', values='units_sold')
        pivot_seasonal['Max_Season'] = pivot_seasonal.idxmax(axis=1)
        pivot_seasonal['Max_Sales'] = pivot_seasonal.max(axis=1)
        pivot_seasonal['Min_Season'] = pivot_seasonal.idxmin(axis=1)
        pivot_seasonal['Min_Sales'] = pivot_seasonal.min(axis=1)
        pivot_seasonal['Seasonal_Variation'] = pivot_seasonal['Max_Sales'] / pivot_seasonal['Min_Sales']
        
        seasonal_insights = pivot_seasonal.sort_values('Seasonal_Variation', ascending=False)
        
        print("\nTop 5 drugs with strongest seasonal patterns:")
        for drug in seasonal_insights.head(5).index:
            max_season = seasonal_insights.loc[drug, 'Max_Season']
            min_season = seasonal_insights.loc[drug, 'Min_Season']
            variation = seasonal_insights.loc[drug, 'Seasonal_Variation']
            print(f"- {drug}: {variation:.2f}x higher sales in {max_season} vs. {min_season}")
    
    # 2. PROMOTION IMPACT ANALYSIS
    print("\n2. ANALYZING PROMOTION IMPACT")
    
    if 'Promotion' in df.columns:
        # Convert to categorical for better visualization
        df['Promotion'] = df['Promotion'].map({0: 'No Promotion', 1: 'With Promotion'})
        
        plt.figure(figsize=(14, 8))
        promo_impact = df.groupby(['Drug_ID', 'Promotion'])['units_sold'].mean().reset_index()
        
        # Get top drugs by promotion impact
        promo_pivot = promo_impact.pivot(index='Drug_ID', columns='Promotion', values='units_sold')
        promo_pivot['Promo_Lift'] = promo_pivot['With Promotion'] / promo_pivot['No Promotion']
        top_promo_drugs = promo_pivot.sort_values('Promo_Lift', ascending=False).head(10).index
        
        # Filter for visualization
        promo_impact_viz = promo_impact[promo_impact['Drug_ID'].isin(top_promo_drugs)]
        
        # Create promotion analysis plot
        sns.barplot(x='Drug_ID', y='units_sold', hue='Promotion', data=promo_impact_viz)
        plt.title('Promotion Impact on Top 10 Responsive Drugs')
        plt.xlabel('Drug')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.legend(title='Promotion Status')
        plt.tight_layout()
        plt.savefig('figures/promotion_impact.png')
        print("✅ Saved promotion impact plot to 'figures/promotion_impact.png'")
        
        # Calculate and display promotion lift for top drugs
        print("\nTop 5 drugs with highest promotion lift:")
        for drug in promo_pivot.sort_values('Promo_Lift', ascending=False).head(5).index:
            lift = promo_pivot.loc[drug, 'Promo_Lift']
            print(f"- {drug}: {lift:.2f}x sales lift with promotions")
    
    # 3. PRICE ELASTICITY ANALYSIS
    print("\n3. ANALYZING PRICE ELASTICITY")
    
    if 'Price_Per_Unit' in df.columns and 'Drug_ID' in df.columns:
        # Calculate price elasticity for each drug
        price_elasticity = {}
        
        for drug in df['Drug_ID'].unique():
            drug_data = df[df['Drug_ID'] == drug]
            
            if len(drug_data) > 10:  # Ensure enough data points
                # Calculate price and sales correlation
                corr = drug_data[['Price_Per_Unit', 'units_sold']].corr().iloc[0, 1]
                
                # Calculate average price and sales
                avg_price = drug_data['Price_Per_Unit'].mean()
                avg_sales = drug_data['units_sold'].mean()
                
                # Calculate price elasticity (approximate)
                if abs(corr) > 0.1:  # Only if there's meaningful correlation
                    elasticity = corr * (avg_price / avg_sales)
                    price_elasticity[drug] = elasticity
        
        # Convert to DataFrame for easier analysis
        elasticity_df = pd.DataFrame(list(price_elasticity.items()), columns=['Drug_ID', 'Price_Elasticity'])
        elasticity_df['Abs_Elasticity'] = elasticity_df['Price_Elasticity'].abs()
        elasticity_df = elasticity_df.sort_values('Abs_Elasticity', ascending=False)
        
        # Create elasticity visualization
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Drug_ID', y='Price_Elasticity', data=elasticity_df.head(15))
        plt.title('Price Elasticity of Top 15 Most Price-Sensitive Drugs')
        plt.xlabel('Drug')
        plt.ylabel('Price Elasticity (negative values = elastic)')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/price_elasticity.png')
        print("✅ Saved price elasticity plot to 'figures/price_elasticity.png'")
        
        # Display insights
        print("\nPrice elasticity insights:")
        elastic_drugs = elasticity_df[elasticity_df['Price_Elasticity'] < -0.5].head(5)
        inelastic_drugs = elasticity_df[elasticity_df['Price_Elasticity'] > -0.2].tail(5)
        
        if not elastic_drugs.empty:
            print("\nTop price-sensitive drugs (elastic):")
            for _, row in elastic_drugs.iterrows():
                print(f"- {row['Drug_ID']}: elasticity = {row['Price_Elasticity']:.2f}")
                
        if not inelastic_drugs.empty:
            print("\nLeast price-sensitive drugs (inelastic):")
            for _, row in inelastic_drugs.iterrows():
                print(f"- {row['Drug_ID']}: elasticity = {row['Price_Elasticity']:.2f}")
    
    # 4. GEOGRAPHIC ANALYSIS
    print("\n4. ANALYZING GEOGRAPHIC PATTERNS")
    
    if 'Province' in df.columns:
        plt.figure(figsize=(14, 8))
        province_sales = df.groupby(['Province', 'Drug_ID'])['units_sold'].mean().reset_index()
        
        # Get top drugs
        top_drugs = df.groupby('Drug_ID')['units_sold'].sum().nlargest(8).index
        province_sales = province_sales[province_sales['Drug_ID'].isin(top_drugs)]
        
        # Create geographic analysis plot
        sns.barplot(x='Province', y='units_sold', hue='Drug_ID', data=province_sales)
        plt.title('Geographic Sales Patterns for Top 8 Drugs')
        plt.xlabel('Province')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.legend(title='Drug', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('figures/geographic_sales.png')
        print("✅ Saved geographic sales plot to 'figures/geographic_sales.png'")
        
        # Calculate geographic variation
        pivot_geo = province_sales.pivot(index='Drug_ID', columns='Province', values='units_sold')
        pivot_geo['Max_Province'] = pivot_geo.idxmax(axis=1)
        pivot_geo['Min_Province'] = pivot_geo.idxmin(axis=1)
        pivot_geo['Geo_Variation'] = pivot_geo.max(axis=1) / pivot_geo.min(axis=1)
        
        # Display geographic insights
        print("\nGeographic sales patterns:")
        for drug in pivot_geo.sort_values('Geo_Variation', ascending=False).head(5).index:
            max_prov = pivot_geo.loc[drug, 'Max_Province']
            min_prov = pivot_geo.loc[drug, 'Min_Province']
            variation = pivot_geo.loc[drug, 'Geo_Variation']
            print(f"- {drug}: {variation:.2f}x higher sales in {max_prov} vs. {min_prov}")
    
    # 5. KEY SALES DRIVERS (FEATURE IMPORTANCE)
    print("\n5. IDENTIFYING KEY SALES DRIVERS")
    
    # Prepare data for modeling
    # Convert categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes
    
    # Create feature matrix
    X = df.drop(columns=['units_sold'] + date_cols)
    y = df['units_sold']
    
    # Get feature importance using permutation importance
    print("Calculating feature importance using permutation importance...")
    
    # Create feature importance plot
    plt.figure(figsize=(14, 10))
    sns.barplot(x=importances.importances_mean, y=X.columns)
    plt.title('Top Features Driving Medication Sales (Permutation Importance)')
    plt.tight_layout()
    plt.savefig('figures/sales_drivers.png')
    print("✅ Saved sales drivers plot to 'figures/sales_drivers.png'")
    
    # Display top drivers
    print("\nTop 10 factors driving medication sales:")
    for i, (feature, importance) in enumerate(zip(X.columns, importances.importances_mean), 1):
        if i > 10:
            break
        print(f"{i}. {feature} (importance: {importance:.4f})")
    
    # 6. TEMPORAL TRENDS ANALYSIS
    print("\n6. ANALYZING TEMPORAL TRENDS")
    
    if 'Date' in df.columns:
        # Create time-based aggregation
        time_sales = df.groupby(['Date', 'Drug_ID'])['units_sold'].sum().reset_index()
        
        # Get top 5 drugs by sales
        top_time_drugs = df.groupby('Drug_ID')['units_sold'].sum().nlargest(5).index
        time_sales_top = time_sales[time_sales['Drug_ID'].isin(top_time_drugs)]
        
        # Create temporal plot
        plt.figure(figsize=(16, 8))
        for drug in top_time_drugs:
            drug_data = time_sales_top[time_sales_top['Drug_ID'] == drug]
            plt.plot(drug_data['Date'], drug_data['units_sold'], label=drug)
        
        plt.title('Sales Trends Over Time for Top 5 Drugs')
        plt.xlabel('Date')
        plt.ylabel('Units Sold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/temporal_trends.png')
        print("✅ Saved temporal trends plot to 'figures/temporal_trends.png'")
        
        # Analyze long-term trends
        # Group by month for clearer trend visualization
        df['YearMonth'] = df['Date'].dt.to_period('M')
        monthly_trends = df.groupby(['YearMonth', 'Drug_ID'])['units_sold'].mean().reset_index()
        
        # Convert period to datetime for plotting
        monthly_trends['YearMonth'] = monthly_trends['YearMonth'].dt.to_timestamp()
        
        # Check for growth/decline
        trend_analysis = {}
        
        for drug in top_time_drugs:
            drug_monthly = monthly_trends[monthly_trends['Drug_ID'] == drug]
            
            if len(drug_monthly) > 3:  # Need at least a few months of data
                # Simple linear regression for trend
                x = np.arange(len(drug_monthly))
                y = drug_monthly['units_sold'].values
                
                # Calculate slope using polyfit
                slope, _ = np.polyfit(x, y, 1)
                
                # Normalize by average sales to get percentage growth
                avg_sales = drug_monthly['units_sold'].mean()
                monthly_growth = (slope / avg_sales) * 100
                
                trend_analysis[drug] = monthly_growth
        
        # Display trend insights
        print("\nLong-term sales trends (monthly growth rate):")
        for drug, growth in sorted(trend_analysis.items(), key=lambda x: x[1], reverse=True):
            trend_type = "growing" if growth > 0.5 else "declining" if growth < -0.5 else "stable"
            print(f"- {drug}: {growth:.2f}% monthly ({trend_type})")
    
    # 7. HEALTH CENTER TYPE ANALYSIS
    if 'Center_Type' in df.columns:
        print("\n7. ANALYZING HEALTH CENTER TYPE IMPACT")
        
        plt.figure(figsize=(14, 8))
        center_sales = df.groupby(['Center_Type', 'Drug_ID'])['units_sold'].mean().reset_index()
        
        # Get top drugs
        center_sales = center_sales[center_sales['Drug_ID'].isin(top_drugs)]
        
        # Create center type analysis plot
        sns.barplot(x='Center_Type', y='units_sold', hue='Drug_ID', data=center_sales)
        plt.title('Sales by Health Center Type for Top Drugs')
        plt.xlabel('Center Type')
        plt.ylabel('Average Units Sold')
        plt.xticks(rotation=45)
        plt.legend(title='Drug', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('figures/center_type_sales.png')
        print("✅ Saved health center type plot to 'figures/center_type_sales.png'")
    
    print("\n" + "=" * 80)
    print("SALES DRIVERS ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAnalysis results and visualizations have been saved to the 'figures' directory.")
    print("The following files were generated:")
    print("- figures/seasonal_drug_sales.png")
    print("- figures/promotion_impact.png")
    print("- figures/price_elasticity.png")
    print("- figures/geographic_sales.png")
    print("- figures/sales_drivers.png")
    print("- figures/temporal_trends.png")
    print("- figures/center_type_sales.png")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
