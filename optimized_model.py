"""
Optimized Medication Demand Prediction
with focus on faster execution and key sales drivers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("OPTIMIZED MEDICATION DEMAND PREDICTION")
print("=" * 80)

print("\n1. LOADING AND PREPROCESSING DATA")
df = pd.read_csv('synthetic_pharma_sales.csv')
print(f" Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

date_cols = ['Date', 'sale_timestamp', 'stock_entry_timestamp', 'expiration_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

df.dropna(inplace=True)
print(f" Preprocessed data: {df.shape[0]} rows remaining")

print("\n2. FEATURE ENGINEERING")

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Quarter'] = df['Date'].dt.quarter

drug_avg_sales = df.groupby('Drug_ID')['units_sold'].mean().reset_index()
df = pd.merge(df, drug_avg_sales.rename(columns={'units_sold': 'Avg_Drug_Sales'}), on='Drug_ID', how='left')

center_avg_sales = df.groupby('Pharmacy_Name')['units_sold'].mean().reset_index()
df = pd.merge(df, center_avg_sales.rename(columns={'units_sold': 'Avg_Center_Sales'}), on='Health_Center', how='left')

df['Days_Until_Expiry'] = (df['expiration_date'] - df['Date']).dt.days
df['Days_Since_Stock_Entry'] = (df['Date'] - df['stock_entry_timestamp']).dt.days
df['Inventory_Turnover'] = df['units_sold'] / df['available_stock']

print(f" Created {df.shape[1] - 23} new features")

print("\n3. ANALYZING SALES DRIVERS")

if not os.path.exists('figures'):
    os.makedirs('figures')

if 'Season' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Season', y='units_sold', data=df)
    plt.title('Impact of Season on Medication Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/seasonal_impact.png')
    
    seasonal_stats = df.groupby('Season')['units_sold'].agg(['mean', 'median', 'std']).reset_index()
    print("\nSeasonal Impact on Sales:")
    print(seasonal_stats)
    print(" Saved seasonal impact visualization")

if 'Promotion' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Promotion', y='units_sold', data=df)
    plt.title('Impact of Promotions on Medication Sales')
    plt.tight_layout()
    plt.savefig('figures/promotion_impact.png')
    
    promo_stats = df.groupby('Promotion')['units_sold'].mean().reset_index()
    if len(promo_stats) >= 2:
        promo_lift = promo_stats.iloc[1]['units_sold'] / promo_stats.iloc[0]['units_sold'] - 1
        print(f"\nPromotion Impact: {promo_lift:.2%} average sales lift with promotions")
    print(" Saved promotion impact visualization")

print("\n4. TRAINING MODELS")

y = df['units_sold']
drop_cols = ['units_sold'] + date_cols
X_df = df.drop(columns=drop_cols)

categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
print(f" Split data: {X_train.shape[0]} training and {X_test.shape[0]} test samples")

print("\nTraining Linear Regression...")
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print(f"Linear Regression: R² = {lr_r2:.4f}, RMSE = {lr_rmse:.2f}")

best_model = lr_pipeline
best_name = "Linear Regression"

print(f"\nBest model: {best_name} with R² = {lr_r2:.4f}")

if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(best_model, f'models/best_model_{best_name.lower().replace(" ", "_")}.pkl')
print(f" Saved best model to models/best_model_{best_name.lower().replace(' ', '_')}.pkl")

print("\n5. GENERATING RESTOCK RECOMMENDATIONS")

latest_date = df['Date'].max()
latest_data = df[df['Date'] == latest_date].copy()

avg_daily_sales = df.groupby(['Drug_ID', 'Health_Center']).agg(
    avg_daily_units=('units_sold', 'mean'),
    current_stock=('available_stock', 'last')
).reset_index()

avg_daily_sales['days_of_stock_remaining'] = avg_daily_sales['current_stock'] / avg_daily_sales['avg_daily_units']
avg_daily_sales['projected_30day_demand'] = avg_daily_sales['avg_daily_units'] * 30
avg_daily_sales['restock_needed'] = avg_daily_sales['projected_30day_demand'] - avg_daily_sales['current_stock'] 
avg_daily_sales['restock_needed'] = avg_daily_sales['restock_needed'].apply(lambda x: max(0, x))

def get_priority(days):
    if days < 7:
        return "URGENT"
    elif days < 14:
        return "HIGH"
    elif days < 30:
        return "MEDIUM"
    else:
        return "LOW"

avg_daily_sales['restock_priority'] = avg_daily_sales['days_of_stock_remaining'].apply(get_priority)

if not os.path.exists('reports'):
    os.makedirs('reports')
avg_daily_sales.to_csv('reports/restock_recommendations.csv', index=False)
print(f" Saved restock recommendations to reports/restock_recommendations.csv")

print("\nUrgent restock items:")
urgent_items = avg_daily_sales[avg_daily_sales['restock_priority'] == 'URGENT']
print(urgent_items[['Drug_ID', 'Health_Center', 'current_stock', 'days_of_stock_remaining', 'restock_needed']].head(5))

print("\n6. ANALYZING KEY SALES DRIVERS")

if best_name == "Linear Regression":
    feature_names = numerical_cols.copy()
    
    ohe = best_model.named_steps['preprocessor'].transformers_[1][1]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    
    all_feature_names = feature_names + list(cat_feature_names)
    
    lr_model = best_model.named_steps['model']
    importances = np.abs(lr_model.coef_)
    
    if len(all_feature_names) >= len(importances):
        feature_importance = pd.DataFrame({
            'Feature': all_feature_names[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 sales drivers:")
        print(feature_importance.head(10))
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Sales Drivers')
        plt.tight_layout()
        plt.savefig('figures/sales_drivers.png')
        print(" Saved sales drivers visualization")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
