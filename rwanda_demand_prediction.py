"""
Rwanda Medication Demand Prediction System
Forecasts future medication demand based on key identified drivers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import joblib
import os

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('reports', exist_ok=True)

print("=" * 80)
print("RWANDA MEDICATION DEMAND PREDICTION SYSTEM")
print("=" * 80)

# Load data
df = pd.read_csv('synthetic_pharma_sales.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['expiration_date'] = pd.to_datetime(df['expiration_date'])
df['stock_entry_timestamp'] = pd.to_datetime(df['stock_entry_timestamp'])
df['sale_timestamp'] = pd.to_datetime(df['sale_timestamp'])

print(f"Loaded dataset with {df.shape[0]} rows")

# Basic cleaning
df.dropna(subset=['units_sold', 'Drug_ID', 'Health_Center', 'Date'], inplace=True)
print(f"After cleaning: {df.shape[0]} rows")

# 1. FEATURE ENGINEERING
print("\n1. FEATURE ENGINEERING")

# Extract date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
df['Quarter'] = df['Date'].dt.quarter

# Create time-based features
df['Days_Until_Expiry'] = (df['expiration_date'] - df['Date']).dt.days
df['Days_Since_Stock_Entry'] = (df['Date'] - df['stock_entry_timestamp']).dt.days
df['Inventory_Turnover'] = df['units_sold'] / df['available_stock']

# Create average features by drug and health center
drug_avg = df.groupby('Drug_ID')['units_sold'].mean().reset_index()
drug_avg.columns = ['Drug_ID', 'Avg_Drug_Sales']
df = pd.merge(df, drug_avg, on='Drug_ID', how='left')

center_avg = df.groupby('Health_Center')['units_sold'].mean().reset_index()
center_avg.columns = ['Health_Center', 'Avg_Center_Sales']
df = pd.merge(df, center_avg, on='Health_Center', how='left')

# Create average features by season
season_avg = df.groupby('Season')['units_sold'].mean().reset_index()
season_avg.columns = ['Season', 'Avg_Season_Sales']
df = pd.merge(df, season_avg, on='Season', how='left')

# Create interaction features for key drivers
if 'Promotion' in df.columns and 'Season' in df.columns:
    df['Promotion_Season'] = df['Season'] + '_Promo' + df['Promotion'].astype(str)

if 'Disease_Outbreak' in df.columns and 'Availability_Score' in df.columns:
    df['Outbreak_X_Availability'] = df['Disease_Outbreak'] * df['Availability_Score']
else:
    print("Warning: 'Disease_Outbreak' or 'Availability_Score' column is missing. Skipping interaction feature creation.")
# Added a check to ensure columns exist before creating the interaction feature.

print(f"Created {df.shape[1] - 23} new features")

# 2. IDENTIFY AND ANALYZE KEY DRIVERS
print("\n2. ANALYZING KEY SALES DRIVERS")

# Identify categorical and numerical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col != 'units_sold']

# Calculate correlations with target
correlations = []
for col in numerical_cols:
    corr = df[['units_sold', col]].corr().iloc[0, 1]
    correlations.append({'Feature': col, 'Correlation': corr})

corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
print("\nTop numerical drivers of medication sales:")
print(corr_df.head(10))

# Visualize top correlations
plt.figure(figsize=(12, 8))
sns.barplot(x='Correlation', y='Feature', data=corr_df.head(10))
plt.title('Top Numerical Drivers of Medication Demand')
plt.axvline(x=0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig('figures/demand_drivers.png')
print("✅ Saved demand drivers visualization")

# 3. BUILD PREDICTION MODEL
print("\n3. BUILDING PREDICTION MODEL")

# Define features and target
target = 'units_sold'
y = df[target]

# Remove unnecessary columns
drop_cols = [target, 'Date', 'expiration_date', 'stock_entry_timestamp', 'sale_timestamp']
X = df.drop(columns=drop_cols)

# Re-identify categorical and numerical features after feature engineering
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Model features: {X.shape[1]} total")
print(f"- Categorical features: {len(categorical_cols)}")
print(f"- Numerical features: {len(numerical_cols)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
print("\nTraining Linear Regression model...")
model.fit(X_train, y_train)

# Evaluate model
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"5-Fold CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 4. ANALYZE FEATURE IMPORTANCE
print("\n4. ANALYZING FEATURE IMPORTANCE")

# Get feature names
cat_encoder = model.named_steps['preprocessor'].transformers_[1][1]
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
feature_names = np.array(numerical_cols + cat_features.tolist())

# Get feature importances
importance = model.named_steps['regressor'].coef_

# Create DataFrame of feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Display top features
print("\nTop 15 most important features for predicting medication demand:")
print(feature_importance.head(15))

# Visualize feature importance
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance for Medication Demand Prediction')
plt.tight_layout()
plt.savefig('figures/feature_importance.png')
print("✅ Saved feature importance visualization")

# 5. GENERATE FUTURE DEMAND PREDICTIONS
print("\n5. GENERATING FUTURE DEMAND PREDICTIONS")

# Get latest date in data
latest_date = df['Date'].max()
print(f"Latest data date: {latest_date}")

# Create a function to generate predictions for future dates
def predict_future_demand(model, df, days_ahead=30):
    # Create empty list to store predictions
    future_predictions = []
    
    # Get the latest date in the data
    latest_date = df['Date'].max()
    
    # Get unique drug and health center combinations
    drug_centers = df[['Drug_ID', 'Health_Center', 'Province']].drop_duplicates()
    
    # For each drug-center combination
    for _, row in drug_centers.iterrows():
        drug = row['Drug_ID']
        center = row['Health_Center']
        province = row['Province']
        
        # Filter data for this drug and center
        drug_center_data = df[(df['Drug_ID'] == drug) & (df['Health_Center'] == center)]
        
        if len(drug_center_data) == 0:
            continue
        
        # Get the most recent record
        latest_record = drug_center_data[drug_center_data['Date'] == drug_center_data['Date'].max()].iloc[0].copy()
        
        # Generate predictions for each future date
        for day in range(1, days_ahead + 1):
            future_date = latest_date + timedelta(days=day)
            
            # Create a copy of the latest record and update date-related features
            future_record = latest_record.copy()
            future_record['Date'] = future_date
            future_record['Year'] = future_date.year
            future_record['Month'] = future_date.month
            future_record['Day'] = future_date.day
            future_record['DayOfWeek'] = future_date.weekday()
            future_record['IsWeekend'] = 1 if future_date.weekday() >= 5 else 0
            future_record['Quarter'] = (future_date.month - 1) // 3 + 1
            
            # Update time-based features if possible
            if 'expiration_date' in future_record and pd.notna(future_record['expiration_date']):
                future_record['Days_Until_Expiry'] = (future_record['expiration_date'] - future_date).days
            
            if 'stock_entry_timestamp' in future_record and pd.notna(future_record['stock_entry_timestamp']):
                future_record['Days_Since_Stock_Entry'] = (future_date - future_record['stock_entry_timestamp']).days
            
            # Convert to DataFrame for prediction
            future_df = pd.DataFrame([future_record])
            
            # Remove columns not needed for prediction
            drop_cols = ['units_sold', 'Date', 'expiration_date', 'stock_entry_timestamp', 'sale_timestamp']
            future_X = future_df.drop(columns=drop_cols)
            
            # Ensure categorical columns are strings before prediction
            for col in categorical_cols:
                if col in future_X.columns:
                    future_X[col] = future_X[col].astype(str)
            # Added a safeguard to convert categorical columns to strings.
            
            # Make prediction
            pred = model.predict(future_X)[0]
            
            # Store prediction
            future_predictions.append({
                'Date': future_date,
                'Drug_ID': drug,
                'Health_Center': center,
                'Province': province,
                'Predicted_Units': max(0, round(pred))
            })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(future_predictions)
    return predictions_df

# Generate predictions for next 30 days
print("Generating demand predictions for next 30 days...")
future_demand = predict_future_demand(model, df, days_ahead=30)

# Summarize predictions
print(f"Generated predictions for {len(future_demand)} drug-location combinations")

# Aggregate by drug
drug_demand = future_demand.groupby('Drug_ID')['Predicted_Units'].sum().reset_index()
drug_demand = drug_demand.sort_values('Predicted_Units', ascending=False)

print("\nTop 10 medications by predicted demand (next 30 days):")
print(drug_demand.head(10))

# Aggregate by health center
center_demand = future_demand.groupby('Health_Center')['Predicted_Units'].sum().reset_index()
center_demand = center_demand.sort_values('Predicted_Units', ascending=False)

print("\nTop 10 health centers by predicted demand (next 30 days):")
print(center_demand.head(10))

# 6. GENERATE RESTOCK RECOMMENDATIONS
print("\n6. GENERATING RESTOCK RECOMMENDATIONS")

# Get current inventory levels
latest_inventory = df[df['Date'] == latest_date].groupby(['Drug_ID', 'Health_Center'])[['available_stock']].first().reset_index()

# Merge with predictions
restock_data = pd.merge(
    future_demand.groupby(['Drug_ID', 'Health_Center'])['Predicted_Units'].sum().reset_index(),
    latest_inventory,
    on=['Drug_ID', 'Health_Center'],
    how='left'
)

# Calculate restock amount
restock_data['restock_amount'] = restock_data['Predicted_Units'] - restock_data['available_stock']
restock_data['restock_amount'] = restock_data['restock_amount'].apply(lambda x: max(0, round(x)))

# Calculate days of supply
restock_data['days_of_supply'] = restock_data['available_stock'] / (restock_data['Predicted_Units'] / 30)
restock_data['days_of_supply'] = restock_data['days_of_supply'].fillna(100)  # Handle division by zero
restock_data['days_of_supply'] = restock_data['days_of_supply'].round(1)

# Add priority level
def get_priority(days):
    if days < 7:
        return "URGENT"
    elif days < 14:
        return "HIGH"
    elif days < 30:
        return "MEDIUM"
    else:
        return "LOW"

restock_data['priority'] = restock_data['days_of_supply'].apply(get_priority)

# Sort by priority
priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
restock_data['priority_value'] = restock_data['priority'].map(priority_order)
restock_data = restock_data.sort_values(['priority_value', 'restock_amount'], ascending=[True, False])

# Save restock recommendations
restock_data.to_csv('reports/restock_recommendations.csv', index=False)
print(f"✅ Saved restock recommendations to reports/restock_recommendations.csv")

# Display urgent items
urgent_items = restock_data[restock_data['priority'] == 'URGENT']
print(f"\nUrgent items requiring immediate restock: {len(urgent_items)}")
if len(urgent_items) > 0:
    print("\nTop 10 urgent restock needs:")
    print(urgent_items[['Drug_ID', 'Health_Center', 'available_stock', 'Predicted_Units', 'restock_amount']].head(10))

# 7. SAVE MODEL AND SUMMARY
print("\n7. SAVING MODEL AND GENERATING FINAL REPORT")

# Save the trained model
joblib.dump(model, 'models/rwanda_medication_demand_model.pkl')
print("✅ Saved prediction model to models/rwanda_medication_demand_model.pkl")

# Create a summary report
with open('reports/demand_prediction_summary.txt', 'w') as f:
    f.write("RWANDA MEDICATION DEMAND PREDICTION SYSTEM\n")
    f.write("="*60 + "\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-"*60 + "\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
    f.write(f"R² Score: {r2:.4f}\n")
    f.write(f"5-Fold Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n\n")
    
    f.write("KEY DEMAND DRIVERS\n")
    f.write("-"*60 + "\n")
    for _, row in feature_importance.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")
    
    f.write("PREDICTION SUMMARY\n")
    f.write("-"*60 + "\n")
    f.write(f"Predictions generated for next 30 days: {latest_date} to {latest_date + timedelta(days=30)}\n")
    f.write(f"Total predicted demand (all medications): {future_demand['Predicted_Units'].sum():,} units\n")
    f.write(f"Total urgent restock items: {len(urgent_items)}\n")
    f.write(f"Total restock quantity needed: {restock_data['restock_amount'].sum():,} units\n\n")
    
    f.write("NEXT STEPS\n")
    f.write("-"*60 + "\n")
    f.write("1. Implement restock recommendations\n")
    f.write("2. Monitor actual vs. predicted demand\n")
    f.write("3. Update model with new data monthly\n")
    f.write("4. Adjust inventory strategies based on seasonal patterns\n")
    f.write("5. Prepare contingency plans for disease outbreaks\n")

print("✅ Saved prediction summary to reports/demand_prediction_summary.txt")

print("\n" + "=" * 80)
print("RWANDA MEDICATION DEMAND PREDICTION SYSTEM COMPLETED")
print("=" * 80)
