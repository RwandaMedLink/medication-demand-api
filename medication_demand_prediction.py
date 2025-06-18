import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('synthetic_pharma_sales.csv')
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

df.dropna(how='all', inplace=True)

# Check for Date column (capital D) since that's what the CSV has
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['lag_1_units_sold'] = df['units_sold'].shift(1)
    df['rolling_mean_3'] = df['units_sold'].rolling(window=3).mean()
    df['rolling_std_3'] = df['units_sold'].rolling(window=3).std()
    df['month'] = df['Date'].dt.month
    # Drop rows with NaN values only in the new engineered features
    df = df.dropna(subset=['lag_1_units_sold', 'rolling_mean_3', 'rolling_std_3'])

# Ensure units_sold is numeric
df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')

# Identify categorical columns but exclude the target column
categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols = [col for col in categorical_cols if col != 'units_sold']

if len(categorical_cols) > 0:
    print(f"Dropping categorical columns: {list(categorical_cols)}")
    df = df.drop(columns=categorical_cols)

# Remove any remaining rows with NaN in target column
df = df.dropna(subset=['units_sold'])

# Drop columns that are mostly empty (like Unnamed columns)
columns_to_drop = []
for col in df.columns:
    if col.startswith('Unnamed') or df[col].isnull().sum() > len(df) * 0.5:
        columns_to_drop.append(col)

if columns_to_drop:
    print(f"Dropping mostly empty columns: {columns_to_drop}")
    df = df.drop(columns=columns_to_drop)

print(f"After preprocessing: {df.shape[0]} rows and {df.shape[1]} columns")

target_column = 'units_sold'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in data!")

# Convert Date to numeric features or drop it since we already have month
if 'Date' in df.columns:
    # Convert Date to day of year which can be useful for seasonality
    df['day_of_year'] = df['Date'].dt.dayofyear
    # Drop the original Date column as it can't be scaled
    df = df.drop(columns=['Date'])

y = df[target_column]
X = df.drop(columns=[target_column])

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features with missing values:")
for col in X.columns:
    missing_count = X[col].isnull().sum()
    if missing_count > 0:
        print(f"  {col}: {missing_count} missing values")

# Handle remaining missing values by filling with median for numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Remove any remaining NaN values if they still exist
combined = pd.concat([X, y], axis=1)
rows_before = combined.shape[0]
combined = combined.dropna()
rows_after = combined.shape[0]
print(f"Dropped {rows_before - rows_after} rows with remaining NaN values")

X = combined.drop(columns=[target_column])
y = combined[target_column]

print(f"Final dataset shape - Features: {X.shape}, Target: {y.shape}")

if X.shape[0] == 0:
    raise ValueError("No samples remaining after preprocessing! Check your data for missing values.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Machine": SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"=== {name} Results ===")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    print(f"5-Fold CV Mean R²: {np.mean(cv_scores):.2f} (Std: {np.std(cv_scores):.2f})")
    if name == "Random Forest":
        importances = model.feature_importances_
        feature_names = X.columns
        importance_dict = dict(zip(feature_names, importances))
        sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 Feature Importances (Random Forest):")
        for feat, imp in sorted_importances[:10]:
            print(f"  {feat}: {imp:.3f}")

features_to_plot = ['lag_1_units_sold', 'rolling_mean_3', 'rolling_std_3', 'month']
for feat in features_to_plot:
    if feat in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[feat], y=df['units_sold'])
        plt.title(f'{feat} vs. units_sold')
        plt.xlabel(feat)
        plt.ylabel('units_sold')
        plt.tight_layout()
        plt.savefig(f'figures/{feat}_vs_units_sold.png')
        plt.close()

plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('figures/feature_correlation_heatmap.png')
plt.close()