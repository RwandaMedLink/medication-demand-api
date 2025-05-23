import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('synthetic_pharma_sales.csv')
df.dropna(inplace=True)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['lag_1_units_sold'] = df['units_sold'].shift(1)
    df['rolling_mean_3'] = df['units_sold'].rolling(window=3).mean()
    df['rolling_std_3'] = df['units_sold'].rolling(window=3).std()
    df['month'] = df['date'].dt.month
    df = df.dropna()


categorical_cols = df.select_dtypes(include=['object']).columns
high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > max_unique_threshold]
low_cardinality_cols = [col for col in categorical_cols if df[col].nunique() <= max_unique_threshold]

print('High-cardinality columns excluded from encoding:', high_cardinality_cols)
print('Low-cardinality columns to encode:', low_cardinality_cols)

if low_cardinality_cols:
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[low_cardinality_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(low_cardinality_cols), index=df.index)
    df = pd.concat([df.drop(columns=low_cardinality_cols), encoded_df], axis=1)
else:
    print('No categorical columns to encode.')

target_column = 'units_sold'
y = df[target_column]
X = df.drop(columns=[target_column])

X = X.select_dtypes(include=[np.number])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

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