import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import joblib
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class MedicationDemandPredictor:
    def __init__(self, data_path, categorical_encoding='label'):
        self.data_path = data_path
        self.categorical_encoding = categorical_encoding  # 'onehot', 'label', 'ordinal', 'target'
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_importances = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.trained_pipeline = None
        self.label_encoders = {}  # Store label encoders for inverse transform

        os.makedirs('models', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        
        # Display column names and missing values for debugging
        print("\nColumn information:")
        print(self.df.info())
        print("\nMissing values per column:")
        print(self.df.isnull().sum())
        
        # Convert date columns
        date_cols = ['Date', 'sale_timestamp', 'stock_entry_timestamp', 'expiration_date']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {str(e)}")
        
        # Handle missing values more carefully
        initial_count = len(self.df)
        
        # Check if target column exists and has values
        if 'units_sold' not in self.df.columns:
            print("Error: Target column 'units_sold' not found in dataset")
            print("Available columns:", list(self.df.columns))
            return self
        
        # Only drop rows where target is missing
        self.df = self.df.dropna(subset=['units_sold'])
        print(f"Dropped {initial_count - len(self.df)} rows with missing target values")
        
        # Fill missing values in other columns
        for col in self.df.columns:
            if col != 'units_sold' and self.df[col].isnull().any():
                if self.df[col].dtype in ['object', 'string']:
                    # Fill categorical columns with 'Unknown'
                    self.df[col].fillna('Unknown', inplace=True)
                else:
                    # Fill numerical columns with median or 0
                    if self.df[col].dtype in ['int64', 'float64']:
                        median_val = self.df[col].median()
                        if pd.isna(median_val):
                            self.df[col].fillna(0, inplace=True)
                        else:
                            self.df[col].fillna(median_val, inplace=True)
        
        print(f"Final dataset shape: {self.df.shape}")
        
        # Ensure we have enough data for training
        if len(self.df) < 10:
            print(f"Warning: Only {len(self.df)} samples remaining. This may not be enough for training.")
        
        return self
    
    def engineer_features(self):
        """Create additional features from the dataset."""
        print("Engineering features...")
        
        # Check if we have data to work with
        if len(self.df) == 0:
            print("No data available for feature engineering")
            return self
        
        # Time-based features
        if 'Date' in self.df.columns:
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day
            self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
            self.df['IsWeekend'] = self.df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            self.df['Quarter'] = self.df['Date'].dt.quarter
            self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
    
            # Lag features - only if we have enough data
            if len(self.df) > 7 and all(col in self.df.columns for col in ['Drug_ID', 'Pharmacy_Name']):
                # Group by Drug and Pharmacy
                group_cols = ['Drug_ID', 'Pharmacy_Name']
                
                # Sort by date within each group
                self.df = self.df.sort_values(group_cols + ['Date'])
                
                # Create lag features (previous day, previous week)
                self.df['Prev_Day_Sales'] = self.df.groupby(group_cols)['units_sold'].shift(1)
                self.df['Prev_Week_Sales'] = self.df.groupby(group_cols)['units_sold'].shift(7)
                
                # Create rolling window features
                self.df['Rolling_7day_Mean'] = self.df.groupby(group_cols)['units_sold'].transform(
                    lambda x: x.rolling(7, min_periods=1).mean())
                
                # Fill missing values created by shifts
                lag_cols = ['Prev_Day_Sales', 'Prev_Week_Sales', 'Rolling_7day_Mean']
                for col in lag_cols:
                    if col in self.df.columns:
                        self.df[col].fillna(0, inplace=True)
        
        # Calculate stock duration (only if columns exist)
        if 'expiration_date' in self.df.columns and 'Date' in self.df.columns:
            try:
                self.df['Days_Until_Expiry'] = (self.df['expiration_date'] - self.df['Date']).dt.days
                # Fill negative or missing values
                self.df['Days_Until_Expiry'].fillna(0, inplace=True)
                self.df['Days_Until_Expiry'] = self.df['Days_Until_Expiry'].clip(lower=0)
            except Exception as e:
                print(f"Warning: Could not calculate Days_Until_Expiry: {str(e)}")
        
        # Calculate days since stock entry (only if columns exist)
        if 'stock_entry_timestamp' in self.df.columns and 'Date' in self.df.columns:
            try:
                self.df['Days_Since_Stock_Entry'] = (self.df['Date'] - self.df['stock_entry_timestamp']).dt.days
                # Fill negative or missing values
                self.df['Days_Since_Stock_Entry'].fillna(0, inplace=True)
                self.df['Days_Since_Stock_Entry'] = self.df['Days_Since_Stock_Entry'].clip(lower=0)
            except Exception as e:
                print(f"Warning: Could not calculate Days_Since_Stock_Entry: {str(e)}")
        
        # Create inventory turnover ratio (only if columns exist)
        if 'units_sold' in self.df.columns and 'available_stock' in self.df.columns:
            try:
                self.df['Inventory_Turnover'] = self.df['units_sold'] / (self.df['available_stock'] + 1)  # Add 1 to avoid division by zero
                # Fill any inf or nan values
                self.df['Inventory_Turnover'].replace([np.inf, -np.inf], 0, inplace=True)
                self.df['Inventory_Turnover'].fillna(0, inplace=True)
            except Exception as e:
                print(f"Warning: Could not calculate Inventory_Turnover: {str(e)}")
        
        # Create drug-specific features (only if enough data and columns exist)
        if len(self.df) > 1 and all(col in self.df.columns for col in ['Drug_ID', 'Pharmacy_Name']):
            try:
                # Average sales per drug
                drug_avg_sales = self.df.groupby('Drug_ID')['units_sold'].mean().reset_index()
                drug_avg_sales.columns = ['Drug_ID', 'Avg_Drug_Sales']
                self.df = pd.merge(self.df, drug_avg_sales, on='Drug_ID', how='left')
                
                # Average sales per pharmacy
                pharmacy_avg_sales = self.df.groupby('Pharmacy_Name')['units_sold'].mean().reset_index()
                pharmacy_avg_sales.columns = ['Pharmacy_Name', 'Avg_Pharmacy_Sales']
                self.df = pd.merge(self.df, pharmacy_avg_sales, on='Pharmacy_Name', how='left')
            except Exception as e:
                print(f"Warning: Could not create drug/pharmacy averages: {str(e)}")
        
        # Interaction features (only if columns exist)
        if 'Promotion' in self.df.columns and 'Holiday_Week' in self.df.columns:
            try:
                self.df['Promotion_Holiday'] = self.df['Promotion'] * self.df['Holiday_Week']
            except Exception as e:
                print(f"Warning: Could not create Promotion_Holiday: {str(e)}")
        
        if 'Disease_Outbreak' in self.df.columns and 'Effectiveness_Rating' in self.df.columns:
            try:
                self.df['Outbreak_Effectiveness'] = self.df['Disease_Outbreak'] * self.df['Effectiveness_Rating']
            except Exception as e:
                print(f"Warning: Could not create Outbreak_Effectiveness: {str(e)}")
        
        # Price-based features (only if columns exist)
        if 'Price_Per_Unit' in self.df.columns and 'Drug_ID' in self.df.columns:
            try:
                # Calculate price position relative to drug average
                drug_avg_price = self.df.groupby('Drug_ID')['Price_Per_Unit'].mean().reset_index()
                drug_avg_price.columns = ['Drug_ID', 'Avg_Drug_Price']
                self.df = pd.merge(self.df, drug_avg_price, on='Drug_ID', how='left')
                self.df['Price_Position'] = self.df['Price_Per_Unit'] / (self.df['Avg_Drug_Price'] + 0.01)  # Add small value to avoid division by zero
                # Fill any inf or nan values
                self.df['Price_Position'].replace([np.inf, -np.inf], 1, inplace=True)
                self.df['Price_Position'].fillna(1, inplace=True)
            except Exception as e:
                print(f"Warning: Could not create Price_Position: {str(e)}")
        
        print(f"Created {self.df.shape[1]} total features")
        return self
    
    def prepare_data(self):
        """Prepare data for modeling by splitting features and target."""
        print("Preparing data for modeling...")
        
        # Check if we have data
        if len(self.df) == 0:
            print("Error: No data available for modeling")
            return self
        
        # Define target
        target_column = 'units_sold'
        if target_column not in self.df.columns:
            print(f"Error: Target column '{target_column}' not found")
            return self
        
        y = self.df[target_column]
        
        # Drop unnecessary columns
        date_cols = [col for col in self.df.columns 
                    if 'date' in col.lower() or 'timestamp' in col.lower()]
        drop_cols = [target_column] + date_cols
        
        # Only drop columns that actually exist
        drop_cols = [col for col in drop_cols if col in self.df.columns]
        X = self.df.drop(columns=drop_cols)
        
        # Handle any remaining missing values
        X = X.fillna(0)
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Features matrix shape: {X.shape}")
        print(f"Categorical features: {len(self.categorical_cols)}")
        print(f"Numerical features: {len(self.numerical_cols)}")
        
        # Check if we have enough samples for train/test split
        if len(X) < 5:
            print(f"Warning: Only {len(X)} samples available. Using all data for training.")
            self.X_train = X
            self.X_test = X.iloc[:1]  # Use first row as test
            self.y_train = y
            self.y_test = y.iloc[:1]
        else:
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self
    
    def build_models(self):
        """Build several model pipelines for comparison."""
        print(f"Building model pipelines with {self.categorical_encoding} encoding...")
        
        # Create different categorical encoders based on selection
        if self.categorical_encoding == 'onehot':
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif self.categorical_encoding == 'label':
            cat_encoder = 'passthrough'  # We'll handle label encoding separately
        elif self.categorical_encoding == 'ordinal':
            cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        elif self.categorical_encoding == 'target':
            cat_encoder = 'passthrough'  # We'll handle target encoding separately
        else:
            raise ValueError(f"Unsupported encoding method: {self.categorical_encoding}")
        
        # Handle label encoding and target encoding separately
        if self.categorical_encoding == 'label':
            # Apply label encoding to categorical columns
            X_train_encoded = self.X_train.copy()
            X_test_encoded = self.X_test.copy()
            
            for col in self.categorical_cols:
                le = LabelEncoder()
                # Fit on training data and transform both train and test
                X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
                # Handle unknown categories in test set
                X_test_encoded[col] = X_test_encoded[col].astype(str)
                unknown_mask = ~X_test_encoded[col].isin(le.classes_)
                X_test_encoded[col] = X_test_encoded[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                X_test_encoded[col] = le.transform(X_test_encoded[col])
                self.label_encoders[col] = le
            
            # Update the datasets
            self.X_train = X_train_encoded
            self.X_test = X_test_encoded
            
            # Create preprocessing pipeline (only numerical scaling needed)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_cols)
                ],
                remainder='passthrough'  # Keep categorical columns as-is since they're already encoded
            )
            
        elif self.categorical_encoding == 'target':
            # Apply target encoding (mean encoding)
            X_train_encoded = self.X_train.copy()
            X_test_encoded = self.X_test.copy()
            
            for col in self.categorical_cols:
                # Calculate mean target value for each category
                target_means = self.X_train.groupby(col)[self.y_train.name].mean() if hasattr(self.y_train, 'name') else \
                              pd.concat([self.X_train[col], self.y_train], axis=1).groupby(col)[self.y_train.index.name or 0].mean()
                
                # Apply encoding
                X_train_encoded[col] = X_train_encoded[col].map(target_means)
                X_test_encoded[col] = X_test_encoded[col].map(target_means)
                
                # Fill unknown categories with global mean
                global_mean = self.y_train.mean()
                X_train_encoded[col].fillna(global_mean, inplace=True)
                X_test_encoded[col].fillna(global_mean, inplace=True)
                
                self.label_encoders[col] = target_means
            
            # Update the datasets
            self.X_train = X_train_encoded
            self.X_test = X_test_encoded
            
            # All columns are now numerical, so treat them all as numerical
            all_numerical_cols = self.numerical_cols + self.categorical_cols
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), all_numerical_cols)
                ]
            )
            
        else:
            # For ordinal and onehot encoding, use the standard pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_cols),
                    ('cat', cat_encoder, self.categorical_cols)
                ]
            )
        
        # Create multiple models with their preprocessing pipeline - prioritize Linear Regression
        self.models = {
            'Linear Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ]),
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'Support Vector Machine': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', SVR())
            ])
        }
        
        print(f"Created {len(self.models)} model pipelines with {self.categorical_encoding} encoding")
        return self
    
    def train_and_evaluate(self, cv=5):
        """Train and evaluate all models."""
        print("Training and evaluating models...")
        
        self.results = {}
        best_score = -float('inf')
        
        for name, model in self.models.items():
            start_time = time.time()
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, 
                                      pd.concat([self.X_train, self.X_test]), 
                                      pd.concat([self.y_train, self.y_test]), 
                                      cv=cv, scoring='r2')
            
            training_time = time.time() - start_time
            
            # Store results
            self.results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'CV_R²': cv_scores.mean(),
                'CV_Std': cv_scores.std(),
                'Training_Time': training_time,
                'Model': model
            }
            
            # Print results
            print(f"Results for {name}:")
            print(f"  MSE: {mse:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R² (mean): {cv_scores.mean():.4f}")
            print(f"  CV R² (std): {cv_scores.std():.4f}")
            print(f"  Training time: {training_time:.2f} seconds")
            
            # Save the model immediately after training
            self.save_individual_model(name, model, r2)
            
            # Check if this is the best model so far
            if r2 > best_score:
                best_score = r2
                self.best_model = name
        
        print(f"\nBest model: {self.best_model} (R² = {self.results[self.best_model]['R²']:.4f})")
        return self
    
    def tune_hyperparameters(self):
        """Tune hyperparameters for the best model."""
        if not self.best_model:
            print("No best model selected yet. Run train_and_evaluate first.")
            return self
        
        print(f"\nTuning hyperparameters for {self.best_model}...")
        
        # Define hyperparameter grids for different models
        param_grids = {
            'Linear Regression': {},  # No hyperparameters to tune
            'Random Forest': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            },
            'Support Vector Machine': {
                'regressor__C': [0.1, 1, 10, 100],
                'regressor__gamma': ['scale', 'auto', 0.001, 0.01],
                'regressor__kernel': ['rbf', 'linear']
            }
        }
        
        # Get the parameter grid for the best model
        param_grid = param_grids.get(self.best_model, {})
        
        if not param_grid:
            print(f"No hyperparameters to tune for {self.best_model}.")
            return self
        
        # Create grid search
        grid_search = GridSearchCV(
            self.models[self.best_model],
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        # Fit grid search
        print("Running grid search (this may take a while)...")
        grid_search.fit(self.X_train, self.y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Update the model with best parameters
        self.models[self.best_model] = grid_search.best_estimator_
        
        # Evaluate the tuned model
        y_pred = self.models[self.best_model].predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Update results
        self.results[self.best_model + ' (Tuned)'] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Model': self.models[self.best_model],
            'Best_Params': best_params
        }
        
        # Print updated results
        print(f"\nTuned {self.best_model} Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Update best model name if tuned version is better
        if r2 > self.results[self.best_model]['R²']:
            old_best = self.best_model
            self.best_model = self.best_model + ' (Tuned)'
            print(f"Tuned model performs better than the original {old_best}")
        
        # Save the trained pipeline for later use
        self.trained_pipeline = self.models[self.best_model.replace(' (Tuned)', '')]
        
        return self
    
    def analyze_feature_importance(self):
        """Analyze feature importance from the best model."""
        print("\nAnalyzing feature importance...")
        
        # Check if the best model supports feature importances
        model_name = self.best_model.replace(' (Tuned)', '')
        model = self.models[model_name]
        
        # Try to get feature names after preprocessing
        try:
            if self.categorical_encoding in ['label', 'target']:
                # For label and target encoding, feature names are the same as original
                feature_names = self.numerical_cols + self.categorical_cols
            else:
                # For onehot and ordinal encoding
                preprocessor = model.named_steps['preprocessor']
                feature_names = []
                
                # Get numerical feature names
                if self.numerical_cols:
                    feature_names.extend(self.numerical_cols)
                
                # Get categorical feature names
                if self.categorical_cols:
                    if self.categorical_encoding == 'onehot':
                        encoder = preprocessor.transformers_[1][1]
                        cat_feature_names = encoder.get_feature_names_out(self.categorical_cols)
                        feature_names.extend(cat_feature_names)
                    elif self.categorical_encoding == 'ordinal':
                        feature_names.extend(self.categorical_cols)
        except Exception as e:
            print(f"Warning: Could not get feature names: {str(e)}")
            feature_names = [f"Feature_{i}" for i in range(1000)]  # Create dummy feature names
        
        # Extract feature importances for tree-based models
        if model_name in ['Random Forest']:
            try:
                regressor = model.named_steps['regressor']
                importances = regressor.feature_importances_
                
                # Create dataframe with feature importances
                n_features = min(len(importances), len(feature_names))
                feature_importance = pd.DataFrame({
                    'Feature': feature_names[:n_features],
                    'Importance': importances[:n_features]
                }).sort_values('Importance', ascending=False)
                
                self.feature_importances = feature_importance
                
                # Plot top features
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
                plt.title(f'Top 20 Most Important Features ({model_name})')
                plt.tight_layout()
                plt.savefig('figures/feature_importance.png')
                
                print("Top 10 most important features:")
                for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
                    print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")
                
                print("\nFeature importance plot saved to 'figures/feature_importance.png'")
            except Exception as e:
                print(f"Could not calculate feature importance: {str(e)}")
        else:
            print(f"Feature importance analysis not available for {model_name}")
        
        return self
    
    def visualize_results(self):
        """Visualize model performance comparison."""
        print("\nVisualizing model performance...")
        
        # Prepare data for visualization
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['R²'] for name in model_names]
        rmse_scores = [self.results[name]['RMSE'] for name in model_names]
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'R²': r2_scores,
            'RMSE': rmse_scores
        })
        
        # Sort by R² score
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        # Visualize R² scores
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Model', y='R²', data=comparison_df)
        plt.title('Model Comparison - R² Scores (higher is better)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(comparison_df['R²']):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        plt.tight_layout()
        plt.savefig('figures/model_r2_comparison.png')
        
        # Visualize RMSE scores
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Model', y='RMSE', data=comparison_df.sort_values('RMSE'))
        plt.title('Model Comparison - RMSE Scores (lower is better)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for i, v in enumerate(comparison_df.sort_values('RMSE')['RMSE']):
            ax.text(i, v + 1, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        plt.savefig('figures/model_rmse_comparison.png')
        
        print("Model comparison visualizations saved to 'figures' directory")
        
        # Visualize actual vs predicted values for the best model
        self.plot_prediction_accuracy()
        
        return self
    
    def plot_prediction_accuracy(self):
        """Plot actual vs predicted values for the best model."""
        best_model_name = self.best_model.replace(' (Tuned)', '')
        model = self.models[best_model_name]
        
        # Generate predictions
        y_pred = model.predict(self.X_test)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'k--', lw=2)
        
        plt.title(f'Actual vs Predicted Values ({self.best_model})')
        plt.xlabel('Actual Units Sold')
        plt.ylabel('Predicted Units Sold')
        
        # Add R² annotation
        r2 = r2_score(self.y_test, y_pred)
        plt.annotate(f'R² = {r2:.4f}', 
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('figures/prediction_accuracy.png')
        print("Prediction accuracy plot saved to 'figures/prediction_accuracy.png'")
        
        # Plot prediction error distribution
        errors = self.y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f'Prediction Error Distribution ({self.best_model})')
        plt.xlabel('Prediction Error')
        plt.axvline(x=0, color='r', linestyle='--')
        
        # Add mean and std annotation
        mean_error = errors.mean()
        std_error = errors.std()
        plt.annotate(f'Mean: {mean_error:.2f}\nStd: {std_error:.2f}', 
                    xy=(0.05, 0.95), 
                    xycoords='axes fraction',
                    fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('figures/prediction_error_distribution.png')
        print("Prediction error distribution plot saved to 'figures/prediction_error_distribution.png'")
        
        return self
    
    def save_model(self):
        """Save the best model to disk."""
        print("\nSaving the best model...")
        
        # Create a clean filename from the model name
        filename = f"best_model_{self.best_model.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
        path = os.path.join('models', filename)
        
        # Save the model
        best_model_name = self.best_model.replace(' (Tuned)', '')
        joblib.dump(self.models[best_model_name], path)
        
        print(f"Best model ({self.best_model}) saved to {path}")
        return self
    
    def save_individual_model(self, model_name, model, r2_score):
        """Save an individual model immediately after training."""
        # Create a clean filename from the model name
        filename = f"{model_name.replace(' ', '_').lower()}_{self.categorical_encoding}_r2_{r2_score:.4f}.pkl"
        path = os.path.join('models', filename)
        
        # Save the model
        joblib.dump(model, path)
        
        print(f"  → Model saved to {path}")
        return self
    
    def generate_restock_recommendations(self, days_to_predict=30):
        """Generate restock recommendations based on predicted demand."""
        print("\nGenerating restock recommendations...")
        
        if not self.trained_pipeline:
            print("No trained model available. Please train the model first.")
            return None
        
        # Create a copy of latest data for each drug/location combination
        latest_date = self.df['Date'].max()
        latest_data = self.df[self.df['Date'] == latest_date].copy()
        
        # Create future dates
        future_dates = [latest_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # List to store predictions
        predictions = []
        
        # For each product and location combination
        for _, row in latest_data.iterrows():
            drug = row['Drug_ID']
            center = row['Pharmacy_Name']
            province = row['Province'] if 'Province' in row else None
            
            # For each future date
            for future_date in future_dates:
                # Create a copy of the row for this future date
                future_row = row.copy()
                future_row['Date'] = future_date
                
                # Update date-based features
                future_row['Year'] = future_date.year
                future_row['Month'] = future_date.month
                future_row['Day'] = future_date.day
                future_row['DayOfWeek'] = future_date.dayofweek
                future_row['IsWeekend'] = 1 if future_date.dayofweek >= 5 else 0
                future_row['Quarter'] = (future_date.month - 1) // 3 + 1
                
                if 'DayOfYear' in future_row:
                    future_row['DayOfYear'] = future_date.timetuple().tm_yday
                
                # Update other time-dependent features
                if 'Days_Until_Expiry' in future_row and 'expiration_date' in future_row:
                    future_row['Days_Until_Expiry'] = (future_row['expiration_date'] - future_date).days
                
                if 'Days_Since_Stock_Entry' in future_row and 'stock_entry_timestamp' in future_row:
                    future_row['Days_Since_Stock_Entry'] = (future_date - future_row['stock_entry_timestamp']).dt.days
                
                # Convert to DataFrame for prediction
                future_df = pd.DataFrame([future_row])
                
                # Drop unnecessary columns
                date_cols = [col for col in future_df.columns 
                            if 'date' in col.lower() or 'timestamp' in col.lower()]
                drop_cols = ['units_sold'] + date_cols
                future_X = future_df.drop(columns=drop_cols, errors='ignore')
                
                # Apply the same categorical encoding used during training
                if self.categorical_encoding == 'label':
                    for col in self.categorical_cols:
                        if col in future_X.columns and col in self.label_encoders:
                            le = self.label_encoders[col]
                            value = future_X[col].iloc[0]
                            if value in le.classes_:
                                future_X[col] = le.transform([str(value)])[0]
                            else:
                                future_X[col] = le.transform([le.classes_[0]])[0]  # Use first class for unknown values
                
                elif self.categorical_encoding == 'target':
                    for col in self.categorical_cols:
                        if col in future_X.columns and col in self.label_encoders:
                            target_means = self.label_encoders[col]
                            value = future_X[col].iloc[0]
                            if value in target_means:
                                future_X[col] = target_means[value]
                            else:
                                future_X[col] = target_means.mean()  # Use global mean for unknown values
                
                # Make sure columns match training data
                for col in self.X_train.columns:
                    if col not in future_X.columns:
                        future_X[col] = 0  # Add missing columns with default values
                
                future_X = future_X[self.X_train.columns]  # Ensure column order matches
                
                # Make prediction
                pred_units = self.trained_pipeline.predict(future_X)[0]
                
                # Store prediction
                prediction_entry = {
                    'Date': future_date,
                    'Drug_ID': drug,
                    'Pharmacy_Name': center,
                    'Predicted_Units': max(0, round(pred_units))
                }
                
                if province:
                    prediction_entry['Province'] = province
                
                predictions.append(prediction_entry)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Aggregate by Drug and Pharmacy
        group_cols = ['Drug_ID', 'Pharmacy_Name']
        if 'Province' in predictions_df.columns:
            group_cols.append('Province')
        
        restock_recommendations = predictions_df.groupby(group_cols)['Predicted_Units'].sum().reset_index()
        restock_recommendations.rename(columns={'Predicted_Units': f'Predicted_Demand_Next_{days_to_predict}_Days'}, inplace=True)
        
        # Get current stock levels
        current_stock = latest_data.groupby(group_cols)['available_stock'].first().reset_index()
        
        # Merge with predictions
        restock_recommendations = pd.merge(restock_recommendations, current_stock, on=group_cols)
        
        # Calculate restock amount
        restock_recommendations['Recommended_Restock'] = restock_recommendations[f'Predicted_Demand_Next_{days_to_predict}_Days'] - restock_recommendations['available_stock']
        restock_recommendations['Recommended_Restock'] = restock_recommendations['Recommended_Restock'].apply(lambda x: max(0, x))
        
        # Calculate weeks of stock remaining
        restock_recommendations['Weeks_Of_Stock_Remaining'] = restock_recommendations['available_stock'] / (restock_recommendations[f'Predicted_Demand_Next_{days_to_predict}_Days'] / 4)
        restock_recommendations['Weeks_Of_Stock_Remaining'] = restock_recommendations['Weeks_Of_Stock_Remaining'].fillna(0)
        
        # Add restock priority
        def get_priority(weeks):
            if weeks < 1:
                return "URGENT"
            elif weeks < 2:
                return "HIGH"
            elif weeks < 4:
                return "MEDIUM"
            else:
                return "LOW"
        
        restock_recommendations['Restock_Priority'] = restock_recommendations['Weeks_Of_Stock_Remaining'].apply(get_priority)
        
        # Save recommendations
        restock_recommendations.to_csv('restock_recommendations.csv', index=False)
        print(f"Restock recommendations saved to 'restock_recommendations.csv'")
        
        # Display sample
        print("\nSample restock recommendations (urgent items):")
        urgent_items = restock_recommendations[restock_recommendations['Restock_Priority'] == 'URGENT'].head(10)
        if len(urgent_items) > 0:
            print(urgent_items)
        else:
            print(restock_recommendations.head(10))
        
        return restock_recommendations

    def run_full_pipeline(self):
        """Run the complete modeling pipeline."""
        print(f"Starting medication demand prediction pipeline with {self.categorical_encoding} encoding...")
        (self.load_data()
             .engineer_features()
             .prepare_data()
             .build_models()
             .train_and_evaluate()
             .tune_hyperparameters()
             .analyze_feature_importance()
             .visualize_results()
             .save_model()
             .generate_restock_recommendations())
        
        print("\n" + "=" * 80)
        print("MEDICATION DEMAND PREDICTION PIPELINE COMPLETE")
        print("=" * 80)
        
        return self

if __name__ == "__main__":
    try:
        # You can now specify different encoding methods:
        # 'onehot' - One-hot encoding (default)
        # 'label' - Label encoding
        # 'ordinal' - Ordinal encoding
        # 'target' - Target/mean encoding
        
        encoding_method = 'label'  # Change this to test different encodings
        
        # Initialize and run the pipeline
        predictor = MedicationDemandPredictor('synthetic_pharma_sales.csv', categorical_encoding=encoding_method)
        predictor.run_full_pipeline()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
