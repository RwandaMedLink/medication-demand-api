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

class PharmacyMedicationDemandPredictor:
    """
    Rwanda Pharmacy-Focused Medication Demand Prediction System
    
    This system is specifically designed for pharmacy operations in Rwanda, incorporating:
    - Rwanda's 4 seasonal patterns (Itumba, Icyi, Umuhindo, Urugaryi)
    - ATC drug classification system for inventory management
    - Pharmacy-specific business intelligence features
    - Population density and income level analysis
    - Seasonal demand multipliers for different drug categories
    
    Key Business Applications:
    - Daily inventory management decisions
    - Seasonal stock planning for malaria/respiratory seasons
    - Pricing optimization based on demographics
    - Promotional campaign effectiveness analysis
    - Multi-location pharmacy coordination
    """
    
    def __init__(self, data_path, categorical_encoding='label'):
        self.data_path = data_path
        self.categorical_encoding = categorical_encoding
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
        self.label_encoders = {}
        
        self.rwanda_seasons = {
            'Itumba': 1,    # Mar-May: Long rainy season
            'Icyi': 2,      # Jun-Aug: Long dry season
            'Umuhindo': 3,  # Sep-Nov: Short rainy season
            'Urugaryi': 4   # Dec-Feb: Short dry season
        }
        
        self.atc_categories = {
            'M01AB': 'Anti-inflammatory',      # Diclofenac, Indomethacin
            'M01AE': 'Propionic_derivatives',  # Ibuprofen, Naproxen
            'N02BA': 'Salicylic_derivatives',  # Aspirin
            'N02BE': 'Paracetamol_group',      # Paracetamol fever/pain relief
            'N02BB': 'Paracetamol_group',      # Alternative paracetamol coding
            'N05B': 'Anxiolytics',             # Diazepam, Lorazepam
            'N05C': 'Sleep_medications',       # Zolpidem, Zopiclone
            'R03': 'Respiratory_drugs',        # Salbutamol, Budesonide
            'R06': 'Antihistamines'            # Loratadine, Cetirizine
        }

        self.seasonal_multipliers = {
            'Malaria_related': {  # N02BE/B, N02BA, M01AE
                'Itumba': 1.3,    # Moderate malaria season
                'Icyi': 0.8,      # Low transmission
                'Umuhindo': 1.4,  # PEAK malaria season
                'Urugaryi': 1.1   # Moderate levels
            },
            'Respiratory': {      # R03, R06
                'Itumba': 1.5,    # Humidity-related issues
                'Icyi': 0.7,      # Lowest respiratory problems
                'Umuhindo': 1.6,  # Peak respiratory season
                'Urugaryi': 1.1   # Cold-triggered conditions
            },
            'Mental_health': {    # N05B, N05C
                'Itumba': 1.0,
                'Icyi': 1.0,
                'Umuhindo': 1.0,
                'Urugaryi': 1.2   # Holiday stress
            },
            'General': {          # Default for other drugs
                'Itumba': 1.0,
                'Icyi': 1.0,
                'Umuhindo': 1.0,
                'Urugaryi': 1.0
            }
        }

        os.makedirs('models', exist_ok=True)
        os.makedirs('figures', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
    def load_data(self):
        """
        Load and validate pharmacy dataset with specific Rwanda structure.
        
        Expected columns: 'Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'Date',
        'units_sold', 'available_stock', 'expiration_date', 'stock_entry_timestamp',
        'sale_timestamp', 'Price_Per_Unit', 'Promotion', 'Season', 'Effectiveness_Rating',
        'Population_Density', 'Income_Level'
        """
        print("Loading Rwanda pharmacy dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        
        # Validate required pharmacy columns
        required_cols = ['Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'Date', 
                        'units_sold', 'available_stock', 'Season', 'Price_Per_Unit',
                        'Promotion', 'Effectiveness_Rating', 'Population_Density', 'Income_Level']
        
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"Warning: Missing required pharmacy columns: {missing_cols}")
            print(f"Available columns: {list(self.df.columns)}")
        
        # Convert date columns
        date_cols = ['Date', 'sale_timestamp', 'stock_entry_timestamp', 'expiration_date']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {str(e)}")
        
        # Validate target column
        if 'units_sold' not in self.df.columns:
            raise ValueError("Target column 'units_sold' not found in dataset")
        
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['units_sold'])
        print(f"Removed {initial_count - len(self.df)} rows with missing units_sold")
        
        if 'Season' in self.df.columns:
            invalid_seasons = set(self.df['Season'].unique()) - set(self.rwanda_seasons.keys()) - {np.nan}
            if invalid_seasons:
                print(f"Warning: Invalid seasons found: {invalid_seasons}")
                print(f"Valid Rwanda seasons: {list(self.rwanda_seasons.keys())}")
        
        self.df = self._handle_pharmacy_missing_values()
        
        print(f"Final pharmacy dataset shape: {self.df.shape}")
        return self
    
    def _handle_pharmacy_missing_values(self):
        """Handle missing values specifically for pharmacy operations."""
        df_clean = self.df.copy()
        
        categorical_defaults = {
            'Season': 'Itumba',  # Default to most common season
            'Population_Density': 'medium',
            'Income_Level': 'medium',
            'ATC_Code': 'Unknown'
        }
        
        for col, default_val in categorical_defaults.items():
            if col in df_clean.columns:
                df_clean[col].fillna(default_val, inplace=True)
        
        numerical_cols = ['available_stock', 'Price_Per_Unit', 'Effectiveness_Rating', 'Promotion']
        for col in numerical_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
        
        return df_clean
    
    def calculate_pharmacy_seasonal_multiplier(self, season, atc_code):
        """
        Calculate pharmacy-specific seasonal demand multiplier based on Rwanda patterns.
        
        Args:
            season: Rwanda season (Itumba, Icyi, Umuhindo, Urugaryi)
            atc_code: ATC classification code
            
        Returns:
            float: Seasonal demand multiplier for pharmacy inventory planning
        """
        if season not in self.rwanda_seasons:
            return 1.0
        
        if atc_code in ['N02BE', 'N02BB', 'N02BA', 'M01AE']:
            category = 'Malaria_related'
        elif atc_code in ['R03', 'R06']:
            category = 'Respiratory'
        elif atc_code in ['N05B', 'N05C']:
            category = 'Mental_health'
        else:
            category = 'General'
        
        return self.seasonal_multipliers[category].get(season, 1.0)
    
    def engineer_focused_features(self):
        """
        Engineer pharmacy-focused features for Rwanda market analysis.
        
        Focus on 7 core pharmacy learning features:
        1. Season (Rwanda's 4 seasons)
        2. Price_Per_Unit (pricing strategy)
        3. available_stock (inventory levels)
        4. Effectiveness_Rating (customer preference)
        5. Promotion (promotional impact)
        6. Population_Density (catchment demographics)
        7. Income_Level (purchasing power)
        """
        print("Engineering pharmacy-focused features for Rwanda market...")
        
        if len(self.df) == 0:
            print("No data available for feature engineering")
            return self
        
        if 'Season' in self.df.columns:
            self.df['Season_Numeric'] = self.df['Season'].map(self.rwanda_seasons)
            self.df['Season_Numeric'].fillna(1, inplace=True)  # Default to Itumba
        
        if 'ATC_Code' in self.df.columns:
            self.df['Drug_Category'] = self.df['ATC_Code'].map(self.atc_categories)
            self.df['Drug_Category'].fillna('Other', inplace=True)
        
        if all(col in self.df.columns for col in ['Season', 'ATC_Code']):
            self.df['Seasonal_Multiplier'] = self.df.apply(
                lambda row: self.calculate_pharmacy_seasonal_multiplier(row['Season'], row['ATC_Code']), 
                axis=1
            )
        
        if all(col in self.df.columns for col in ['units_sold', 'available_stock']):
            self.df['Stock_Turnover_Ratio'] = self.df['units_sold'] / (self.df['available_stock'] + 1)
            self.df['Stock_Turnover_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
        
        if all(col in self.df.columns for col in ['Date', 'expiration_date']):
            try:
                self.df['Days_Until_Expiry'] = (self.df['expiration_date'] - self.df['Date']).dt.days
                self.df['Days_Until_Expiry'] = self.df['Days_Until_Expiry'].clip(lower=0)
                self.df['Days_Until_Expiry'].fillna(0, inplace=True)
            except Exception as e:
                print(f"Warning: Could not calculate expiration days: {str(e)}")
        
        if all(col in self.df.columns for col in ['Date', 'stock_entry_timestamp']):
            try:
                self.df['Days_Since_Stock_Entry'] = (self.df['Date'] - self.df['stock_entry_timestamp']).dt.days
                self.df['Days_Since_Stock_Entry'] = self.df['Days_Since_Stock_Entry'].clip(lower=0)
                self.df['Days_Since_Stock_Entry'].fillna(0, inplace=True)
            except Exception as e:
                print(f"Warning: Could not calculate stock entry days: {str(e)}")
        
        if len(self.df) > 7 and all(col in self.df.columns for col in ['Drug_ID', 'Pharmacy_Name', 'Date']):
            self.df = self.df.sort_values(['Pharmacy_Name', 'Drug_ID', 'Date'])
            
            self.df['Prev_Day_Sales'] = self.df.groupby(['Pharmacy_Name', 'Drug_ID'])['units_sold'].shift(1)
            self.df['Prev_Week_Sales'] = self.df.groupby(['Pharmacy_Name', 'Drug_ID'])['units_sold'].shift(7)
            
            self.df['Rolling_7day_Mean'] = self.df.groupby(['Pharmacy_Name', 'Drug_ID'])['units_sold'].transform(
                lambda x: x.rolling(7, min_periods=1).mean())
            
            lag_cols = ['Prev_Day_Sales', 'Prev_Week_Sales', 'Rolling_7day_Mean']
            for col in lag_cols:
                if col in self.df.columns:
                    self.df[col].fillna(0, inplace=True)
        
        if all(col in self.df.columns for col in ['Province', 'Drug_ID']):
            try:
                province_avg = self.df.groupby(['Province', 'Drug_ID'])['units_sold'].mean().reset_index()
                province_avg.columns = ['Province', 'Drug_ID', 'Province_Drug_Avg']
                self.df = pd.merge(self.df, province_avg, on=['Province', 'Drug_ID'], how='left')
            except Exception as e:
                print(f"Warning: Could not create province patterns: {str(e)}")
        
        if 'Income_Level' in self.df.columns:
            income_mapping = {'low': 1, 'medium': 2, 'higher': 3, 'high': 4}
            self.df['Income_Level_Numeric'] = self.df['Income_Level'].map(income_mapping)
            self.df['Income_Level_Numeric'].fillna(2, inplace=True)  # Default to medium
        
        if 'Population_Density' in self.df.columns:
            density_mapping = {'low': 1, 'medium': 2, 'high': 3}
            self.df['Population_Density_Numeric'] = self.df['Population_Density'].map(density_mapping)
            self.df['Population_Density_Numeric'].fillna(2, inplace=True)  # Default to medium
        
        if all(col in self.df.columns for col in ['Price_Per_Unit', 'Effectiveness_Rating']):
            self.df['Price_Effectiveness_Ratio'] = self.df['Price_Per_Unit'] / (self.df['Effectiveness_Rating'] + 0.1)
            self.df['Price_Effectiveness_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
        
        if all(col in self.df.columns for col in ['Promotion', 'Population_Density_Numeric', 'Income_Level_Numeric']):
            self.df['Promo_Demo_Impact'] = (self.df['Promotion'] * 
                                          self.df['Population_Density_Numeric'] * 
                                          self.df['Income_Level_Numeric'])
        
        print(f"Engineered {self.df.shape[1]} pharmacy features focusing on Rwanda market dynamics")
        return self
    
    def prepare_data(self):
        """Prepare pharmacy data for modeling with focus on 7 core features."""
        print("Preparing pharmacy data for Rwanda demand prediction...")
        
        if len(self.df) == 0:
            raise ValueError("No pharmacy data available for modeling")
        
        y = self.df['units_sold']
        
        core_features = [
            'Season_Numeric', 'Price_Per_Unit', 'available_stock', 
            'Effectiveness_Rating', 'Promotion', 'Population_Density_Numeric', 
            'Income_Level_Numeric'
        ]
        
        engineered_features = [
            'Seasonal_Multiplier', 'Stock_Turnover_Ratio', 'Days_Until_Expiry',
            'Days_Since_Stock_Entry', 'Price_Effectiveness_Ratio', 'Promo_Demo_Impact'
        ]
        
        lag_features = ['Prev_Day_Sales', 'Prev_Week_Sales', 'Rolling_7day_Mean', 'Province_Drug_Avg']
        
        # Select available features
        all_features = core_features + engineered_features + lag_features
        available_features = [col for col in all_features if col in self.df.columns]
        
        categorical_features = []
        if 'Drug_Category' in self.df.columns:
            categorical_features.append('Drug_Category')
        if 'Pharmacy_Name' in self.df.columns:
            categorical_features.append('Pharmacy_Name')
        if 'Province' in self.df.columns:
            categorical_features.append('Province')
        
        X = self.df[available_features + categorical_features].copy()
        X = X.fillna(0)
        
        self.categorical_cols = categorical_features
        self.numerical_cols = available_features
        
        print(f"Pharmacy features matrix shape: {X.shape}")
        print(f"Core pharmacy features: {len(core_features)} (Rwanda-specific)")
        print(f"Engineered features: {len([f for f in engineered_features if f in X.columns])}")
        print(f"Categorical features: {len(self.categorical_cols)}")
        
        # Split data
        if len(X) < 10:
            print(f"Warning: Limited pharmacy data ({len(X)} samples). Using minimal split.")
            test_size = max(1, int(len(X) * 0.1))
            self.X_train = X.iloc[:-test_size]
            self.X_test = X.iloc[-test_size:]
            self.y_train = y.iloc[:-test_size]
            self.y_test = y.iloc[-test_size:]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None)
        
        print(f"Training set: {self.X_train.shape[0]} pharmacy records")
        print(f"Test set: {self.X_test.shape[0]} pharmacy records")
        
        return self
    
    def build_models(self):
        """Build pharmacy-focused model pipelines for Rwanda market."""
        print(f"Building pharmacy prediction models with {self.categorical_encoding} encoding...")
        
        # Handle categorical encoding for pharmacy data
        if self.categorical_encoding == 'label':
            X_train_encoded = self.X_train.copy()
            X_test_encoded = self.X_test.copy()
            
            for col in self.categorical_cols:
                le = LabelEncoder()
                X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
                
                # Handle unknown categories in test set
                X_test_encoded[col] = X_test_encoded[col].astype(str)
                unknown_mask = ~X_test_encoded[col].isin(le.classes_)
                if unknown_mask.any():
                    X_test_encoded.loc[unknown_mask, col] = le.classes_[0]
                X_test_encoded[col] = le.transform(X_test_encoded[col])
                self.label_encoders[col] = le
            
            self.X_train = X_train_encoded
            self.X_test = X_test_encoded
            
            preprocessor = ColumnTransformer(
                transformers=[('num', StandardScaler(), self.numerical_cols)],
                remainder='passthrough'
            )
        else:
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) if self.categorical_encoding == 'onehot' else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), self.numerical_cols),
                    ('cat', cat_encoder, self.categorical_cols)
                ]
            )
        
        self.models = {
            'Linear Regression': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ]),
            'Random Forest': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
            ]),
            'Support Vector Machine': Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', SVR(kernel='rbf', C=100))
            ])
        }
        
        print(f"Created {len(self.models)} pharmacy prediction models")
        return self
    
    def train_and_evaluate(self, cv=5):
        """Train and evaluate pharmacy models with immediate saving and figure generation."""
        print("Training pharmacy models for Rwanda demand prediction...")
        
        self.results = {}
        best_score = -float('inf')
        
        for name, model in self.models.items():
            start_time = time.time()
            print(f"\n{'='*50}")
            print(f"Training {name} for Pharmacy Operations")
            print(f"{'='*50}")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate pharmacy-relevant metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation for robustness
            try:
                cv_scores = cross_val_score(model, 
                                          pd.concat([self.X_train, self.X_test]), 
                                          pd.concat([self.y_train, self.y_test]), 
                                          cv=min(cv, len(self.X_train)), scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = r2
                cv_std = 0
            
            training_time = time.time() - start_time
            
            # Store pharmacy model results
            self.results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'CV_R²': cv_mean,
                'CV_Std': cv_std,
                'Training_Time': training_time,
                'Model': model,
                'Predictions': y_pred
            }
            
            # Print pharmacy-relevant interpretation
            print(f"\n {name} Pharmacy Performance:")
            print(f"   R² Score: {r2:.4f} (Prediction Accuracy)")
            print(f"   RMSE: {rmse:.2f} units (Average Prediction Error)")
            print(f"   MAE: {mae:.2f} units (Typical Error Range)")
            print(f"   Cross-Validation R²: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"   Training Time: {training_time:.2f} seconds")
            
            # Pharmacy business interpretation
            if r2 > 0.8:
                print("    EXCELLENT: Highly reliable for pharmacy inventory planning")
            elif r2 > 0.6:
                print("    GOOD: Suitable for pharmacy demand forecasting with caution")
            elif r2 > 0.4:
                print("    MODERATE: Useful for trend analysis, supplement with domain expertise")
            else:
                print("    POOR: Requires additional data or feature engineering")
            
            self.save_individual_model(name, model, r2)
            
            self.generate_model_figures(name, model, y_pred)
            
            if r2 > best_score:
                best_score = r2
                self.best_model = name
                self.trained_pipeline = model
        
        self.generate_comparison_figures()
        
        print(f"\n Best Pharmacy Model: {self.best_model} (R² = {self.results[self.best_model]['R²']:.4f})")
        print(f" Recommended for Rwanda pharmacy demand prediction")
        
        return self
    
    def generate_model_figures(self, model_name, model, y_pred):
        """Generate pharmacy-specific figures for each model."""
        clean_name = model_name.replace(' ', '_').lower()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        
        plt.title(f'Pharmacy Demand Prediction Accuracy\n{model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Units Sold (Pharmacy Records)', fontsize=12)
        plt.ylabel('Predicted Units Sold', fontsize=12)
        
        r2 = r2_score(self.y_test, y_pred)
        plt.annotate(f'R² = {r2:.4f}\nPharmacy Prediction Accuracy', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/actual_vs_predicted_{clean_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        errors = self.y_test - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, color='skyblue', alpha=0.7)
        plt.title(f'Pharmacy Prediction Error Distribution\n{model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Error (Actual - Predicted Units)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        
        mean_error = errors.mean()
        std_error = errors.std()
        plt.annotate(f'Mean Error: {mean_error:.2f}\nStd Error: {std_error:.2f}\n\nPharmacy Planning:\n±{std_error:.0f} units typical variation', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'figures/prediction_errors_{clean_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.generate_feature_correlation_figure(clean_name)
        
        # 4. Demand drivers analysis for pharmacy business intelligence
        self.generate_demand_drivers_figure(model_name, model, clean_name)
        
        print(f"    Generated pharmacy analysis figures for {model_name}")
    
    def generate_feature_correlation_figure(self, clean_name):
        """Generate feature correlation heatmap for pharmacy feature relationships."""
        try:
            corr_data = self.X_train[self.numerical_cols].copy()
            
            corr_data['units_sold'] = self.y_train.values
            
            correlation_matrix = corr_data.corr()
            
            plt.figure(figsize=(14, 10))
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(correlation_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdYlBu_r', 
                       center=0,
                       fmt='.2f',
                       square=True,
                       cbar_kws={'label': 'Correlation Coefficient'})
            
            plt.title(f'Pharmacy Feature Correlation Analysis\nRwanda Market Relationships', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Pharmacy Features', fontsize=12)
            plt.ylabel('Pharmacy Features', fontsize=12)
            
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            plt.savefig(f'figures/feature_correlation_{clean_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            target_corr = correlation_matrix['units_sold'].drop('units_sold').abs().sort_values(ascending=False)
            print(f"  Top 5 features correlated with demand:")
            for i, (feature, corr) in enumerate(target_corr.head(5).items(), 1):
                print(f"      {i}. {feature}: {corr:.3f}")
            
        except Exception as e:
            print(f"  Could not generate correlation figure: {str(e)}")
    
    def generate_demand_drivers_figure(self, model_name, model, clean_name):
        """Generate demand drivers analysis showing key factors influencing pharmacy sales."""
        try:
            demand_data = self.X_train[self.numerical_cols].copy()
            demand_data['units_sold'] = self.y_train.values
            
            impact_scores = {}
            
            for feature in self.numerical_cols:
                if feature in demand_data.columns:
                    correlation = abs(demand_data[feature].corr(demand_data['units_sold']))
                    if not pd.isna(correlation):
                        impact_scores[feature] = correlation
            
            sorted_impacts = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)
            top_drivers = sorted_impacts[:10] 
            
            if len(top_drivers) == 0:
                print(f"  No demand drivers data available")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            features = [item[0] for item in top_drivers]
            impacts = [item[1] for item in top_drivers]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            bars = ax1.barh(range(len(features)), impacts, color=colors)
            
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features)
            ax1.set_xlabel('Impact Score (Correlation with Demand)', fontsize=11)
            ax1.set_title(f'Top Pharmacy Demand Drivers\n{model_name}', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            for i, (bar, impact) in enumerate(zip(bars, impacts)):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{impact:.3f}', va='center', fontsize=9)
            
            ax1.invert_yaxis()
            
            if 'Season_Numeric' in demand_data.columns and 'Seasonal_Multiplier' in demand_data.columns:
                seasonal_impact = demand_data.groupby('Season_Numeric')['units_sold'].mean()
                
                seasons = ['Urugaryi\n(Dec-Feb)', 'Itumba\n(Mar-May)', 'Icyi\n(Jun-Aug)', 'Umuhindo\n(Sep-Nov)']
                season_values = []
                
                for i in range(1, 5):
                    if i in seasonal_impact.index:
                        season_values.append(seasonal_impact[i])
                    else:
                        season_values.append(0)
                
                colors_seasonal = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                bars_seasonal = ax2.bar(range(len(seasons)), season_values, color=colors_seasonal, alpha=0.8)
                
                ax2.set_xticks(range(len(seasons)))
                ax2.set_xticklabels(seasons, fontsize=10)
                ax2.set_ylabel('Average Units Sold', fontsize=11)
                ax2.set_title('Rwanda Seasonal Demand Patterns\nPharmacy Business Intelligence', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                for bar, value in zip(bars_seasonal, season_values):
                    if value > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(season_values)*0.01, 
                                f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                peak_season_idx = season_values.index(max(season_values)) if max(season_values) > 0 else 0
                peak_season = seasons[peak_season_idx].split('\n')[0]
                ax2.text(0.5, 0.95, f'Peak Season: {peak_season}', transform=ax2.transAxes, 
                        ha='center', va='top', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
            else:
                ax2.text(0.5, 0.5, 'Seasonal Data\nNot Available\nfor Analysis', 
                        transform=ax2.transAxes, ha='center', va='center', 
                        fontsize=14, style='italic')
                ax2.set_title('Rwanda Seasonal Analysis', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'figures/demand_drivers_{clean_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   Top 3 pharmacy demand drivers:")
            for i, (feature, impact) in enumerate(top_drivers[:3], 1):
                business_insight = self._get_demand_driver_insight(feature, impact)
                print(f"      {i}. {feature}: {impact:.3f} - {business_insight}")
            
        except Exception as e:
            print(f"   ⚠️  Could not generate demand drivers figure: {str(e)}")
    
    def _get_demand_driver_insight(self, feature, impact):
        """Get business insight for demand drivers."""
        insights = {
            'Season_Numeric': "Rwanda seasonal patterns drive medication demand cycles",
            'Price_Per_Unit': "Pricing strategy directly impacts pharmacy accessibility and sales",
            'available_stock': "Inventory availability influences customer purchasing behavior", 
            'Effectiveness_Rating': "Drug effectiveness perception drives customer loyalty and repeat purchases",
            'Promotion': "Promotional activities significantly boost pharmacy sales volume",
            'Population_Density_Numeric': "Catchment area demographics determine pharmacy demand potential",
            'Income_Level_Numeric': "Customer purchasing power affects medication accessibility",
            'Seasonal_Multiplier': "Rwanda climate seasons affect disease patterns and drug demand",
            'Stock_Turnover_Ratio': "Inventory efficiency indicates pharmacy operational performance",
            'Days_Until_Expiry': "Product freshness urgency influences sales velocity",
            'Price_Effectiveness_Ratio': "Value perception affects customer medication choices",
            'Promo_Demo_Impact': "Targeted promotions work better in specific demographic areas",
            'Rolling_7day_Mean': "Recent sales trends indicate emerging demand patterns",
            'Province_Drug_Avg': "Regional demand variations reflect local health patterns"
        }
        
        return insights.get(feature, f"Key factor influencing pharmacy demand ({impact:.1%} correlation)")

    def analyze_feature_importance(self):
        """Analyze pharmacy feature importance with Rwanda business insights."""
        print("\n🔍 Analyzing Pharmacy Feature Importance for Rwanda Market...")
        
        model_name = self.best_model.replace(' (Tuned)', '')
        model = self.models[model_name]
        
        if model_name == 'Random Forest':
            try:
                regressor = model.named_steps['regressor']
                importances = regressor.feature_importances_
                
                if self.categorical_encoding == 'label':
                    feature_names = self.numerical_cols + self.categorical_cols
                else:
                    feature_names = self.numerical_cols.copy()
                    if self.categorical_cols:
                        feature_names.extend(self.categorical_cols)
                
                n_features = min(len(importances), len(feature_names))
                self.feature_importances = pd.DataFrame({
                    'Feature': feature_names[:n_features],
                    'Importance': importances[:n_features]
                }).sort_values('Importance', ascending=False)
                
                plt.figure(figsize=(14, 10))
                top_features = self.feature_importances.head(15)
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                         '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
                
                bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                               color=colors[:len(top_features)])
                plt.yticks(range(len(top_features)), top_features['Feature'])
                plt.xlabel('Feature Importance (Impact on Pharmacy Demand Prediction)', fontsize=12)
                plt.title('Rwanda Pharmacy Feature Importance Analysis\nKey Factors Driving Medication Demand', 
                         fontsize=14, fontweight='bold')
                
                for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
                    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{importance:.3f}', va='center', fontsize=10)
                
                plt.gca().invert_yaxis()
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                plt.savefig('figures/feature_importance_pharmacy.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("\n Top 10 Pharmacy Business Insights (Rwanda Market):")
                for i, (_, row) in enumerate(self.feature_importances.head(10).iterrows(), 1):
                    importance = row['Importance']
                    feature = row['Feature']
                    
                    insight = self._get_pharmacy_feature_insight(feature, importance)
                    print(f"{i:2d}. {feature}: {importance:.4f} - {insight}")
                
                print(f"\n Feature importance analysis saved to 'figures/feature_importance_pharmacy.png'")
                
            except Exception as e:
                print(f"Could not analyze feature importance: {str(e)}")
        else:
            print(f"Feature importance analysis not available for {model_name}")
        
        return self
    
    def generate_pharmacy_seasonal_patterns(self):
        """Generate Rwanda pharmacy seasonal demand pattern visualization."""
        print("\n Analyzing Rwanda Pharmacy Seasonal Patterns...")
        
        if 'Season' not in self.df.columns or 'Drug_Category' not in self.df.columns:
            print("Seasonal or drug category data not available")
            return self
        
        seasonal_data = self.df.groupby(['Season', 'Drug_Category'])['units_sold'].mean().reset_index()
        
        seasonal_pivot = seasonal_data.pivot(index='Drug_Category', columns='Season', values='units_sold')
        
        for season in self.rwanda_seasons.keys():
            if season not in seasonal_pivot.columns:
                seasonal_pivot[season] = 0
        
        season_order = ['Urugaryi', 'Itumba', 'Icyi', 'Umuhindo']  # Dec-Feb, Mar-May, Jun-Aug, Sep-Nov
        seasonal_pivot = seasonal_pivot.reindex(columns=[s for s in season_order if s in seasonal_pivot.columns])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(seasonal_pivot, annot=True, cmap='YlOrRd', fmt='.1f', 
                   cbar_kws={'label': 'Average Units Sold'})
        
        plt.title('Rwanda Pharmacy Seasonal Demand Patterns by Drug Category\n(Business Intelligence for Inventory Planning)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Rwanda Seasons', fontsize=12)
        plt.ylabel('Drug Categories (ATC Classification)', fontsize=12)
        
        season_descriptions = {
            'Urugaryi': 'Dec-Feb\n(Short Dry)',
            'Itumba': 'Mar-May\n(Long Rainy)', 
            'Icyi': 'Jun-Aug\n(Long Dry)',
            'Umuhindo': 'Sep-Nov\n(Short Rainy)'
        }
        
        current_labels = plt.gca().get_xticklabels()
        new_labels = [season_descriptions.get(label.get_text(), label.get_text()) for label in current_labels]
        plt.gca().set_xticklabels(new_labels)
        
        plt.tight_layout()
        plt.savefig('figures/pharmacy_seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" Pharmacy seasonal pattern analysis saved to 'figures/pharmacy_seasonal_patterns.png'")
        return self
    
    def predict_pharmacy_demand_range(self, pharmacy_name, start_date, end_date, drug_list=None):
        """
        Predict demand for specific pharmacy over date range.
        
        Args:
            pharmacy_name: Name of the pharmacy
            start_date: Start date for prediction
            end_date: End date for prediction  
            drug_list: List of Drug_IDs to predict (None for all drugs)
            
        Returns:
            DataFrame with daily predictions for pharmacy
        """
        print(f"\n Predicting demand for {pharmacy_name} from {start_date} to {end_date}")
        
        if not self.trained_pipeline:
            print("No trained model available. Please train the model first.")
            return None
        
        pharmacy_data = self.df[self.df['Pharmacy_Name'] == pharmacy_name].copy()
        if len(pharmacy_data) == 0:
            print(f"No data found for pharmacy: {pharmacy_name}")
            return None
        
        latest_records = pharmacy_data.groupby('Drug_ID').last().reset_index()
        
        if drug_list:
            latest_records = latest_records[latest_records['Drug_ID'].isin(drug_list)]
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        predictions = []
        
        for _, drug_record in latest_records.iterrows():
            for date in date_range:
                pred_record = drug_record.copy()
                pred_record['Date'] = date
                
                pred_record['Season'] = self._get_rwanda_season(date)
                pred_record['Season_Numeric'] = self.rwanda_seasons.get(pred_record['Season'], 1)
                
                if 'ATC_Code' in pred_record:
                    pred_record['Seasonal_Multiplier'] = self.calculate_pharmacy_seasonal_multiplier(
                        pred_record['Season'], pred_record['ATC_Code'])
                
                pred_df = pd.DataFrame([pred_record])
                
                pred_features = self._prepare_prediction_features(pred_df)
                
                if pred_features is not None:
                    prediction = self.trained_pipeline.predict(pred_features)[0]
                    
                    predictions.append({
                        'Date': date,
                        'Pharmacy_Name': pharmacy_name,
                        'Drug_ID': drug_record['Drug_ID'],
                        'Season': pred_record['Season'],
                        'Predicted_Units': max(0, round(prediction))
                    })
        
        result_df = pd.DataFrame(predictions)
        
        # Save results
        filename = f"reports/pharmacy_demand_prediction_{pharmacy_name}_{start_date}_{end_date}.csv"
        result_df.to_csv(filename.replace(' ', '_'), index=False)
        
        print(f" Pharmacy demand predictions saved to: {filename}")
        print(f" Predicted {len(result_df)} drug-day combinations")
        
        return result_df
    
    def generate_pharmacy_restock_alerts(self, pharmacy_name=None, days_ahead=30):
        """Generate automated restock alerts for pharmacy management."""
        print(f"\n Generating Pharmacy Restock Alerts ({days_ahead} days ahead)")
        
        if not self.trained_pipeline:
            print("No trained model available. Please train the model first.")
            return None
        
        if pharmacy_name:
            pharmacy_data = self.df[self.df['Pharmacy_Name'] == pharmacy_name]
            if len(pharmacy_data) == 0:
                print(f"No data found for pharmacy: {pharmacy_name}")
                return None
        else:
            pharmacy_data = self.df
        
        latest_inventory = pharmacy_data.groupby(['Pharmacy_Name', 'Drug_ID']).last().reset_index()
        
        alerts = []
        
        for _, item in latest_inventory.iterrows():
            future_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, days_ahead+1)]
            total_predicted_demand = 0
            
            for future_date in future_dates:
                season = self._get_rwanda_season(future_date)
                seasonal_multiplier = self.calculate_pharmacy_seasonal_multiplier(season, item.get('ATC_Code', ''))
                
                base_demand = item['units_sold'] * seasonal_multiplier
                total_predicted_demand += base_demand
            
            current_stock = item['available_stock']
            safety_stock = total_predicted_demand * 0.2  # 20% safety buffer
            recommended_restock = max(0, (total_predicted_demand + safety_stock) - current_stock)
            
            days_of_stock = current_stock / (total_predicted_demand / days_ahead) if total_predicted_demand > 0 else 999
            
            if days_of_stock < 7:
                urgency = "CRITICAL"
            elif days_of_stock < 14:
                urgency = "HIGH"
            elif days_of_stock < 21:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"
            
            alerts.append({
                'Pharmacy_Name': item['Pharmacy_Name'],
                'Drug_ID': item['Drug_ID'],
                'Current_Stock': current_stock,
                'Predicted_Demand_30d': round(total_predicted_demand),
                'Days_Of_Stock_Remaining': round(days_of_stock, 1),
                'Recommended_Restock': round(recommended_restock),
                'Urgency': urgency,
                'Province': item.get('Province', 'Unknown')
            })
        
        alerts_df = pd.DataFrame(alerts)
        alerts_df = alerts_df.sort_values(['Urgency', 'Days_Of_Stock_Remaining'])
        
        filename = f"reports/pharmacy_restock_alerts_{datetime.now().strftime('%Y%m%d')}.csv"
        alerts_df.to_csv(filename, index=False)
        
        critical_alerts = alerts_df[alerts_df['Urgency'] == 'CRITICAL']
        if len(critical_alerts) > 0:
            print(f"\n CRITICAL RESTOCK ALERTS ({len(critical_alerts)} items):")
            print(critical_alerts.head(10).to_string(index=False))
        
        print(f"\n Complete restock alerts saved to: {filename}")
        return alerts_df
    
    def analyze_pharmacy_performance(self, pharmacy_name):
        """Analyze individual pharmacy performance and provide business insights."""
        print(f"\n Analyzing Performance for {pharmacy_name}")
        
        pharmacy_data = self.df[self.df['Pharmacy_Name'] == pharmacy_name]
        if len(pharmacy_data) == 0:
            print(f"No data found for pharmacy: {pharmacy_name}")
            return None
        
        total_sales = pharmacy_data['units_sold'].sum()
        avg_daily_sales = pharmacy_data['units_sold'].mean()
        total_drugs = pharmacy_data['Drug_ID'].nunique()
        avg_stock_turnover = pharmacy_data['Stock_Turnover_Ratio'].mean() if 'Stock_Turnover_Ratio' in pharmacy_data.columns else 0
        
        seasonal_performance = pharmacy_data.groupby('Season')['units_sold'].sum().to_dict()
        
        report = {
            'Pharmacy_Name': pharmacy_name,
            'Province': pharmacy_data['Province'].iloc[0] if 'Province' in pharmacy_data.columns else 'Unknown',
            'Total_Sales_Period': total_sales,
            'Average_Daily_Sales': avg_daily_sales,
            'Number_of_Drugs': total_drugs,
            'Average_Stock_Turnover': avg_stock_turnover,
            'Seasonal_Performance': seasonal_performance,
            'Top_5_Drugs': top_drugs.to_dict(),
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d')
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Pharmacy Performance Analysis: {pharmacy_name}', fontsize=16, fontweight='bold')
        
        seasons = list(seasonal_performance.keys())
        sales = list(seasonal_performance.values())
        axes[0,0].bar(seasons, sales, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
        axes[0,0].set_title('Seasonal Sales Performance')
        axes[0,0].set_ylabel('Total Units Sold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        axes[0,1].barh(range(len(top_drugs)), top_drugs.values, color='skyblue')
        axes[0,1].set_yticks(range(len(top_drugs)))
        axes[0,1].set_yticklabels(top_drugs.index)
        axes[0,1].set_title('Top 5 Performing Drugs')
        axes[0,1].set_xlabel('Total Units Sold')
        
        if 'Date' in pharmacy_data.columns:
            daily_sales = pharmacy_data.groupby('Date')['units_sold'].sum()
            axes[1,0].plot(daily_sales.index, daily_sales.values, color='green', alpha=0.7)
            axes[1,0].set_title('Daily Sales Trend')
            axes[1,0].set_ylabel('Units Sold')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        if 'available_stock' in pharmacy_data.columns:
            axes[1,1].hist(pharmacy_data['available_stock'], bins=20, color='orange', alpha=0.7)
            axes[1,1].set_title('Stock Level Distribution')
            axes[1,1].set_xlabel('Available Stock')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        filename = f"reports/pharmacy_performance_{pharmacy_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        report_filename = f"reports/pharmacy_analysis_{pharmacy_name.replace(' ', '_')}.csv"
        pd.DataFrame([report]).to_csv(report_filename, index=False)
        
        print(f" Performance analysis completed for {pharmacy_name}")
        print(f" Charts saved to: {filename}")
        print(f" Report saved to: {report_filename}")
        
        return report
    
    def _get_rwanda_season(self, date):
        """Get Rwanda season for a given date."""
        month = date.month
        if month in [12, 1, 2]:
            return 'Urugaryi'  # Short dry season
        elif month in [3, 4, 5]:
            return 'Itumba'    # Long rainy season
        elif month in [6, 7, 8]:
            return 'Icyi'      # Long dry season
        else:  # 9, 10, 11
            return 'Umuhindo'  # Short rainy season
    
    def _prepare_prediction_features(self, pred_df):
        """Prepare features for prediction matching training format."""
        try:
            drop_cols = ['units_sold', 'Date', 'sale_timestamp', 'stock_entry_timestamp', 'expiration_date']
            features = pred_df.drop(columns=[col for col in drop_cols if col in pred_df.columns])
            
            for col in self.X_train.columns:
                if col not in features.columns:
                    features[col] = 0
            
            if self.categorical_encoding == 'label':
                for col in self.categorical_cols:
                    if col in features.columns and col in self.label_encoders:
                        le = self.label_encoders[col]
                        value = str(features[col].iloc[0])
                        if value in le.classes_:
                            features[col] = le.transform([value])[0]
                        else:
                            features[col] = le.transform([le.classes_[0]])[0]
            
            features = features[self.X_train.columns]
            return features
            
        except Exception as e:
            print(f"Error preparing prediction features: {str(e)}")
            return None
    
    def save_individual_model(self, model_name, model, r2_score):
        """Save individual pharmacy model with performance metrics."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f"pharmacy_{model_name.replace(' ', '_').lower()}_{self.categorical_encoding}_r2_{r2_score:.4f}_{timestamp}.pkl"
        path = os.path.join('models', filename)
        
        model_data = {
            'model': model,
            'performance': {
                'r2_score': r2_score,
                'model_name': model_name,
                'encoding': self.categorical_encoding,
                'training_date': timestamp,
                'rwanda_seasons': self.rwanda_seasons,
                'atc_categories': self.atc_categories,
                'seasonal_multipliers': self.seasonal_multipliers
            }
        }
        
        joblib.dump(model_data, path)
        print(f"    Pharmacy model saved: {path}")
        return self
    
    def run_full_pipeline(self):
        """Run the complete pharmacy prediction pipeline for Rwanda market."""
        print(" Starting Rwanda Pharmacy Medication Demand Prediction Pipeline")
        print("="*80)
        
        (self.load_data()
             .engineer_focused_features()
             .prepare_data()
             .build_models()
             .train_and_evaluate()
             .analyze_feature_importance()
             .generate_pharmacy_seasonal_patterns())
        
        print("\n" + "="*80)
        print("🇷🇼 RWANDA PHARMACY PREDICTION PIPELINE COMPLETE")
        print("="*80)
        print(" Business Intelligence Reports Generated:")
        print("   • Model performance comparisons")
        print("   • Pharmacy feature importance analysis")
        print("   • Rwanda seasonal demand patterns")
        print("   • Individual model predictions")
        print("\n Ready for Pharmacy Operations:")
        print("   • Daily inventory management")
        print("   • Seasonal stock planning")
        print("   • Restock alert generation")
        print("   • Performance analysis")
        
        return self

if __name__ == "__main__":
    try:
        encoding_method = 'label'  # Optimal for pharmacy categorical data
        
        predictor = PharmacyMedicationDemandPredictor(
            'synthetic_pharma_sales.csv', 
            categorical_encoding=encoding_method
        )
        
        predictor.run_full_pipeline()
        
        print("\n" + "="*60)
        print(" EXAMPLE PHARMACY BUSINESS OPERATIONS")
        print("="*60)
        
        print("\n1. Generating pharmacy restock alerts...")

        print("\n Rwanda Pharmacy Prediction System Ready for Deployment!")
        
    except Exception as e:
        print(f" Error in pharmacy prediction pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
