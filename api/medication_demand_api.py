from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import traceback
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

class MedicationDemandAPI:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.model_loaded = False
        self.feature_order = None
        
    def load_model(self, model_path='models/linear_regression_label_r2_0.9986.pkl'):
        """Load the trained model and encoders."""
        try:
            possible_paths = [
                model_path,
                f'../{model_path}',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'linear_regression_label_r2_0.9986.pkl'),
                os.path.join('models', 'linear_regression_label_r2_0.9986.pkl')
            ]
            
            model_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    print(f"Model loaded from {path}")
                    
                    import sklearn
                    print(f"Current sklearn version: {sklearn.__version__}")
                    
                    print(f"Model type: {type(self.model)}")
                    if hasattr(self.model, 'steps'):
                        print("Pipeline steps:")
                        for i, (name, transformer) in enumerate(self.model.steps):
                            print(f"  {i}: {name} -> {type(transformer)}")
                    
                    model_found = True
                    
                    model_dir = os.path.dirname(path)
                    encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
                    if os.path.exists(encoders_path):
                        self.label_encoders = joblib.load(encoders_path)
                        print("Label encoders loaded")
                    
                    try:
                        if hasattr(self.model, 'feature_names_in_'):
                            self.feature_order = list(self.model.feature_names_in_)
                            print(f"Model expects {len(self.feature_order)} features")
                        elif hasattr(self.model, 'steps') and len(self.model.steps) > 0:
                            first_step = self.model.steps[0][1]
                            if hasattr(first_step, 'feature_names_in_'):
                                self.feature_order = list(first_step.feature_names_in_)
                    except Exception as e:
                        print(f"Could not determine feature order: {e}")
                    
                    self.model_loaded = True
                    model_type = self.inspect_model()
                    print(f"Model type: {model_type}")
                    
                    return True
            
            if not model_found:
                print(f"Model file not found in any of these locations:")
                for path in possible_paths:
                    print(f"  - {os.path.abspath(path)}")
                return False
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return False

    def preprocess_input(self, data):
        """Preprocess input data for prediction."""
        try:
            df = pd.DataFrame([data])
            
            # First, ensure all values are properly converted to strings for date parsing
            date_cols = ['Date', 'expiration_date', 'stock_entry_timestamp', 'sale_timestamp']
            for col in date_cols:
                if col in df.columns:
                    # Convert to string first, then to datetime
                    df[col] = pd.to_datetime(df[col].astype(str), errors='coerce')
            
            if 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['DayOfWeek'] = df['Date'].dt.dayofweek
                df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
                df['Quarter'] = df['Date'].dt.quarter
                df['DayOfYear'] = df['Date'].dt.dayofyear
            
            if 'expiration_date' in df.columns and 'Date' in df.columns:
                df['Days_Until_Expiry'] = (df['expiration_date'] - df['Date']).dt.days
                df['Days_Until_Expiry'] = df['Days_Until_Expiry'].clip(lower=0)
            
            if 'stock_entry_timestamp' in df.columns and 'Date' in df.columns:
                df['Days_Since_Stock_Entry'] = (df['Date'] - df['stock_entry_timestamp']).dt.days
                df['Days_Since_Stock_Entry'] = df['Days_Since_Stock_Entry'].clip(lower=0)
            
            if 'available_stock' in df.columns:
                df['Inventory_Turnover'] = 0 
            
            # Improved safe numeric conversion function
            def safe_numeric_convert(value, default=0):
                """Safely convert value to numeric, handling various input types."""
                try:
                    # Handle None/NaN cases
                    if value is None:
                        return float(default)
                    
                    # Handle pandas Series
                    if isinstance(value, pd.Series):
                        if len(value) > 0:
                            value = value.iloc[0]
                        else:
                            return float(default)
                    
                    # Handle pandas DataFrame
                    if isinstance(value, pd.DataFrame):
                        if not value.empty:
                            value = value.iloc[0, 0]
                        else:
                            return float(default)
                    
                    # Convert to string first, then to numeric
                    str_value = str(value).strip()
                    
                    # Handle empty strings
                    if str_value == '' or str_value.lower() in ['nan', 'none', 'null']:
                        return float(default)
                    
                    # Try direct numeric conversion
                    numeric_value = pd.to_numeric(str_value, errors='coerce')
                    
                    # Check if conversion was successful
                    if pd.isna(numeric_value):
                        return float(default)
                    
                    return float(numeric_value)
                    
                except Exception as e:
                    print(f"Warning: Could not convert {value} to numeric: {e}")
                    return float(default)
            
            # Extract values safely
            stock_val = safe_numeric_convert(df.get('available_stock', pd.Series([0])), 0)
            price_val = safe_numeric_convert(df.get('Price_Per_Unit', pd.Series([30.0])), 30.0)
            
            df['Avg_Drug_Sales'] = stock_val * 0.1
            df['Prev_Day_Sales'] = stock_val * 0.05
            df['Avg_Pharmacy_Sales'] = stock_val * 0.15
            df['Prev_Week_Sales'] = stock_val * 0.3
            df['Rolling_7day_Mean'] = stock_val * 0.08
            df['Avg_Drug_Price'] = price_val
            
            # Create interaction features with proper numeric conversion
            disease_outbreak = safe_numeric_convert(df.get('Disease_Outbreak', pd.Series([0])), 0)
            effectiveness = safe_numeric_convert(df.get('Effectiveness_Rating', pd.Series([0])), 0)
            promotion = safe_numeric_convert(df.get('Promotion', pd.Series([0])), 0)
            holiday = safe_numeric_convert(df.get('Holiday_Week', pd.Series([0])), 0)
            
            df['Outbreak_Effectiveness'] = disease_outbreak * effectiveness
            df['Price_Position'] = price_val / price_val if price_val > 0 else 1.0
            df['Promotion_Holiday'] = promotion * holiday
            
            # Handle categorical columns with better error handling
            categorical_cols = ['Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'Season', 
                              'Supply_Chain_Delay', 'Income_Level', 'Population_Density']
            
            for col in categorical_cols:
                if col in df.columns:
                    try:
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            # Extract the actual value from the Series
                            if isinstance(df[col], pd.Series):
                                value = str(df[col].iloc[0])
                            else:
                                value = str(df[col])
                            
                            if value in le.classes_:
                                df[col] = le.transform([value])[0]
                            else:
                                df[col] = le.transform([le.classes_[0]])[0]
                        else:
                            # Create a simple hash-based encoding for unknown categorical columns
                            if isinstance(df[col], pd.Series):
                                value = str(df[col].iloc[0])
                            else:
                                value = str(df[col])
                            df[col] = abs(hash(value)) % 1000
                    except Exception as e:
                        print(f"Warning: Error encoding {col}: {e}")
                        df[col] = 0
            
            expected_features = [
                'Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'available_stock',
                'Price_Per_Unit', 'Promotion', 'Season', 'Disease_Outbreak',
                'Supply_Chain_Delay', 'Effectiveness_Rating', 'Competitor_Count',
                'Time_On_Market', 'Population_Density', 'Income_Level', 'Holiday_Week',
                'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear',
                'Days_Until_Expiry', 'Days_Since_Stock_Entry', 'Inventory_Turnover',
                'Avg_Drug_Sales', 'Prev_Day_Sales', 'Avg_Pharmacy_Sales', 'Outbreak_Effectiveness',
                'Price_Position', 'Prev_Week_Sales', 'Rolling_7day_Mean', 'Avg_Drug_Price', 'Promotion_Holiday'
            ]
            
            # Ensure all expected features exist
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Drop date columns that shouldn't be in the final feature set
            drop_cols = ['Date', 'expiration_date', 'stock_entry_timestamp', 'sale_timestamp', 'units_sold']
            drop_cols = [col for col in drop_cols if col in df.columns]
            df = df.drop(columns=drop_cols)
            
            # Reindex to ensure proper column order
            df = df.reindex(columns=expected_features, fill_value=0.0)
            
            # Enhanced final data type conversion
            for col in df.columns:
                try:
                    # Get the column values
                    col_values = df[col]
                    
                    # Handle different data types
                    if col_values.dtype == 'object':
                        # Convert object columns to numeric
                        df[col] = pd.to_numeric(col_values.astype(str), errors='coerce')
                    
                    # Fill any NaN values
                    df[col] = df[col].fillna(0.0)
                    
                    # Ensure float64 dtype
                    df[col] = df[col].astype(np.float64, errors='ignore')
                    
                    # Final safety check - replace non-finite values
                    mask = ~np.isfinite(df[col])
                    if mask.any():
                        df.loc[mask, col] = 0.0
                        
                except Exception as e:
                    df[col] = np.full(len(df), 0.0, dtype=np.float64)
            
            # Verify all columns are numeric
            for col in df.columns:
                if df[col].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(np.float64)
            
            return df
            
        except Exception as e:
            return None
    
    def inspect_model(self):
        """Inspect the loaded model structure to understand its requirements."""
        if not self.model_loaded:
            return None
        
        try:
            # Check if model is a pipeline
            if hasattr(self.model, 'steps'):
                return 'pipeline'
            elif hasattr(self.model, 'predict'):
                return 'simple'
            else:
                return 'unknown'
        except Exception as e:
            return None

    def create_simple_features(self, data):
        """Create a simplified feature set that bypasses categorical encoding issues."""
        try:
            # Extract key numeric features that don't need encoding
            features = {}
            
            # Direct numeric features
            features['available_stock'] = float(data.get('available_stock', 0))
            features['Price_Per_Unit'] = float(data.get('Price_Per_Unit', 30.0))
            features['Promotion'] = float(data.get('Promotion', 0))
            features['Disease_Outbreak'] = float(data.get('Disease_Outbreak', 0))
            features['Effectiveness_Rating'] = float(data.get('Effectiveness_Rating', 5))
            features['Competitor_Count'] = float(data.get('Competitor_Count', 3))
            features['Time_On_Market'] = float(data.get('Time_On_Market', 24))
            features['Holiday_Week'] = float(data.get('Holiday_Week', 0))
            
            # Date features
            if 'Date' in data:
                try:
                    date_obj = pd.to_datetime(str(data['Date']))
                    features['Year'] = float(date_obj.year)
                    features['Month'] = float(date_obj.month)
                    features['Day'] = float(date_obj.day)
                    features['DayOfWeek'] = float(date_obj.dayofweek)
                    features['IsWeekend'] = float(1 if date_obj.dayofweek >= 5 else 0)
                    features['Quarter'] = float(date_obj.quarter)
                    features['DayOfYear'] = float(date_obj.dayofyear)
                except:
                    # Default date features
                    features.update({
                        'Year': 2024.0, 'Month': 1.0, 'Day': 1.0, 'DayOfWeek': 0.0,
                        'IsWeekend': 0.0, 'Quarter': 1.0, 'DayOfYear': 1.0
                    })
            
            # Calculated date features with defaults
            features['Days_Until_Expiry'] = 60.0  # Default 60 days
            features['Days_Since_Stock_Entry'] = 30.0  # Default 30 days
            
            # Hash-based categorical encoding (consistent across calls)
            def simple_hash_encode(value, max_val=1000):
                return float(abs(hash(str(value))) % max_val)
            
            features['Pharmacy_Name'] = simple_hash_encode(data.get('Pharmacy_Name', 'default'))
            features['Province'] = simple_hash_encode(data.get('Province', 'default'))
            features['Drug_ID'] = simple_hash_encode(data.get('Drug_ID', 'default'))
            features['ATC_Code'] = simple_hash_encode(data.get('ATC_Code', 'default'))
            features['Season'] = simple_hash_encode(data.get('Season', 'default'))
            features['Supply_Chain_Delay'] = simple_hash_encode(data.get('Supply_Chain_Delay', 'Medium'))
            features['Income_Level'] = simple_hash_encode(data.get('Income_Level', 'medium'))
            features['Population_Density'] = simple_hash_encode(data.get('Population_Density', 'medium'))
            
            # Engineered features
            stock_val = features['available_stock']
            price_val = features['Price_Per_Unit']
            
            features['Inventory_Turnover'] = 0.0
            features['Avg_Drug_Sales'] = stock_val * 0.1
            features['Prev_Day_Sales'] = stock_val * 0.05
            features['Avg_Pharmacy_Sales'] = stock_val * 0.15
            features['Prev_Week_Sales'] = stock_val * 0.3
            features['Rolling_7day_Mean'] = stock_val * 0.08
            features['Avg_Drug_Price'] = price_val
            features['Outbreak_Effectiveness'] = features['Disease_Outbreak'] * features['Effectiveness_Rating']
            features['Price_Position'] = 1.0
            features['Promotion_Holiday'] = features['Promotion'] * features['Holiday_Week']
            
            return features
            
        except Exception as e:
            print(f"Error creating simple features: {e}")
            return None

    def predict(self, input_data):
        """Make prediction using the loaded model."""
        if not self.model_loaded:
            return None, "Model not loaded"
        
        try:
            # Approach 1: Try with preprocessed data
            try:
                processed_data = self.preprocess_input(input_data)
                if processed_data is not None:
                    # Clean the data one more time
                    processed_data = processed_data.replace([np.inf, -np.inf], 0.0)
                    processed_data = processed_data.fillna(0.0)
                    
                    # Convert all to float64
                    for col in processed_data.columns:
                        processed_data[col] = processed_data[col].astype(np.float64)
                    
                    # Debug: Print data shape and types
                    print(f"Processed data shape: {processed_data.shape}")
                    print(f"Processed data columns: {list(processed_data.columns)}")
                    print(f"Data sample: {processed_data.iloc[0].head()}")
                    
                    prediction = self.model.predict(processed_data)[0]
                    prediction = max(0, round(prediction))
                    return prediction, "Success"
            except Exception as preprocess_error:
                print(f"Preprocessing approach failed: {preprocess_error}")
                print(f"Full error: {traceback.format_exc()}")
            
            # Approach 2: Use simple features that bypass categorical encoding
            try:
                simple_features = self.create_simple_features(input_data)
                if simple_features is not None:
                    print(f"Simple features created: {len(simple_features)} features")
                    
                    # Create DataFrame with expected feature order
                    if self.feature_order is not None:
                        print(f"Using model feature order: {len(self.feature_order)} features")
                        # Use model's expected feature order
                        feature_vector = []
                        for feature_name in self.feature_order:
                            if feature_name in simple_features:
                                feature_vector.append(simple_features[feature_name])
                            else:
                                feature_vector.append(0.0)  # Default value for missing features
                        
                        # Create DataFrame
                        feature_df = pd.DataFrame([feature_vector], columns=self.feature_order)
                    else:
                        print("No feature order found, using default order")
                        # Use all available features
                        expected_features = [
                            'Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'available_stock',
                            'Price_Per_Unit', 'Promotion', 'Season', 'Disease_Outbreak',
                            'Supply_Chain_Delay', 'Effectiveness_Rating', 'Competitor_Count',
                            'Time_On_Market', 'Population_Density', 'Income_Level', 'Holiday_Week',
                            'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear',
                            'Days_Until_Expiry', 'Days_Since_Stock_Entry', 'Inventory_Turnover',
                            'Avg_Drug_Sales', 'Prev_Day_Sales', 'Avg_Pharmacy_Sales', 'Outbreak_Effectiveness',
                            'Price_Position', 'Prev_Week_Sales', 'Rolling_7day_Mean', 'Avg_Drug_Price', 'Promotion_Holiday'
                        ]
                        
                        feature_vector = []
                        for feature_name in expected_features:
                            if feature_name in simple_features:
                                feature_vector.append(simple_features[feature_name])
                            else:
                                feature_vector.append(0.0)
                        
                        feature_df = pd.DataFrame([feature_vector], columns=expected_features)
                    
                    # Ensure all values are float64
                    for col in feature_df.columns:
                        feature_df[col] = feature_df[col].astype(np.float64)
                    
                    print(f"Feature DataFrame shape: {feature_df.shape}")
                    print(f"Feature DataFrame dtypes: {feature_df.dtypes.unique()}")
                    
                    prediction = self.model.predict(feature_df)[0]
                    prediction = max(0, round(prediction))
                    return prediction, "Success (simplified features)"
                    
            except Exception as simple_error:
                print(f"Simple features approach failed: {simple_error}")
                print(f"Full error: {traceback.format_exc()}")
            
            # Approach 3: Try to bypass pipeline preprocessing
            if hasattr(self.model, 'steps'):
                try:
                    # Get the final estimator
                    final_estimator = self.model.steps[-1][1]
                    
                    # Create basic numeric features
                    basic_features = self.create_simple_features(input_data)
                    if basic_features is not None:
                        # Convert to array format that most models expect
                        feature_array = np.array([list(basic_features.values())]).astype(np.float64)
                        
                        # Use only the final estimator
                        prediction = final_estimator.predict(feature_array)[0]
                        prediction = max(0, round(prediction))
                        return prediction, "Success (direct estimator)"
                        
                except Exception as direct_error:
                    print(f"Direct estimator approach failed: {direct_error}")
            
            # If all model approaches fail, return error with debug info
            return None, "Model prediction failed - check sklearn version compatibility"
            
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"

demand_predictor = MedicationDemandAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': demand_predictor.model_loaded,
        'model_type': 'Linear Regression'
    })

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load or reload the model."""
    model_path = request.json.get('model_path', '../models/linear_regression_label_r2_0.9986.pkl')
    success = demand_predictor.load_model(model_path)
    
    return jsonify({
        'success': success,
        'message': 'Linear Regression model loaded successfully' if success else 'Failed to load model'
    })

@app.route('/predict', methods=['POST'])
def predict_demand():
    """Predict medication demand for a specific date and pharmacy."""
    try:
        input_data = request.json
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        required_fields = ['Pharmacy_Name', 'Province', 'Drug_ID', 'Date']
        missing_fields = [field for field in required_fields if field not in input_data]
        
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        prediction, message = demand_predictor.predict(input_data)
        
        if prediction is None:
            return jsonify({'error': message}), 500
        
        return jsonify({
            'predicted_demand': int(prediction),
            'message': message,
            'model_type': 'Linear Regression',
            'input_data': input_data
        })
        
    except Exception as e:
        error_msg = f"Error in prediction endpoint: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict medication demand for multiple records."""
    try:
        input_data = request.json
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Handle different input formats
        records = None
        if 'records' in input_data:
            records = input_data['records']
        elif isinstance(input_data, list):
            # Direct list of records
            records = input_data
        elif isinstance(input_data, dict) and len(input_data) > 0:
            # Single record wrapped in dict, convert to list
            records = [input_data]
        
        if not records:
            return jsonify({
                'error': 'No records provided. Expected format: {"records": [...]} or direct array [...]',
                'example': {
                    'records': [
                        {
                            'Pharmacy_Name': 'CityMeds 795',
                            'Province': 'Kigali',
                            'Drug_ID': 'DICLOFENAC',
                            'Date': '2024-01-01'
                        }
                    ]
                }
            }), 400
        
        if not isinstance(records, list):
            return jsonify({'error': 'Records must be a list/array'}), 400
        
        predictions = []
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                predictions.append({
                    'record_index': i,
                    'predicted_demand': None,
                    'message': 'Invalid record format - must be an object/dict',
                    'input_data': record
                })
                continue
            
            # Check for required fields
            required_fields = ['Pharmacy_Name', 'Province', 'Drug_ID', 'Date']
            missing_fields = [field for field in required_fields if field not in record]
            
            if missing_fields:
                predictions.append({
                    'record_index': i,
                    'predicted_demand': None,
                    'message': f'Missing required fields: {missing_fields}',
                    'input_data': record
                })
                continue
            
            prediction, message = demand_predictor.predict(record)
            predictions.append({
                'record_index': i,
                'predicted_demand': int(prediction) if prediction is not None else None,
                'message': message,
                'input_data': record
            })
        
        # Calculate summary statistics
        successful_predictions = [p for p in predictions if p['predicted_demand'] is not None]
        failed_predictions = [p for p in predictions if p['predicted_demand'] is None]
        
        return jsonify({
            'predictions': predictions,
            'summary': {
                'total_records': len(records),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(failed_predictions),
                'success_rate': f"{(len(successful_predictions)/len(records)*100):.1f}%" if records else "0%"
            },
            'model_type': 'Linear Regression'
        })
        
    except Exception as e:
        error_msg = f"Error in batch prediction endpoint: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({
            'error': error_msg,
            'example_format': {
                'records': [
                    {
                        'Pharmacy_Name': 'CityMeds 795',
                        'Province': 'Kigali',
                        'Drug_ID': 'DICLOFENAC',
                        'Date': '2024-01-01',
                        'available_stock': 470,
                        'Price_Per_Unit': 33.04
                    }
                ]
            }
        }), 500

@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    """Web form for making predictions."""
    if request.method == 'GET':
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medication Demand Prediction - Linear Regression</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .form-group { margin: 10px 0; }
                label { display: inline-block; width: 200px; }
                input, select { width: 200px; padding: 5px; }
                button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
                .result { margin-top: 20px; padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; }
                .header { background: #e9ecef; padding: 15px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Medication Demand Prediction</h1>
                <p><strong>Model:</strong> Linear Regression | <strong>API Status:</strong> Active</p>
            </div>
            <form method="POST">
                <div class="form-group">
                    <label>Pharmacy Name:</label>
                    <input type="text" name="Pharmacy_Name" value="CityMeds 795" required>
                </div>
                <div class="form-group">
                    <label>Province:</label>
                    <input type="text" name="Province" value="Kigali" required>
                </div>
                <div class="form-group">
                    <label>Drug ID:</label>
                    <input type="text" name="Drug_ID" value="DICLOFENAC" required>
                </div>
                <div class="form-group">
                    <label>ATC Code:</label>
                    <input type="text" name="ATC_Code" value="M01AB" required>
                </div>
                <div class="form-group">
                    <label>Date:</label>
                    <input type="date" name="Date" value="2024-01-01" required>
                </div>
                <div class="form-group">
                    <label>Available Stock:</label>
                    <input type="number" name="available_stock" value="470" required>
                </div>
                <div class="form-group">
                    <label>Expiration Date:</label>
                    <input type="date" name="expiration_date" value="2024-02-28" required>
                </div>
                <div class="form-group">
                    <label>Stock Entry Date:</label>
                    <input type="date" name="stock_entry_timestamp" value="2023-12-06" required>
                </div>
                <div class="form-group">
                    <label>Price Per Unit:</label>
                    <input type="number" step="0.01" name="Price_Per_Unit" value="33.04" required>
                </div>
                <div class="form-group">
                    <label>Promotion:</label>
                    <select name="Promotion">
                        <option value="0">No</option>
                        <option value="1" selected>Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Season:</label>
                    <input type="text" name="Season" value="Urugaryi" required>
                </div>
                <div class="form-group">
                    <label>Disease Outbreak:</label>
                    <select name="Disease_Outbreak">
                        <option value="0">No</option>
                        <option value="1" selected>Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Supply Chain Delay:</label>
                    <input type="text" name="Supply_Chain_Delay" value="Medium" required>
                </div>
                <div class="form-group">
                    <label>Effectiveness Rating:</label>
                    <input type="number" name="Effectiveness_Rating" value="5" min="1" max="10" required>
                </div>
                <div class="form-group">
                    <label>Competitor Count:</label>
                    <input type="number" name="Competitor_Count" value="4" required>
                </div>
                <div class="form-group">
                    <label>Time On Market:</label>
                    <input type="number" name="Time_On_Market" value="47" required>
                </div>
                <div class="form-group">
                    <label>Population Density:</label>
                    <input type="text" name="Population_Density" value="high" required>
                </div>
                <div class="form-group">
                    <label>Income Level:</label>
                    <input type="text" name="Income_Level" value="higher" required>
                </div>
                <div class="form-group">
                    <label>Holiday Week:</label>
                    <select name="Holiday_Week">
                        <option value="0">No</option>
                        <option value="1" selected>Yes</option>
                    </select>
                </div>
                <button type="submit">Predict Demand</button>
            </form>
        </body>
        </html>
        '''
    
    else:  # POST request
        try:
            # Convert form data to appropriate types
            form_data = dict(request.form)
            
            # Convert numeric fields
            numeric_fields = ['available_stock', 'Price_Per_Unit', 'Promotion', 'Disease_Outbreak', 
                            'Effectiveness_Rating', 'Competitor_Count', 'Time_On_Market', 'Holiday_Week']
            
            for field in numeric_fields:
                if field in form_data:
                    form_data[field] = float(form_data[field]) if '.' in str(form_data[field]) else int(form_data[field])
            
            # Make prediction
            prediction, message = demand_predictor.predict(form_data)
            
            result_html = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .result {{ margin-top: 20px; padding: 20px; background: #f8f9fa; border: 1px solid #dee2e6; }}
                    .success {{ background: #d4edda; border-color: #c3e6cb; }}
                    .error {{ background: #f8d7da; border-color: #f5c6cb; }}
                    .header {{ background: #e9ecef; padding: 15px; margin-bottom: 20px; }}
                    a {{ display: inline-block; margin-top: 15px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Medication Demand Prediction Result</h1>
                    <p><strong>Model:</strong> Linear Regression</p>
                </div>
                <div class="result {'success' if prediction is not None else 'error'}">
                    <h3>Prediction Result</h3>
                    <p><strong>Predicted Demand:</strong> {prediction if prediction is not None else "Error"} units</p>
                    <p><strong>Status:</strong> {message}</p>
                    <p><strong>Pharmacy:</strong> {form_data.get('Pharmacy_Name', 'N/A')}</p>
                    <p><strong>Drug:</strong> {form_data.get('Drug_ID', 'N/A')}</p>
                    <p><strong>Date:</strong> {form_data.get('Date', 'N/A')}</p>
                    <a href="/predict_form">Make Another Prediction</a>
                </div>
            </body>
            </html>
            '''
            
            return result_html
            
        except Exception as e:
            return f'''
            <div class="result error">
                <h3>Error</h3>
                <p>{str(e)}</p>
                <a href="/predict_form">Try Again</a>
            </div>
            '''

if __name__ == '__main__':
    print("=" * 60)
    print("MEDICATION DEMAND PREDICTION API")
    print("=" * 60)
    print("Model: Linear Regression")
    print("Port: 5000")
    print("Host: 0.0.0.0 (accessible from any IP)")
    print("=" * 60)
    

    print("\n1. Loading trained model...")
    model_loaded = demand_predictor.load_model()
    
    if model_loaded:
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Warning: Model not loaded. You'll need to train the model first.")
        print("   Run: python ../advanced_demand_prediction.py")
    
    print("\n2. Starting Flask server...")
    print("\nAPI Endpoints:")
    print("- Health Check:    GET  http://localhost:5000/health")
    print("- Single Predict:  POST http://localhost:5000/predict")
    print("- Batch Predict:   POST http://localhost:5000/predict_batch")
    print("- Web Form:        GET  http://localhost:5000/predict_form")
    
    print("\n3. Testing Methods:")
    print("a) Web Browser: http://localhost:5000/predict_form")
    print("b) API Testing:  python test_api.py")
    print("c) curl command: curl -X GET http://localhost:5000/health")
    
    print("\n" + "=" * 60)
    print("Starting server... (Press Ctrl+C to stop)")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
