"""
API Routes module
================

This module contains all the Flask route definitions for the medication demand prediction API.
Routes are organized by functionality:
- Health and status endpoints
- Model management endpoints  
- Prediction endpoints
- Web interface endpoints
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any
import logging

from services import ModelService
from utils.validators import validate_prediction_input, validate_batch_prediction_input
from utils.responses import error_response, success_response

logger = logging.getLogger(__name__)

# Create blueprints for different route groups
health_bp = Blueprint('health', __name__)
model_bp = Blueprint('model', __name__)
prediction_bp = Blueprint('prediction', __name__)
web_bp = Blueprint('web', __name__)

# Model service instance (will be injected)
model_service: ModelService = None


def create_blueprints(service: ModelService):
    """
    Create and return blueprints with the model service injected.
    
    Args:
        service: ModelService instance
        
    Returns:
        List of Flask blueprints
    """
    global model_service
    model_service = service
    
    return [health_bp, model_bp, prediction_bp, web_bp]


# ============================================================================
# Health and Status Routes
# ============================================================================

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        is_healthy = model_service.is_healthy() if model_service else False
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'model_loaded': is_healthy,
            'model_type': 'Linear Regression',
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return error_response("Health check failed", 500)


@health_bp.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        return jsonify({
            'status': 'operational',
            'model_loaded': model_service.is_healthy(),
            'model_type': 'Linear Regression',
            'features_available': len(model_service.default_features),
            'encoders_loaded': len(model_service.label_encoders) > 0,
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return error_response("Status check failed", 500)


# ============================================================================
# Model Management Routes
# ============================================================================

@model_bp.route('/load', methods=['POST'])
def load_model():
    """Load or reload the model."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        data = request.json or {}
        model_path = data.get('model_path')
        
        success = model_service.load_model(model_path)
        
        if success:
            return success_response({
                'message': 'Model loaded successfully',
                'model_type': 'Linear Regression'
            })
        else:
            return error_response("Failed to load model", 500)
            
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return error_response(f"Error loading model: {str(e)}", 500)


@model_bp.route('/info', methods=['GET'])
def model_info():
    """Get model information."""
    try:
        if not model_service or not model_service.is_healthy():
            return error_response("Model not loaded", 404)
        
        return jsonify({
            'model_type': 'Linear Regression',
            'features_count': len(model_service.feature_order or model_service.default_features),
            'features_expected': model_service.feature_order or model_service.default_features,
            'encoders_available': list(model_service.label_encoders.keys()),
            'model_loaded': True
        })
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return error_response("Failed to get model info", 500)


# ============================================================================
# Prediction Routes
# ============================================================================

@prediction_bp.route('/', methods=['POST'])
@prediction_bp.route('/predict', methods=['POST'])
@prediction_bp.route('/single', methods=['POST'])
def predict_single():
    """Predict medication demand for a single record."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        input_data = request.json
        if not input_data:
            return error_response("No input data provided", 400)
        
        # Validate input
        try:
            validated_data = validate_prediction_input(input_data)
        except Exception as validation_error:
            return error_response(str(validation_error), 400)
        
        # Make prediction (will use fallback if model not loaded)
        prediction, message = model_service.predict(validated_data)
        
        if prediction is None:
            return error_response(message, 500)
        
        return success_response({
            'predicted_demand': int(prediction),
            'message': message,
            'model_type': 'Linear Regression',
            'input_data': input_data
        })
        
    except Exception as e:
        logger.error(f"Single prediction error: {e}")
        return error_response(f"Prediction failed: {str(e)}", 500)


@prediction_bp.route('/batch', methods=['POST'])
@prediction_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict medication demand for multiple records."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        input_data = request.json
        if not input_data:
            return error_response("No input data provided", 400)
        
        # Handle different input formats
        records = None
        if 'records' in input_data:
            records = input_data['records']
        elif isinstance(input_data, list):
            records = input_data
        elif isinstance(input_data, dict) and len(input_data) > 0:
            records = [input_data]
        
        if not records:
            return error_response("No records found in input data", 400)
        
        # Validate input
        try:
            validate_batch_prediction_input(records)
        except Exception as validation_error:
            return error_response(str(validation_error), 400)
        
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
            
            # Validate individual record
            try:
                validated_record = validate_prediction_input(record)
            except Exception as validation_error:
                predictions.append({
                    'record_index': i,
                    'predicted_demand': None,
                    'message': str(validation_error),
                    'input_data': record
                })
                continue
            
            # Make prediction
            prediction, message = model_service.predict(validated_record)
            predictions.append({
                'record_index': i,
                'predicted_demand': int(prediction) if prediction is not None else None,
                'message': message,
                'input_data': record
            })
        
        # Calculate summary statistics
        successful_predictions = [p for p in predictions if p['predicted_demand'] is not None]
        failed_predictions = [p for p in predictions if p['predicted_demand'] is None]
        
        return success_response({
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
        logger.error(f"Batch prediction error: {e}")
        return error_response(f"Batch prediction failed: {str(e)}", 500)


# ============================================================================
# Web Interface Routes
# ============================================================================

@web_bp.route('/predict', methods=['GET', 'POST'])
def web_predict_form():
    """Web form for making predictions."""
    if request.method == 'GET':
        return _render_prediction_form()
    else:  # POST request
        return _handle_form_submission()


def _render_prediction_form():
    """Render the prediction form HTML."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rwanda MedLink - Medication Demand Prediction</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; background: #f8fafc; 
            }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: 600; color: #374151; }
            input, select { 
                width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 6px; 
                font-size: 14px; transition: border-color 0.2s;
            }
            input:focus, select:focus { outline: none; border-color: #667eea; }
            .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            .btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 15px 30px; border: none; border-radius: 6px; 
                font-size: 16px; font-weight: 600; cursor: pointer; width: 100%; 
                transition: transform 0.2s;
            }
            .btn:hover { transform: translateY(-2px); }
            .required { color: #ef4444; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Rwanda MedLink</h1>
                <p>AI-Powered Medication Demand Prediction</p>
                <small>Linear Regression Model | Version 1.0.0</small>
            </div>
            
            <form method="POST">
                <div class="form-row">
                    <div class="form-group">
                        <label>Pharmacy Name <span class="required">*</span></label>
                        <input type="text" name="Pharmacy_Name" value="CityMeds 795" required>
                    </div>
                    <div class="form-group">
                        <label>Province <span class="required">*</span></label>
                        <select name="Province" required>
                            <option value="Kigali" selected>Kigali</option>
                            <option value="Northern">Northern</option>
                            <option value="Southern">Southern</option>
                            <option value="Eastern">Eastern</option>
                            <option value="Western">Western</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Drug ID <span class="required">*</span></label>
                        <input type="text" name="Drug_ID" value="DICLOFENAC" required>
                    </div>
                    <div class="form-group">
                        <label>ATC Code</label>
                        <input type="text" name="ATC_Code" value="M01AB">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Date <span class="required">*</span></label>
                        <input type="date" name="Date" value="2024-01-01" required>
                    </div>
                    <div class="form-group">
                        <label>Available Stock</label>
                        <input type="number" name="available_stock" value="470" min="0">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Price Per Unit ($)</label>
                        <input type="number" step="0.01" name="Price_Per_Unit" value="33.04" min="0">
                    </div>
                    <div class="form-group">
                        <label>Effectiveness Rating (1-10)</label>
                        <input type="number" name="Effectiveness_Rating" value="5" min="1" max="10">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Promotion Active</label>
                        <select name="Promotion">
                            <option value="0">No</option>
                            <option value="1" selected>Yes</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Disease Outbreak</label>
                        <select name="Disease_Outbreak">
                            <option value="0">No</option>
                            <option value="1" selected>Yes</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Holiday Week</label>
                        <select name="Holiday_Week">
                            <option value="0">No</option>
                            <option value="1" selected>Yes</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Competitor Count</label>
                        <input type="number" name="Competitor_Count" value="4" min="0">
                    </div>
                </div>
                
                <button type="submit" class="btn">🔮 Predict Demand</button>
            </form>
        </div>
    </body>
    </html>
    '''


def _handle_form_submission():
    """Handle form submission and return results."""
    try:
        if not model_service or not model_service.is_healthy():
            return _render_error_page("Model not loaded")
        
        # Convert form data to appropriate types
        form_data = dict(request.form)
        
        # Convert numeric fields
        numeric_fields = ['available_stock', 'Price_Per_Unit', 'Promotion', 'Disease_Outbreak', 
                        'Effectiveness_Rating', 'Competitor_Count', 'Holiday_Week']
        
        for field in numeric_fields:
            if field in form_data and form_data[field]:
                try:
                    form_data[field] = float(form_data[field]) if '.' in str(form_data[field]) else int(form_data[field])
                except ValueError:
                    form_data[field] = 0
        
        # Make prediction
        prediction, message = model_service.predict(form_data)
        
        return _render_result_page(prediction, message, form_data)
        
    except Exception as e:
        logger.error(f"Form submission error: {e}")
        return _render_error_page(f"Prediction failed: {str(e)}")


def _render_result_page(prediction, message, input_data):
    """Render the prediction result page."""
    success = prediction is not None
    demand = prediction if prediction is not None else "Error"
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result - Rwanda MedLink</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
            .result {{ margin: 20px 0; padding: 25px; border-radius: 8px; text-align: center; }}
            .success {{ background: #ecfdf5; border: 2px solid #10b981; color: #065f46; }}
            .error {{ background: #fef2f2; border: 2px solid #ef4444; color: #991b1b; }}
            .demand-number {{ font-size: 48px; font-weight: bold; margin: 15px 0; }}
            .details {{ background: #f9fafb; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: left; }}
            .btn {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; border: none; border-radius: 6px; text-decoration: none; display: inline-block; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 Prediction Result</h1>
                <p>Rwanda MedLink AI Analysis</p>
            </div>
            
            <div class="result {'success' if success else 'error'}">
                <h3>{'✅ Prediction Complete' if success else '❌ Prediction Failed'}</h3>
                <div class="demand-number">{demand} {'units' if success else ''}</div>
                <p><strong>Status:</strong> {message}</p>
            </div>
            
            <div class="details">
                <h4>📋 Input Summary</h4>
                <p><strong>Pharmacy:</strong> {input_data.get('Pharmacy_Name', 'N/A')}</p>
                <p><strong>Drug:</strong> {input_data.get('Drug_ID', 'N/A')}</p>
                <p><strong>Province:</strong> {input_data.get('Province', 'N/A')}</p>
                <p><strong>Date:</strong> {input_data.get('Date', 'N/A')}</p>
                <p><strong>Stock:</strong> {input_data.get('available_stock', 'N/A')} units</p>
                <p><strong>Price:</strong> ${input_data.get('Price_Per_Unit', 'N/A')}</p>
            </div>
            
            <div style="text-align: center;">
                <a href="/web/predict" class="btn">🔄 Make Another Prediction</a>
            </div>
        </div>
    </body>
    </html>
    '''


def _render_error_page(error_message):
    """Render an error page."""
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error - Rwanda MedLink</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8fafc; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .error {{ background: #fef2f2; border: 2px solid #ef4444; color: #991b1b; padding: 20px; border-radius: 8px; text-align: center; }}
            .btn {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 24px; border: none; border-radius: 6px; text-decoration: none; display: inline-block; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="error">
                <h3>❌ Error</h3>
                <p>{error_message}</p>
                <a href="/web/predict" class="btn">🔄 Try Again</a>
            </div>
        </div>
    </body>
    </html>
    '''


# ============================================================================
# Backward Compatibility Routes (Deprecated)
# ============================================================================

@health_bp.route('/load_model', methods=['POST'])
def load_model_deprecated():
    """Deprecated endpoint - redirects to new API structure."""
    return error_response(
        message="This endpoint has been moved. Please use POST /api/model/load instead.",
        status_code=404,
        error_code="ENDPOINT_MOVED",
        details={
            "old_endpoint": "POST /load_model",
            "new_endpoint": "POST /api/model/load",
            "documentation": "See /api/web/predict for interactive documentation"
        }
    )

@health_bp.route('/model_info', methods=['GET'])
def model_info_deprecated():
    """Deprecated endpoint - redirects to new API structure."""
    return error_response(
        message="This endpoint has been moved. Please use GET /api/model/info instead.",
        status_code=404,
        error_code="ENDPOINT_MOVED", 
        details={
            "old_endpoint": "GET /model_info",
            "new_endpoint": "GET /api/model/info",
            "documentation": "See /api/web/predict for interactive documentation"
        }
    )

@health_bp.route('/predict', methods=['POST'])
def predict_deprecated():
    """Deprecated endpoint - redirects to new API structure."""
    return error_response(
        message="This endpoint has been moved. Please use POST /api/predict instead.",
        status_code=404,
        error_code="ENDPOINT_MOVED",
        details={
            "old_endpoint": "POST /predict", 
            "new_endpoint": "POST /api/predict",
            "documentation": "See /api/web/predict for interactive documentation"
        }
    )
