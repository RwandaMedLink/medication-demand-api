"""
API Routes module for Rwanda Pharmacy Demand Prediction
======================================================

This module contains all the Flask route definitions for the medication demand prediction API.
Routes are organized by functionality:
- Health and status endpoints
- Model management endpoints  
- Pharmacy prediction endpoints with Rwanda-specific patterns
- Web interface endpoints with business intelligence
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any
import logging

from services import ModelService
from utils.validators import (
    validate_prediction_input, 
    validate_batch_prediction_input,
    validate_pharmacy_request,
    validate_pharmacy_batch_request
)
from utils.responses import error_response, success_response

logger = logging.getLogger(__name__)

# Create blueprints for different route groups
health_bp = Blueprint('health', __name__)
model_bp = Blueprint('model', __name__)
prediction_bp = Blueprint('prediction', __name__)
pharmacy_bp = Blueprint('pharmacy', __name__)  # New pharmacy-specific routes
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
    
    return [health_bp, model_bp, prediction_bp, pharmacy_bp, web_bp]


# ============================================================================
# Health and Status Routes
# ============================================================================

@health_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Rwanda pharmacy prediction system."""
    try:
        is_healthy = model_service.is_healthy() if model_service else False
        
        return jsonify({
            'status': 'healthy' if is_healthy else 'unhealthy',
            'model_loaded': is_healthy,
            'model_type': 'Rwanda Pharmacy Prediction Model',
            'version': '1.0.0',
            'rwanda_patterns': 'enabled',
            'seasonal_analysis': 'available'
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return error_response("Health check failed", 500)


@health_bp.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint with Rwanda pharmacy capabilities."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        return jsonify({
            'status': 'operational',
            'model_loaded': model_service.is_healthy(),
            'model_type': 'Rwanda Pharmacy Prediction Model',
            'features_available': len(model_service.core_pharmacy_features),
            'encoders_loaded': len(model_service.label_encoders) > 0,
            'rwanda_seasons': list(model_service.rwanda_seasons.keys()),
            'drug_categories': list(model_service.drug_category_mapping.values()),
            'business_intelligence': 'enabled',
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
# Rwanda Pharmacy Prediction Routes
# ============================================================================

@pharmacy_bp.route('/predict', methods=['POST'])
def pharmacy_demand_prediction():
    """
    Comprehensive pharmacy demand prediction with Rwanda-specific business intelligence.
    
    Expected request format:
    {
        "Pharmacy_Name": "CityMeds 795",
        "Province": "Kigali", 
        "Drug_ID": "DICLOFENAC",
        "ATC_Code": "M01AB",
        "s-Date": "2024-01-01",
        "E-Date": "2024-01-30",
        "available_stock": 470,
        "expiration_date": "2024-12-31",
        "stock_entry_timestamp": "2023-12-01",
        "Price_Per_Unit": 33.04,
        "Promotion": 1,
        "Season": "Urugaryi",
        "Effectiveness_Rating": 5,
        "Population_Density": "high",
        "Income_Level": "medium"
    }
    """
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        input_data = request.json
        if not input_data:
            return error_response("No input data provided", 400)
        
        # Validate pharmacy request format
        try:
            validation_result = validate_pharmacy_request(input_data)
            if not validation_result['valid']:
                return error_response(f"Validation failed: {validation_result['message']}", 400)
        except Exception as validation_error:
            return error_response(f"Input validation error: {str(validation_error)}", 400)
        
        # Generate comprehensive pharmacy prediction
        result = model_service.predict_pharmacy_demand(input_data)
        
        if not result.get('success', False):
            return error_response(
                result.get('error', 'Prediction failed'), 
                500,
                details=result
            )
        
        return success_response({
            'prediction_type': 'Rwanda Pharmacy Demand Analysis',
            'pharmacy_info': result['pharmacy_info'],
            'prediction_period': result['prediction_period'],
            'daily_predictions': result['daily_predictions'],
            'summary': result['summary'],
            'business_intelligence': result['business_intelligence'],
            'model_metadata': result['model_metadata'],
            'rwanda_insights': {
                'seasonal_patterns_applied': True,
                'drug_category_analysis': True,
                'demographic_factors_considered': True,
                'business_recommendations_included': True
            }
        })
        
    except Exception as e:
        logger.error(f"Pharmacy prediction error: {e}")
        return error_response(f"Pharmacy prediction failed: {str(e)}", 500)


@pharmacy_bp.route('/predict/batch', methods=['POST'])
def pharmacy_batch_prediction():
    """
    Batch pharmacy demand prediction for multiple drugs/pharmacies.
    
    Expected request format:
    {
        "predictions": [
            {
                "Pharmacy_Name": "CityMeds 795",
                "Province": "Kigali",
                "Drug_ID": "DICLOFENAC", 
                "ATC_Code": "M01AB",
                "s-Date": "2024-01-01",
                "E-Date": "2024-01-30",
                ...
            },
            {
                "Pharmacy_Name": "HealthPlus 123",
                "Province": "Northern",
                "Drug_ID": "IBUPROFEN",
                "ATC_Code": "M01AE", 
                "s-Date": "2024-01-01",
                "E-Date": "2024-01-30",
                ...
            }
        ]
    }
    """
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        input_data = request.json
        if not input_data:
            return error_response("No input data provided", 400)
        
        # Extract prediction records
        records = input_data.get('predictions', [])
        if not records:
            return error_response("No prediction records found", 400)
        
        # Validate batch request
        try:
            validation_result = validate_pharmacy_batch_request(records)
            if not validation_result['valid']:
                return error_response(f"Batch validation failed: {validation_result['message']}", 400)
        except Exception as validation_error:
            return error_response(f"Batch validation error: {str(validation_error)}", 400)
        
        # Process each pharmacy prediction
        batch_results = []
        successful_predictions = 0
        failed_predictions = 0
        
        for i, record in enumerate(records):
            try:
                # Validate individual record
                validation_result = validate_pharmacy_request(record)
                if not validation_result['valid']:
                    batch_results.append({
                        'record_index': i,
                        'success': False,
                        'error': validation_result['message'],
                        'pharmacy_info': record.get('Pharmacy_Name', 'Unknown')
                    })
                    failed_predictions += 1
                    continue
                
                # Generate prediction
                result = model_service.predict_pharmacy_demand(record)
                
                if result.get('success', False):
                    batch_results.append({
                        'record_index': i,
                        'success': True,
                        'prediction': result,
                        'pharmacy_info': result['pharmacy_info']
                    })
                    successful_predictions += 1
                else:
                    batch_results.append({
                        'record_index': i,
                        'success': False,
                        'error': result.get('error', 'Unknown error'),
                        'pharmacy_info': record.get('Pharmacy_Name', 'Unknown')
                    })
                    failed_predictions += 1
                    
            except Exception as record_error:
                batch_results.append({
                    'record_index': i,
                    'success': False,
                    'error': str(record_error),
                    'pharmacy_info': record.get('Pharmacy_Name', 'Unknown')
                })
                failed_predictions += 1
        
        # Generate batch summary
        batch_summary = {
            'total_records': len(records),
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'success_rate': f"{(successful_predictions/len(records)*100):.1f}%" if records else "0%",
            'processing_date': request.json.get('processing_date', 'Not specified')
        }
        
        return success_response({
            'prediction_type': 'Rwanda Pharmacy Batch Analysis',
            'batch_summary': batch_summary,
            'predictions': batch_results,
            'model_metadata': {
                'model_type': 'Rwanda Pharmacy Prediction Model',
                'rwanda_patterns': 'Applied to all predictions',
                'business_intelligence': 'Generated for successful predictions'
            }
        })
        
    except Exception as e:
        logger.error(f"Pharmacy batch prediction error: {e}")
        return error_response(f"Batch prediction failed: {str(e)}", 500)


@pharmacy_bp.route('/analytics/seasonal', methods=['GET'])
def seasonal_analytics():
    """Get Rwanda seasonal demand patterns and analytics."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        return success_response({
            'rwanda_seasons': {
                season: {
                    'numeric_code': code,
                    'period': model_service._get_season_period(season),
                    'characteristics': _get_season_characteristics(season)
                }
                for season, code in model_service.rwanda_seasons.items()
            },
            'drug_seasonal_multipliers': model_service.seasonal_multipliers,
            'business_insights': {
                'peak_malaria_season': 'Umuhindo (Sep-Nov) - Stock paracetamol group',
                'peak_respiratory_season': 'Umuhindo (Sep-Nov) - Stock respiratory drugs',
                'holiday_stress_period': 'Urugaryi (Dec-Feb) - Mental health medications',
                'inventory_planning': 'Adjust stock 2-4 weeks before seasonal peaks'
            },
            'seasonal_stocking_guide': _generate_seasonal_stocking_guide()
        })
        
    except Exception as e:
        logger.error(f"Seasonal analytics error: {e}")
        return error_response(f"Seasonal analytics failed: {str(e)}", 500)


@pharmacy_bp.route('/analytics/market', methods=['POST'])
def market_analytics():
    """
    Market analysis for specific pharmacy demographics.
    
    Expected request format:
    {
        "Province": "Kigali",
        "Population_Density": "high", 
        "Income_Level": "medium",
        "Drug_Categories": ["Anti_Inflammatory", "Paracetamol_Group"]
    }
    """
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        input_data = request.json
        if not input_data:
            return error_response("No input data provided", 400)
        
        province = input_data.get('Province', 'Unknown')
        population_density = input_data.get('Population_Density', 'medium')
        income_level = input_data.get('Income_Level', 'medium')
        drug_categories = input_data.get('Drug_Categories', [])
        
        # Generate market analysis
        market_analysis = {
            'market_profile': {
                'province': province,
                'population_density': population_density,
                'income_level': income_level,
                'market_attractiveness': _calculate_market_attractiveness(population_density, income_level),
                'target_demographics': _get_target_demographics(population_density, income_level)
            },
            'demand_patterns': _analyze_demand_patterns(drug_categories, population_density, income_level),
            'pricing_recommendations': _generate_pricing_recommendations(income_level, population_density),
            'promotional_strategy': _suggest_promotional_strategy(population_density, income_level),
            'competitive_analysis': _provide_competitive_insights(province, population_density),
            'business_opportunities': _identify_business_opportunities(province, population_density, income_level)
        }
        
        return success_response({
            'analysis_type': 'Rwanda Pharmacy Market Analysis',
            'market_analysis': market_analysis,
            'recommendations': _generate_market_recommendations(market_analysis)
        })
        
    except Exception as e:
        logger.error(f"Market analytics error: {e}")
        return error_response(f"Market analytics failed: {str(e)}", 500)


# ============================================================================
# Legacy Prediction Routes (Backward Compatibility)
# ============================================================================

@prediction_bp.route('/', methods=['POST'])
@prediction_bp.route('/predict', methods=['POST'])
@prediction_bp.route('/single', methods=['POST'])
def predict_single():
    """Legacy single prediction endpoint with enhanced pharmacy features."""
    try:
        if not model_service:
            return error_response("Model service not initialized", 500)
        
        input_data = request.json
        if not input_data:
            return error_response("No input data provided", 400)
        
        # Check if this is a pharmacy request format
        if 's-Date' in input_data and 'E-Date' in input_data:
            # Redirect to pharmacy prediction
            return pharmacy_demand_prediction()
        
        # Validate legacy input
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
            'model_type': 'Rwanda Pharmacy Prediction Model',
            'input_data': input_data,
            'recommendation': 'Use /api/pharmacy/predict for comprehensive business intelligence'
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
# Helper Functions for Rwanda Pharmacy Analytics
# ============================================================================

def _get_season_characteristics(season):
    """Get characteristics of Rwanda season for business intelligence."""
    characteristics = {
        'Urugaryi': {
            'weather': 'Short dry season (Dec-Feb)',
            'health_risks': 'Holiday stress, cold triggers',
            'business_impact': 'Mental health medications increase by 20%',
            'inventory_recommendations': 'Stock anxiolytics and sleep medications'
        },
        'Itumba': {
            'weather': 'Long rainy season (Mar-May)',
            'health_risks': 'Moderate malaria transmission, humidity-related issues',
            'business_impact': 'Paracetamol and respiratory drugs increase by 30-50%',
            'inventory_recommendations': 'Prepare for moderate malaria season'
        },
        'Icyi': {
            'weather': 'Long dry season (Jun-Aug)',
            'health_risks': 'Lowest disease transmission period',
            'business_impact': 'General maintenance period with 20% decreased demand',
            'inventory_recommendations': 'Optimize inventory turnover, reduce stock levels'
        },
        'Umuhindo': {
            'weather': 'Short rainy season (Sep-Nov)',
            'health_risks': 'PEAK malaria and respiratory disease season',
            'business_impact': 'Highest demand period - up to 60% increase',
            'inventory_recommendations': 'Maximum stock for malaria and respiratory medications'
        }
    }
    return characteristics.get(season, {
        'weather': 'Unknown season',
        'health_risks': 'No specific patterns identified',
        'business_impact': 'Standard demand patterns',
        'inventory_recommendations': 'Monitor seasonal trends'
    })


def _generate_seasonal_stocking_guide():
    """Generate comprehensive seasonal stocking recommendations for Rwanda pharmacies."""
    return {
        'quarterly_planning': {
            'Q1_Jan_Mar': {
                'focus': 'Holiday recovery and Itumba preparation',
                'key_categories': ['Mental health medications', 'Paracetamol group preparation'],
                'stock_adjustment': 'Reduce holiday stress meds, prepare for malaria season',
                'timing': 'Start Itumba prep by mid-February'
            },
            'Q2_Apr_Jun': {
                'focus': 'Itumba peak and Icyi transition',
                'key_categories': ['Malaria-related medications', 'Respiratory drugs'],
                'stock_adjustment': 'Peak demand management, then gradual reduction',
                'timing': 'Peak stock by April, start reducing by June'
            },
            'Q3_Jul_Sep': {
                'focus': 'Icyi optimization and Umuhindo preparation',
                'key_categories': ['Inventory optimization', 'Respiratory drug preparation'],
                'stock_adjustment': 'Optimize turnover, then massive Umuhindo preparation',
                'timing': 'Critical preparation period - August is key'
            },
            'Q4_Oct_Dec': {
                'focus': 'Umuhindo peak and Urugaryi transition',
                'key_categories': ['Peak respiratory and malaria', 'Holiday stress preparation'],
                'stock_adjustment': 'Maximum seasonal demand management',
                'timing': 'Peak stock October-November, holiday prep December'
            }
        },
        'critical_preparation_windows': {
            'pre_umuhindo': 'August 15 - September 1: Stock 150% normal respiratory/malaria inventory',
            'pre_urugaryi': 'November 15 - December 1: Prepare mental health medications',
            'pre_itumba': 'February 15 - March 1: Moderate malaria medication increase',
            'icyi_optimization': 'June 1 - August 15: Focus on inventory turnover and cash flow'
        },
        'emergency_protocols': {
            'outbreak_response': 'Maintain 30-day emergency stock of essential medications',
            'supply_chain_disruption': 'Identify alternative suppliers for critical drugs',
            'seasonal_shortage_mitigation': 'Pre-order 60 days before peak seasons'
        }
    }


def _calculate_market_attractiveness(population_density, income_level):
    """Calculate market attractiveness score for pharmacy business intelligence."""
    density_scores = {'low': 1, 'medium': 2, 'high': 3}
    income_scores = {'low': 1, 'medium': 2, 'higher': 3, 'high': 4}
    
    density_score = density_scores.get(population_density, 2)
    income_score = income_scores.get(income_level, 2)
    
    total_score = density_score + income_score
    
    if total_score >= 6:
        return 'Very High'
    elif total_score >= 5:
        return 'High'
    elif total_score >= 4:
        return 'Medium'
    else:
        return 'Low'


def _get_target_demographics(population_density, income_level):
    """Get target demographic insights for pharmacy market positioning."""
    demographic_profiles = {
        ('high', 'high'): {
            'profile': 'Premium urban market',
            'characteristics': 'High purchasing power, quality-focused, convenience-oriented',
            'strategy': 'Premium brands, extended hours, consultancy services'
        },
        ('high', 'medium'): {
            'profile': 'Urban middle class',
            'characteristics': 'Price-conscious but quality-aware, family-focused',
            'strategy': 'Balance of quality and affordability, family health packages'
        },
        ('medium', 'medium'): {
            'profile': 'Suburban/town market',
            'characteristics': 'Community-oriented, relationship-focused, price-sensitive',
            'strategy': 'Community engagement, loyalty programs, competitive pricing'
        },
        ('low', 'low'): {
            'profile': 'Rural/low-income market',
            'characteristics': 'Highly price-sensitive, basic healthcare needs',
            'strategy': 'Generic medications, basic services, payment flexibility'
        }
    }
    
    key = (population_density, income_level)
    return demographic_profiles.get(key, {
        'profile': 'Mixed demographic market',
        'characteristics': 'Diverse customer base with varying needs',
        'strategy': 'Flexible approach with multiple pricing tiers'
    })


def _analyze_demand_patterns(drug_categories, population_density, income_level):
    """Analyze demand patterns for specific drug categories and demographics."""
    patterns = {}
    
    for category in drug_categories:
        if category == 'Paracetamol_Group':
            patterns[category] = {
                'seasonal_variation': 'High (40% peak in Umuhindo)',
                'demographic_impact': 'Universal demand across all demographics',
                'price_sensitivity': 'High in low-income areas',
                'business_opportunity': 'Stock generic and branded options'
            }
        elif category == 'Respiratory_Drugs':
            patterns[category] = {
                'seasonal_variation': 'Very High (60% peak in Umuhindo)',
                'demographic_impact': 'Higher demand in high-density areas',
                'price_sensitivity': 'Medium - health necessity',
                'business_opportunity': 'Premium margins during peak season'
            }
        elif category == 'Anti_Inflammatory':
            patterns[category] = {
                'seasonal_variation': 'Moderate (20% variation)',
                'demographic_impact': 'Higher in active/working populations',
                'price_sensitivity': 'Medium to high',
                'business_opportunity': 'Consistent year-round demand'
            }
    
    return patterns


def _generate_pricing_recommendations(income_level, population_density):
    """Generate pricing strategy recommendations based on market demographics."""
    recommendations = {
        'strategy': '',
        'price_positioning': '',
        'margin_optimization': '',
        'competitive_approach': ''
    }
    
    if income_level in ['high', 'higher'] and population_density == 'high':
        recommendations.update({
            'strategy': 'Premium pricing with value-added services',
            'price_positioning': '10-20% above market average',
            'margin_optimization': 'Focus on branded medications and consultancy',
            'competitive_approach': 'Compete on quality and service, not price'
        })
    elif income_level == 'medium':
        recommendations.update({
            'strategy': 'Competitive pricing with selective premium options',
            'price_positioning': 'Market average with premium tier available',
            'margin_optimization': 'Balance volume and margin',
            'competitive_approach': 'Price matching with superior service'
        })
    else:  # Low income
        recommendations.update({
            'strategy': 'Value pricing with generic focus',
            'price_positioning': '5-10% below market average',
            'margin_optimization': 'High volume, lower margin model',
            'competitive_approach': 'Compete aggressively on price'
        })
    
    return recommendations


def _suggest_promotional_strategy(population_density, income_level):
    """Suggest promotional strategies based on market characteristics."""
    strategies = []
    
    if population_density == 'high':
        strategies.extend([
            'Digital marketing campaigns targeting urban professionals',
            'Social media presence for health education',
            'Partnership with local healthcare providers'
        ])
    
    if income_level in ['low', 'medium']:
        strategies.extend([
            'Loyalty card programs with accumulating discounts',
            'Bulk purchase discounts for families',
            'Generic medication promotions'
        ])
    
    if income_level in ['higher', 'high']:
        strategies.extend([
            'Premium health consultation services',
            'VIP customer programs',
            'Exclusive brand partnerships'
        ])
    
    return {
        'recommended_strategies': strategies,
        'seasonal_timing': 'Increase promotional activity 2-3 weeks before peak seasons',
        'budget_allocation': '15-20% of revenue for medium-income areas, 10% for high-income',
        'success_metrics': 'Track customer retention, average transaction value, seasonal stock turnover'
    }


def _provide_competitive_insights(province, population_density):
    """Provide competitive analysis insights for pharmacy market positioning."""
    insights = {
        'market_saturation': '',
        'competitive_landscape': '',
        'differentiation_opportunities': [],
        'threat_assessment': ''
    }
    
    if province == 'Kigali':
        insights.update({
            'market_saturation': 'High - established pharmacy chains present',
            'competitive_landscape': 'Mix of international chains and local pharmacies',
            'differentiation_opportunities': [
                'Specialized services (diabetes care, elderly care)',
                'Extended operating hours',
                'Home delivery services',
                'Health screening services'
            ],
            'threat_assessment': 'High competition but good market size'
        })
    else:
        insights.update({
            'market_saturation': 'Medium to Low - opportunities for market leadership',
            'competitive_landscape': 'Primarily local pharmacies and small chains',
            'differentiation_opportunities': [
                'Modern pharmacy services',
                'Reliable stock availability',
                'Professional consultation',
                'Community health programs'
            ],
            'threat_assessment': 'Lower competition, focus on service quality'
        })
    
    return insights


def _identify_business_opportunities(province, population_density, income_level):
    """Identify specific business opportunities for pharmacy expansion."""
    opportunities = []
    
    # Location-based opportunities
    if province == 'Kigali' and population_density == 'high':
        opportunities.extend([
            'Corporate health programs for office buildings',
            'Partnership with private health insurance',
            'Specialty pharmacy services (oncology, diabetes)'
        ])
    
    # Income-based opportunities
    if income_level in ['higher', 'high']:
        opportunities.extend([
            'Premium health and wellness products',
            'Cosmetic and beauty products',
            'Health technology devices (BP monitors, glucometers)'
        ])
    elif income_level in ['low', 'medium']:
        opportunities.extend([
            'Generic medication focus',
            'Community health education programs',
            'Payment plan services for expensive medications'
        ])
    
    # Rwanda-specific opportunities
    opportunities.extend([
        'Seasonal malaria prevention campaigns',
        'Respiratory health awareness during peak seasons',
        'Mental health support during holiday seasons',
        'Partnership with community health workers'
    ])
    
    return {
        'immediate_opportunities': opportunities[:3],
        'medium_term_opportunities': opportunities[3:6] if len(opportunities) > 3 else [],
        'long_term_opportunities': opportunities[6:] if len(opportunities) > 6 else [],
        'investment_priority': 'Focus on seasonal optimization first, then demographic-specific services'
    }


def _generate_market_recommendations(market_analysis):
    """Generate comprehensive market recommendations based on analysis."""
    recommendations = {
        'priority_actions': [],
        'investment_focus': '',
        'risk_mitigation': [],
        'growth_strategy': ''
    }
    
    market_attractiveness = market_analysis['market_profile']['market_attractiveness']
    
    if market_attractiveness in ['High', 'Very High']:
        recommendations.update({
            'priority_actions': [
                'Invest in premium services and extended product range',
                'Focus on customer retention and loyalty programs',
                'Explore expansion opportunities in similar demographics'
            ],
            'investment_focus': 'Service quality and premium product offerings',
            'growth_strategy': 'Market share expansion through differentiation'
        })
    else:
        recommendations.update({
            'priority_actions': [
                'Optimize operational efficiency and cost management',
                'Focus on essential medications and basic services',
                'Build strong community relationships'
            ],
            'investment_focus': 'Operational efficiency and community engagement',
            'growth_strategy': 'Market penetration through competitive pricing'
        })
    
    # Universal risk mitigation strategies
    recommendations['risk_mitigation'] = [
        'Maintain diverse supplier relationships',
        'Implement robust inventory management for seasonal variations',
        'Develop emergency response protocols for health outbreaks',
        'Monitor regulatory changes in Rwanda healthcare policies'
    ]
    
    return recommendations


def _handle_pharmacy_form_submission():
    """Handle pharmacy prediction form submission from web interface."""
    try:
        if not model_service or not model_service.is_healthy():
            return _render_error_page("Model not loaded - please contact administrator")
        
        # Convert form data to appropriate types for pharmacy prediction
        form_data = dict(request.form)
        
        # Convert numeric fields
        numeric_fields = [
            'available_stock', 'Price_Per_Unit', 'Promotion', 
            'Effectiveness_Rating'
        ]
        
        for field in numeric_fields:
            if field in form_data and form_data[field]:
                try:
                    if field in ['available_stock', 'Promotion', 'Effectiveness_Rating']:
                        form_data[field] = int(form_data[field])
                    else:
                        form_data[field] = float(form_data[field])
                except ValueError:
                    form_data[field] = 0
        
        # Use pharmacy prediction service
        result = model_service.predict_pharmacy_demand(form_data)
        
        if result.get('success', False):
            return _render_pharmacy_result_page(result, form_data)
        else:
            return _render_error_page(f"Prediction failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Form submission error: {e}")
        return _render_error_page(f"Form processing failed: {str(e)}")


def _render_pharmacy_result_page(result, input_data):
    """Render comprehensive pharmacy prediction results page."""
    pharmacy_info = result.get('pharmacy_info', {})
    summary = result.get('summary', {})
    business_intelligence = result.get('business_intelligence', {})
    
    # Extract key metrics
    total_demand = summary.get('total_predicted_demand', 0)
    avg_daily_demand = summary.get('average_daily_demand', 0)
    stock_days = summary.get('stock_sufficiency_days', 0)
    restock_rec = summary.get('restock_recommendation', {})
    
    # Extract business insights
    seasonal_insights = business_intelligence.get('seasonal_insights', {})
    pricing_analysis = business_intelligence.get('pricing_analysis', {})
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rwanda Pharmacy Prediction Results - MedLink</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; color: #333;
            }}
            .container {{ 
                max-width: 1200px; margin: 0 auto; background: white; padding: 30px; 
                border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
            }}
            .header {{ 
                text-align: center; margin-bottom: 30px; padding: 25px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 12px; 
            }}
            .results-grid {{ 
                display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; 
            }}
            .result-card {{ 
                background: #f8fafc; padding: 20px; border-radius: 10px; 
                border-left: 4px solid #667eea; 
            }}
            .metric {{ 
                font-size: 32px; font-weight: bold; color: #667eea; margin: 10px 0; 
            }}
            .insight-section {{ 
                margin: 25px 0; padding: 20px; background: #f0fdf4; 
                border-radius: 8px; border-left: 4px solid #10b981; 
            }}
            .btn {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 12px 24px; border: none; border-radius: 6px; 
                text-decoration: none; display: inline-block; margin: 10px 5px; 
            }}
            .seasonal-highlight {{ 
                background: linear-gradient(45deg, #667eea22, #764ba222); 
                padding: 15px; border-radius: 8px; margin: 15px 0; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div style="font-size: 2em;">🇷🇼 🏥</div>
                <h1>Rwanda MedLink - Pharmacy Prediction Results</h1>
                <p><strong>{pharmacy_info.get('Pharmacy_Name', 'Unknown')}</strong> | {pharmacy_info.get('Province', 'Unknown')} Province</p>
                <p>Drug: <strong>{pharmacy_info.get('Drug_ID', 'Unknown')}</strong> ({pharmacy_info.get('Drug_Category', 'Unknown')})</p>
            </div>
            
            <div class="results-grid">
                <div class="result-card">
                    <h3>📊 Total Predicted Demand</h3>
                    <div class="metric">{total_demand:,} units</div>
                    <p>For the specified prediction period</p>
                </div>
                <div class="result-card">
                    <h3>📅 Average Daily Demand</h3>
                    <div class="metric">{avg_daily_demand:.1f} units/day</div>
                    <p>Rwanda seasonal patterns applied</p>
                </div>
                <div class="result-card">
                    <h3>📦 Stock Sufficiency</h3>
                    <div class="metric">{stock_days:.1f} days</div>
                    <p>Current stock will last this many days</p>
                </div>
                <div class="result-card">
                    <h3>🔄 Restock Status</h3>
                    <div class="metric" style="font-size: 20px;">{restock_rec.get('urgency', 'Unknown')}</div>
                    <p>{restock_rec.get('recommendation', 'Monitor stock levels')}</p>
                </div>
            </div>
            
            <div class="insight-section">
                <h3>🌍 Rwanda Seasonal Business Intelligence</h3>
                <div class="seasonal-highlight">
                    <p><strong>Drug Category:</strong> {seasonal_insights.get('drug_category', 'Unknown')}</p>
                    <p><strong>Peak Season:</strong> {seasonal_insights.get('peak_season', {}).get('season', 'Unknown')} 
                       ({seasonal_insights.get('peak_season', {}).get('period', 'Unknown')})</p>
                    <p><strong>Business Implication:</strong> {seasonal_insights.get('business_implication', 'Monitor trends')}</p>
                    <p><strong>Seasonal Variation:</strong> {seasonal_insights.get('seasonal_variation', 'Unknown')}</p>
                </div>
            </div>
            
            <div class="insight-section">
                <h3>💰 Pricing & Market Analysis</h3>
                <p><strong>Value Score:</strong> {pricing_analysis.get('value_score', 'N/A')}</p>
                <p><strong>Price Positioning:</strong> {pricing_analysis.get('price_positioning', 'Unknown')}</p>
                <p><strong>Market Recommendation:</strong> {pricing_analysis.get('pricing_recommendation', 'Monitor pricing strategy')}</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/api/web/predict" class="btn">🔄 New Prediction</a>
                <a href="/api/health" class="btn">📊 System Status</a>
                <a href="/api/pharmacy/analytics/seasonal" class="btn">🌍 Seasonal Analytics</a>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #eff6ff; border-radius: 8px; font-size: 12px; color: #1e40af;">
                <strong>Rwanda MedLink AI System</strong> - Prediction generated using Rwanda-specific seasonal patterns, 
                demographic analysis, and pharmacy business intelligence. Results include malaria seasonality, 
                respiratory disease patterns, and local market dynamics.
            </div>
        </div>
    </body>
    </html>
    '''


def _render_error_page(error_message):
    """Render an enhanced error page with Rwanda branding."""
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error - Rwanda MedLink</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
            }}
            .container {{ 
                max-width: 600px; margin: 0 auto; background: white; padding: 30px; 
                border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
            }}
            .error {{ 
                background: #fef2f2; border: 2px solid #ef4444; color: #991b1b; 
                padding: 20px; border-radius: 8px; text-align: center; 
            }}
            .btn {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 12px 24px; border: none; border-radius: 6px; 
                text-decoration: none; display: inline-block; margin-top: 20px; 
            }}
            .support-info {{ 
                margin-top: 20px; padding: 15px; background: #f0fdf4; 
                border-radius: 8px; font-size: 14px; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 3em;">🇷🇼</div>
                <h2>Rwanda MedLink</h2>
            </div>
            
            <div class="error">
                <h3>❌ Prediction Error</h3>
                <p>{error_message}</p>
                <a href="/api/web/predict" class="btn">🔄 Try Again</a>
            </div>
            
            <div class="support-info">
                <h4>💡 Troubleshooting Tips:</h4>
                <ul>
                    <li>Ensure all required fields are filled correctly</li>
                    <li>Check that dates are in YYYY-MM-DD format</li>
                    <li>Verify Rwanda season matches the date period</li>
                    <li>Ensure numeric values are within valid ranges</li>
                </ul>
                <p><strong>Support:</strong> Contact system administrator if the problem persists.</p>
            </div>
        </div>
    </body>
    </html>
    '''


# ============================================================================
# Web Interface Routes
# ============================================================================

@web_bp.route('/predict', methods=['GET', 'POST'])
def web_predict_form():
    """Enhanced web form for Rwanda pharmacy predictions."""
    if request.method == 'GET':
        return _render_pharmacy_prediction_form()
    else:  # POST request
        return _handle_pharmacy_form_submission()


def _render_pharmacy_prediction_form():
    """Render the enhanced pharmacy prediction form HTML."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rwanda MedLink - Pharmacy Demand Prediction</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh;
            }
            .container { 
                max-width: 900px; margin: 0 auto; background: white; padding: 30px; 
                border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
            }
            .header { 
                text-align: center; margin-bottom: 30px; padding: 25px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 12px; 
            }
            .rwanda-flag { font-size: 2em; margin-bottom: 10px; }
            .form-section { margin: 25px 0; padding: 20px; background: #f8fafc; border-radius: 8px; }
            .section-title { font-size: 18px; font-weight: 600; color: #374151; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 5px; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: 600; color: #374151; }
            input, select { 
                width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 6px; 
                font-size: 14px; transition: all 0.2s;
            }
            input:focus, select:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
            .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            .form-row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
            .btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 18px 40px; border: none; border-radius: 8px; 
                font-size: 16px; font-weight: 600; cursor: pointer; width: 100%; 
                transition: all 0.3s; text-transform: uppercase; letter-spacing: 1px;
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
            .required { color: #ef4444; }
            .info-box { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px; margin: 10px 0; border-radius: 4px; }
            .seasonal-info { background: #f0fdf4; border-left: 4px solid #10b981; padding: 12px; margin: 15px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="rwanda-flag">🇷🇼</div>
                <h1>Rwanda MedLink</h1>
                <p>AI-Powered Pharmacy Demand Prediction</p>
                <small>Rwanda Pharmacy Model | Seasonal Patterns | Business Intelligence</small>
            </div>
            
            <form method="POST">
                <div class="form-section">
                    <div class="section-title">🏥 Pharmacy Information</div>
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
                </div>
                
                <div class="form-section">
                    <div class="section-title">💊 Medication Information</div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Drug ID <span class="required">*</span></label>
                            <input type="text" name="Drug_ID" value="DICLOFENAC" required>
                        </div>
                        <div class="form-group">
                            <label>ATC Code</label>
                            <select name="ATC_Code">
                                <option value="M01AB" selected>M01AB (Anti-inflammatory)</option>
                                <option value="M01AE">M01AE (Propionic Acid)</option>
                                <option value="N02BA">N02BA (Salicylic Acid)</option>
                                <option value="N02BE">N02BE (Paracetamol Group)</option>
                                <option value="N05B">N05B (Anxiolytics)</option>
                                <option value="N05C">N05C (Sleep Medications)</option>
                                <option value="R03">R03 (Respiratory Drugs)</option>
                                <option value="R06">R06 (Antihistamines)</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="form-section">
                    <div class="section-title">📅 Prediction Period</div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Start Date (s-Date) <span class="required">*</span></label>
                            <input type="date" name="s-Date" value="2024-01-01" required>
                        </div>
                        <div class="form-group">
                            <label>End Date (E-Date) <span class="required">*</span></label>
                            <input type="date" name="E-Date" value="2024-01-30" required>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Rwanda Season <span class="required">*</span></label>
                            <select name="Season" required>
                                <option value="Urugaryi" selected>Urugaryi (Dec-Feb) - Short Dry</option>
                                <option value="Itumba">Itumba (Mar-May) - Long Rainy</option>
                                <option value="Icyi">Icyi (Jun-Aug) - Long Dry</option>
                                <option value="Umuhindo">Umuhindo (Sep-Nov) - Short Rainy</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Available Stock</label>
                            <input type="number" name="available_stock" value="470" min="0">
                        </div>
                    </div>
                    <div class="seasonal-info">
                        <strong>🌍 Rwanda Seasonal Patterns:</strong><br>
                        • <strong>Umuhindo:</strong> Peak malaria & respiratory season<br>
                        • <strong>Urugaryi:</strong> Holiday stress period for mental health meds<br>
                        • <strong>Itumba:</strong> Moderate malaria & respiratory issues<br>
                        • <strong>Icyi:</strong> Lowest disease transmission period
                    </div>
                </div>
                
                <div class="form-section">
                    <div class="section-title">💰 Pricing & Promotion</div>
                    <div class="form-row-3">
                        <div class="form-group">
                            <label>Price Per Unit ($)</label>
                            <input type="number" step="0.01" name="Price_Per_Unit" value="33.04" min="0">
                        </div>
                        <div class="form-group">
                            <label>Effectiveness Rating (1-10)</label>
                            <input type="number" name="Effectiveness_Rating" value="5" min="1" max="10">
                        </div>
                        <div class="form-group">
                            <label>Promotion Active</label>
                            <select name="Promotion">
                                <option value="0">No</option>
                                <option value="1" selected>Yes</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="form-section">
                    <div class="section-title">👥 Market Demographics</div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Population Density</label>
                            <select name="Population_Density">
                                <option value="low">Low Density</option>
                                <option value="medium">Medium Density</option>
                                <option value="high" selected>High Density</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Income Level</label>
                            <select name="Income_Level">
                                <option value="low">Low Income</option>
                                <option value="medium" selected>Medium Income</option>
                                <option value="higher">Higher Income</option>
                                <option value="high">High Income</option>
                            </select>
                        </div>
                    </div>
                    <div class="info-box">
                        <strong>💡 Business Intelligence:</strong> Our AI considers Rwanda's unique demographic patterns, seasonal disease cycles, and pharmacy market dynamics to provide actionable insights for inventory planning and business optimization.
                    </div>
                </div>
                
                <button type="submit" class="btn">🔮 Generate Pharmacy Prediction & Business Intelligence</button>
            </form>
        </div>
    </body>
    </html>
    '''


def _handle_pharmacy_prediction_form_submission():
    """Handle the pharmacy prediction form submission."""
    try:
        if not model_service or not model_service.is_healthy():
            return _render_error_page("Model not loaded - please contact administrator")
        
        # Convert form data to appropriate types
        form_data = dict(request.form)
        
        # Convert numeric fields
        numeric_fields = [
            'available_stock', 'Price_Per_Unit', 'Promotion', 
            'Effectiveness_Rating'
        ]
        
        for field in numeric_fields:
            if field in form_data and form_data[field]:
                try:
                    if field in ['available_stock', 'Promotion', 'Effectiveness_Rating']:
                        form_data[field] = int(form_data[field])
                    else:
                        form_data[field] = float(form_data[field])
                except ValueError:
                    form_data[field] = 0
        
        # Handle date fields - no conversion needed, keep as strings
        # s-Date and E-Date should be passed as-is for pharmacy prediction
        
        # Use pharmacy prediction service
        result = model_service.predict_pharmacy_demand(form_data)
        
        if result.get('success', False):
            return _render_pharmacy_result_page(result, form_data)
        else:
            return _render_error_page(f"Prediction failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Form submission error: {e}")
        return _render_error_page(f"Form processing failed: {str(e)}")


def _render_pharmacy_result_page(result, input_data):
    """Render comprehensive pharmacy prediction results page."""
    pharmacy_info = result.get('pharmacy_info', {})
    summary = result.get('summary', {})
    business_intelligence = result.get('business_intelligence', {})
    
    # Extract key metrics
    total_demand = summary.get('total_predicted_demand', 0)
    avg_daily_demand = summary.get('average_daily_demand', 0)
    stock_days = summary.get('stock_sufficiency_days', 0)
    restock_rec = summary.get('restock_recommendation', {})
    
    # Extract business insights
    seasonal_insights = business_intelligence.get('seasonal_insights', {})
    pricing_analysis = business_intelligence.get('pricing_analysis', {})
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rwanda Pharmacy Prediction Results - MedLink</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; color: #333;
            }}
            .container {{ 
                max-width: 1200px; margin: 0 auto; background: white; padding: 30px; 
                border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
            }}
            .header {{ 
                text-align: center; margin-bottom: 30px; padding: 25px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 12px; 
            }}
            .results-grid {{ 
                display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; 
            }}
            .result-card {{ 
                background: #f8fafc; padding: 20px; border-radius: 10px; 
                border-left: 4px solid #667eea; 
            }}
            .metric {{ 
                font-size: 32px; font-weight: bold; color: #667eea; margin: 10px 0; 
            }}
            .insight-section {{ 
                margin: 25px 0; padding: 20px; background: #f0fdf4; 
                border-radius: 8px; border-left: 4px solid #10b981; 
            }}
            .btn {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 12px 24px; border: none; border-radius: 6px; 
                text-decoration: none; display: inline-block; margin: 10px 5px; 
            }}
            .warning {{ background: #fef3cd; border-left: 4px solid #f59e0b; }}
            .success {{ background: #d1fae5; border-left: 4px solid #10b981; }}
            .seasonal-highlight {{ 
                background: linear-gradient(45deg, #667eea22, #764ba222); 
                padding: 15px; border-radius: 8px; margin: 15px 0; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div style="font-size: 2em;">🇷🇼 🏥</div>
                <h1>Rwanda MedLink - Pharmacy Prediction Results</h1>
                <p><strong>{pharmacy_info.get('Pharmacy_Name', 'Unknown')}</strong> | {pharmacy_info.get('Province', 'Unknown')} Province</p>
                <p>Drug: <strong>{pharmacy_info.get('Drug_ID', 'Unknown')}</strong> ({pharmacy_info.get('Drug_Category', 'Unknown')})</p>
            </div>
            
            <div class="results-grid">
                <div class="result-card">
                    <h3>📊 Total Predicted Demand</h3>
                    <div class="metric">{total_demand:,} units</div>
                    <p>For the specified prediction period</p>
                </div>
                <div class="result-card">
                    <h3>📅 Average Daily Demand</h3>
                    <div class="metric">{avg_daily_demand:.1f} units/day</div>
                    <p>Rwanda seasonal patterns applied</p>
                </div>
                <div class="result-card">
                    <h3>📦 Stock Sufficiency</h3>
                    <div class="metric">{stock_days:.1f} days</div>
                    <p>Current stock will last this many days</p>
                </div>
                <div class="result-card">
                    <h3>🔄 Restock Status</h3>
                    <div class="metric" style="font-size: 20px;">{restock_rec.get('urgency', 'Unknown')}</div>
                    <p>{restock_rec.get('recommendation', 'Monitor stock levels')}</p>
                </div>
            </div>
            
            <div class="insight-section">
                <h3>🌍 Rwanda Seasonal Business Intelligence</h3>
                <div class="seasonal-highlight">
                    <p><strong>Drug Category:</strong> {seasonal_insights.get('drug_category', 'Unknown')}</p>
                    <p><strong>Peak Season:</strong> {seasonal_insights.get('peak_season', {}).get('season', 'Unknown')} 
                       ({seasonal_insights.get('peak_season', {}).get('period', 'Unknown')})</p>
                    <p><strong>Business Implication:</strong> {seasonal_insights.get('business_implication', 'Monitor trends')}</p>
                    <p><strong>Seasonal Variation:</strong> {seasonal_insights.get('seasonal_variation', 'Unknown')}</p>
                </div>
            </div>
            
            <div class="insight-section">
                <h3>💰 Pricing & Market Analysis</h3>
                <p><strong>Value Score:</strong> {pricing_analysis.get('value_score', 'N/A')}</p>
                <p><strong>Price Positioning:</strong> {pricing_analysis.get('price_positioning', 'Unknown')}</p>
                <p><strong>Market Recommendation:</strong> {pricing_analysis.get('pricing_recommendation', 'Monitor pricing strategy')}</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/api/web/predict" class="btn">🔄 New Prediction</a>
                <a href="/api/health" class="btn">📊 System Status</a>
                <a href="/api/pharmacy/analytics/seasonal" class="btn">🌍 Seasonal Analytics</a>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #eff6ff; border-radius: 8px; font-size: 12px; color: #1e40af;">
                <strong>Rwanda MedLink AI System</strong> - Prediction generated using Rwanda-specific seasonal patterns, 
                demographic analysis, and pharmacy business intelligence. Results include malaria seasonality, 
                respiratory disease patterns, and local market dynamics.
            </div>
        </div>
    </body>
    </html>
    '''


def _render_error_page(error_message):
    """Render an enhanced error page with Rwanda branding."""
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Error - Rwanda MedLink</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
            }}
            .container {{ 
                max-width: 600px; margin: 0 auto; background: white; padding: 30px; 
                border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); 
            }}
            .error {{ 
                background: #fef2f2; border: 2px solid #ef4444; color: #991b1b; 
                padding: 20px; border-radius: 8px; text-align: center; 
            }}
            .btn {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 12px 24px; border: none; border-radius: 6px; 
                text-decoration: none; display: inline-block; margin-top: 20px; 
            }}
            .support-info {{ 
                margin-top: 20px; padding: 15px; background: #f0fdf4; 
                border-radius: 8px; font-size: 14px; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 3em;">🇷🇼</div>
                <h2>Rwanda MedLink</h2>
            </div>
            
            <div class="error">
                <h3>❌ Prediction Error</h3>
                <p>{error_message}</p>
                <a href="/api/web/predict" class="btn">🔄 Try Again</a>
            </div>
            
            <div class="support-info">
                <h4>💡 Troubleshooting Tips:</h4>
                <ul>
                    <li>Ensure all required fields are filled correctly</li>
                    <li>Check that dates are in YYYY-MM-DD format</li>
                    <li>Verify Rwanda season matches the date period</li>
                    <li>Ensure numeric values are within valid ranges</li>
                </ul>
                <p><strong>Support:</strong> Contact system administrator if the problem persists.</p>
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
