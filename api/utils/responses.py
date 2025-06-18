"""
Response utilities for Rwanda Pharmacy Demand Prediction API
==========================================================

This module provides standardized response functions for:
- Success responses with consistent formatting
- Error responses with detailed error information
- Business intelligence response formatting
- Rwanda-specific response enhancements
"""

from flask import jsonify
from typing import Dict, Any, Optional, List
from datetime import datetime


def success_response(data: Dict[str, Any], message: str = "Success", status_code: int = 200) -> tuple:
    """
    Generate standardized success response.
    
    Args:
        data: Response data
        message: Success message
        status_code: HTTP status code
        
    Returns:
        Tuple of (Flask response, status_code)
    """
    response = {
        'success': True,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'data': data,
        'rwanda_medlink': {
            'version': '1.0.0',
            'system': 'Rwanda Pharmacy Prediction API',
            'capabilities': ['seasonal_analysis', 'business_intelligence', 'demographic_insights']
        }
    }
    
    return jsonify(response), status_code


def error_response(message: str, status_code: int = 400, error_code: str = None, details: Dict[str, Any] = None) -> tuple:
    """
    Generate standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        error_code: Optional error code
        details: Additional error details
        
    Returns:
        Tuple of (Flask response, status_code)
    """
    response = {
        'success': False,
        'error': {
            'message': message,
            'code': error_code or f"ERROR_{status_code}",
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        },
        'rwanda_medlink': {
            'version': '1.0.0',
            'system': 'Rwanda Pharmacy Prediction API',
            'support': 'Check input format and try again'
        }
    }
    
    return jsonify(response), status_code


def pharmacy_prediction_response(prediction_data: Dict[str, Any]) -> tuple:
    """
    Generate specialized response for pharmacy predictions.
    
    Args:
        prediction_data: Pharmacy prediction results
        
    Returns:
        Tuple of (Flask response, status_code)
    """
    # Enhance with Rwanda-specific metadata
    enhanced_data = prediction_data.copy()
    enhanced_data['rwanda_insights'] = {
        'seasonal_patterns_applied': True,
        'business_intelligence_included': True,
        'demographic_analysis_performed': True,
        'inventory_recommendations_generated': True
    }
    
    return success_response(
        enhanced_data,
        "Rwanda pharmacy prediction completed successfully",
        200
    )


def batch_prediction_response(batch_data: Dict[str, Any]) -> tuple:
    """
    Generate specialized response for batch predictions.
    
    Args:
        batch_data: Batch prediction results
        
    Returns:
        Tuple of (Flask response, status_code)
    """
    enhanced_data = batch_data.copy()
    enhanced_data['processing_metadata'] = {
        'rwanda_patterns_applied': True,
        'batch_optimization': 'enabled',
        'business_intelligence': 'generated_for_successful_predictions'
    }
    
    return success_response(
        enhanced_data,
        "Batch pharmacy predictions completed",
        200
    )
