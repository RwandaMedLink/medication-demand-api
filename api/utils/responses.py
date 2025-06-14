"""
Response formatting utilities for the Rwanda MedLink API.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from flask import jsonify
import logging

logger = logging.getLogger(__name__)


def success_response(data: Any, message: str = "Success", status_code: int = 200) -> tuple:
    """
    Create a standardized success response.
    
    Args:
        data: The response data
        message: Success message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response, status_code)
    """
    response = {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code
    }
    
    logger.info(f"Success response: {message} (status: {status_code})")
    return jsonify(response), status_code


def error_response(message: str, status_code: int = 500, error_code: Optional[str] = None, details: Optional[Dict] = None) -> tuple:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        error_code: Optional error code for categorization
        details: Optional additional error details
        
    Returns:
        Tuple of (response, status_code)
    """
    response = {
        "success": False,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code
    }
    
    if error_code:
        response["error_code"] = error_code
    
    if details:
        response["details"] = details
    
    logger.error(f"Error response: {message} (status: {status_code}, code: {error_code})")
    return jsonify(response), status_code


def validation_error_response(errors: List[str], status_code: int = 400) -> tuple:
    """
    Create a standardized validation error response.
    
    Args:
        errors: List of validation error messages
        status_code: HTTP status code
        
    Returns:
        Tuple of (response, status_code)
    """
    response = {
        "success": False,
        "message": "Validation failed",
        "errors": errors,
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": status_code,
        "error_code": "VALIDATION_ERROR"
    }
    
    logger.warning(f"Validation error response: {len(errors)} errors")
    return jsonify(response), status_code


def prediction_response(prediction: float, confidence: Optional[float] = None, 
                       features_used: Optional[List[str]] = None, 
                       model_info: Optional[Dict] = None) -> tuple:
    """
    Create a standardized prediction response.
    
    Args:
        prediction: The prediction value
        confidence: Optional confidence score
        features_used: Optional list of features used in prediction
        model_info: Optional model information
        
    Returns:
        Tuple of (response, status_code)
    """
    data = {
        "prediction": round(prediction, 4),
        "prediction_date": datetime.utcnow().isoformat()
    }
    
    if confidence is not None:
        data["confidence"] = round(confidence, 4)
    
    if features_used:
        data["features_used"] = features_used
    
    if model_info:
        data["model_info"] = model_info
    
    return success_response(data, "Prediction generated successfully")


def batch_prediction_response(predictions: List[Dict], summary: Optional[Dict] = None) -> tuple:
    """
    Create a standardized batch prediction response.
    
    Args:
        predictions: List of prediction results
        summary: Optional summary statistics
        
    Returns:
        Tuple of (response, status_code)
    """
    data = {
        "predictions": predictions,
        "batch_size": len(predictions),
        "processing_date": datetime.utcnow().isoformat()
    }
    
    if summary:
        data["summary"] = summary
    
    return success_response(data, f"Batch predictions generated successfully for {len(predictions)} items")


def model_info_response(model_info: Dict) -> tuple:
    """
    Create a standardized model information response.
    
    Args:
        model_info: Dictionary containing model information
        
    Returns:
        Tuple of (response, status_code)
    """
    data = {
        "model_info": model_info,
        "query_date": datetime.utcnow().isoformat()
    }
    
    return success_response(data, "Model information retrieved successfully")


def health_check_response(status: str = "healthy", checks: Optional[Dict] = None) -> tuple:
    """
    Create a standardized health check response.
    
    Args:
        status: Overall health status
        checks: Optional detailed health checks
        
    Returns:
        Tuple of (response, status_code)
    """
    data = {
        "status": status,
        "uptime": "Available",
        "version": "1.0.0"
    }
    
    if checks:
        data["checks"] = checks
    
    status_code = 200 if status == "healthy" else 503
    return success_response(data, f"Service is {status}", status_code)


def paginated_response(items: List[Any], page: int, per_page: int, total: int, 
                      message: str = "Data retrieved successfully") -> tuple:
    """
    Create a standardized paginated response.
    
    Args:
        items: List of items for current page
        page: Current page number
        per_page: Items per page
        total: Total number of items
        message: Response message
        
    Returns:
        Tuple of (response, status_code)
    """
    total_pages = (total + per_page - 1) // per_page
    
    data = {
        "items": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }
    
    return success_response(data, message)
