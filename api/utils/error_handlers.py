"""
Error handling utilities for the Rwanda MedLink API.
"""

from typing import Dict, Any
from flask import Flask, request
import logging
from werkzeug.exceptions import HTTPException
import traceback

from .responses import error_response

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception class for API errors."""
    
    def __init__(self, message: str, status_code: int = 500, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "API_ERROR"
        self.details = details or {}


class ValidationError(APIError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, 400, "VALIDATION_ERROR", details)


class ModelError(APIError):
    """Exception for model-related errors."""
    
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, 500, "MODEL_ERROR", details)


class ConfigurationError(APIError):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, 500, "CONFIGURATION_ERROR", details)


class DataError(APIError):
    """Exception for data-related errors."""
    
    def __init__(self, message: str, details: Dict = None):
        super().__init__(message, 422, "DATA_ERROR", details)


def register_error_handlers(app: Flask) -> None:
    """
    Register error handlers for the Flask application.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        """Handle custom API errors."""
        logger.error(f"API Error: {error.message} (Code: {error.error_code})")
        return error_response(
            message=error.message,
            status_code=error.status_code,
            error_code=error.error_code,
            details=error.details
        )
    
    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        """Handle validation errors."""
        logger.warning(f"Validation Error: {error.message}")
        return error_response(
            message=error.message,
            status_code=error.status_code,
            error_code=error.error_code,
            details=error.details
        )
    
    @app.errorhandler(ModelError)
    def handle_model_error(error: ModelError):
        """Handle model-related errors."""
        logger.error(f"Model Error: {error.message}")
        return error_response(
            message=error.message,
            status_code=error.status_code,
            error_code=error.error_code,
            details=error.details
        )
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        logger.warning(f"404 Error: {request.url} not found")
        return error_response(
            message="Resource not found",
            status_code=404,
            error_code="NOT_FOUND",
            details={"url": request.url, "method": request.method}
        )
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 errors."""
        logger.warning(f"405 Error: Method {request.method} not allowed for {request.url}")
        return error_response(
            message="Method not allowed",
            status_code=405,
            error_code="METHOD_NOT_ALLOWED",
            details={"url": request.url, "method": request.method}
        )
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 errors."""
        logger.warning(f"400 Error: Bad request for {request.url}")
        return error_response(
            message="Bad request",
            status_code=400,
            error_code="BAD_REQUEST",
            details={"url": request.url, "method": request.method}
        )
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors."""
        logger.error(f"500 Error: Internal server error")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return error_response(
            message="Internal server error",
            status_code=500,
            error_code="INTERNAL_ERROR",
            details={"url": request.url, "method": request.method}
        )
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle generic HTTP exceptions."""
        logger.warning(f"HTTP Exception: {error.code} - {error.description}")
        return error_response(
            message=error.description,
            status_code=error.code,
            error_code="HTTP_ERROR",
            details={"url": request.url, "method": request.method}
        )
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle unexpected errors."""
        logger.error(f"Unexpected Error: {str(error)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # In production, don't expose internal error details
        if app.config.get('ENV') == 'production':
            message = "An unexpected error occurred"
            details = {}
        else:
            message = f"Unexpected error: {str(error)}"
            details = {"traceback": traceback.format_exc()}
        
        return error_response(
            message=message,
            status_code=500,
            error_code="UNEXPECTED_ERROR",
            details=details
        )
