"""
Helper utilities for the Rwanda MedLink API.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import os
import json
from functools import wraps
import time

logger = logging.getLogger(__name__)


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format datetime object to string.
    
    Args:
        dt: Datetime object to format
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_str)


def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse datetime string to datetime object.
    
    Args:
        date_str: Datetime string to parse
        format_str: Format string
        
    Returns:
        Parsed datetime object
    """
    return datetime.strptime(date_str, format_str)


def calculate_model_metrics(predictions: List[float], actuals: List[float]) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        predictions: List of predicted values
        actuals: List of actual values
        
    Returns:
        Dictionary containing performance metrics
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have the same length")
    
    if len(predictions) == 0:
        return {}
    
    # Calculate metrics
    n = len(predictions)
    mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n
    rmse = mse ** 0.5
    mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
    
    # Calculate R²
    actual_mean = sum(actuals) / n
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
    ss_tot = sum((a - actual_mean) ** 2 for a in actuals)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = sum(abs((a - p) / a) for a, p in zip(actuals, predictions) if a != 0) / n * 100
    
    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 2)
    }


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON file saved successfully: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        raise


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to ensure exists
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory ensured: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        raise


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(file_path)
    except OSError as e:
        logger.error(f"Error getting file size for {file_path}: {e}")
        return 0


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    return wrapper


def retry_decorator(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function execution on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        return wrapper
    return decorator


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if valid, False otherwise
    """
    if not file_path:
        return False
    
    if must_exist:
        return os.path.isfile(file_path)
    else:
        # Check if parent directory exists or can be created
        parent_dir = os.path.dirname(file_path)
        return os.path.isdir(parent_dir) or not parent_dir


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters for Windows and Unix
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing periods and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'untitled'
    
    return filename


def calculate_prediction_confidence(prediction: float, model_metrics: Dict[str, float]) -> float:
    """
    Calculate confidence score for a prediction based on model metrics.
    
    Args:
        prediction: Predicted value
        model_metrics: Model performance metrics
        
    Returns:
        Confidence score between 0 and 1
    """
    # Base confidence from R² score
    r2_score = model_metrics.get('r2', 0)
    base_confidence = max(0, min(1, r2_score))
    
    # Adjust based on MAPE (lower MAPE = higher confidence)
    mape = model_metrics.get('mape', 100)
    mape_factor = max(0, 1 - (mape / 100))
    
    # Combined confidence
    confidence = (base_confidence + mape_factor) / 2
    
    return round(confidence, 4)


def generate_batch_summary(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for batch predictions.
    
    Args:
        predictions: List of prediction results
        
    Returns:
        Summary statistics
    """
    if not predictions:
        return {}
    
    values = [p.get('prediction', 0) for p in predictions]
    confidences = [p.get('confidence', 0) for p in predictions if 'confidence' in p]
    
    summary = {
        "total_predictions": len(predictions),
        "min_prediction": round(min(values), 4) if values else 0,
        "max_prediction": round(max(values), 4) if values else 0,
        "avg_prediction": round(sum(values) / len(values), 4) if values else 0,
        "median_prediction": round(sorted(values)[len(values) // 2], 4) if values else 0
    }
    
    if confidences:
        summary.update({
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "max_confidence": round(max(confidences), 4)
        })
    
    return summary
