"""
Configuration settings for Rwanda Pharmacy Demand Prediction System
================================================================

This module contains all configuration settings for the pharmacy
prediction API, model service, and business intelligence components.
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for Rwanda pharmacy prediction system."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    REPORTS_DIR = BASE_DIR / "reports"
    FIGURES_DIR = BASE_DIR / "figures"
    
    # Model configuration
    DEFAULT_MODEL_PATH = MODEL_DIR / "pharmacy_linear_regression_label_r2_best.pkl"
    LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"
    FEATURE_MAPPINGS_PATH = MODEL_DIR / "feature_mappings.pkl"
    MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
    
    # Rwanda-specific settings
    RWANDA_SEASONS = {
        'Itumba': 1,    # Mar-May: Long rainy season
        'Icyi': 2,      # Jun-Aug: Long dry season
        'Umuhindo': 3,  # Sep-Nov: Short rainy season
        'Urugaryi': 4   # Dec-Feb: Short dry season
    }
    
    # Pharmacy business settings
    MIN_STOCK_ALERT_THRESHOLD = 50
    CRITICAL_STOCK_THRESHOLD = 20
    DEFAULT_SAFETY_STOCK_PERCENTAGE = 0.2
    MAX_PREDICTION_DAYS = 90
    
    # API settings
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.MODEL_DIR, cls.DATA_DIR, cls.LOGS_DIR, 
                         cls.REPORTS_DIR, cls.FIGURES_DIR]:
            directory.mkdir(exist_ok=True)
