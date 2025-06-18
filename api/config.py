"""
Configuration module for Rwanda MedLink API
==========================================

This module contains all configuration settings for the Flask application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

class Config:
    """Base configuration class"""
    
    # Flask settings
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # API settings
    API_TITLE = "Rwanda MedLink Medication Demand Prediction API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "ML-powered medication demand forecasting for pharmaceutical inventory management"
    
    # Model settings
    MODEL_DIR = BASE_DIR / 'models'
    DEFAULT_MODEL_PATH = MODEL_DIR / 'pharmacy_linear_regression_label_r2_0.9985_20250618_1040.pkl'
    ENCODERS_PATH = MODEL_DIR / 'label_encoders.pkl'
    
    # Feature engineering settings
    DEFAULT_FEATURES = [
        'Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'available_stock',
        'Price_Per_Unit', 'Promotion', 'Season', 'Disease_Outbreak',
        'Supply_Chain_Delay', 'Effectiveness_Rating', 'Competitor_Count',
        'Time_On_Market', 'Population_Density', 'Income_Level', 'Holiday_Week',
        'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear',
        'Days_Until_Expiry', 'Days_Since_Stock_Entry', 'Inventory_Turnover',
        'Avg_Drug_Sales', 'Prev_Day_Sales', 'Avg_Pharmacy_Sales', 'Outbreak_Effectiveness',
        'Price_Position', 'Prev_Week_Sales', 'Rolling_7day_Mean', 'Avg_Drug_Price', 'Promotion_Holiday'
    ]
    
    # Validation settings
    REQUIRED_PREDICTION_FIELDS = ['Pharmacy_Name', 'Province', 'Drug_ID', 'Date']
    MAX_BATCH_SIZE = 1000
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Add production-specific settings here


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
