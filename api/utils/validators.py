"""
Input validation utilities for Rwanda Pharmacy Demand Prediction API
===================================================================

This module provides comprehensive validation functions for:
- Legacy prediction input validation
- Rwanda pharmacy-specific request validation
- Batch prediction validation
- Business rules validation
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import re

# Rwanda-specific validation constants
RWANDA_SEASONS = {'Itumba', 'Icyi', 'Umuhindo', 'Urugaryi'}
RWANDA_PROVINCES = {'Kigali', 'Northern', 'Southern', 'Eastern', 'Western'}
POPULATION_DENSITIES = {'low', 'medium', 'high'}
INCOME_LEVELS = {'low', 'medium', 'higher', 'high'}
VALID_ATC_CODES = {'M01AB', 'M01AE', 'N02BA', 'N02BE', 'N02BB', 'N05B', 'N05C', 'R03', 'R06'}

def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate legacy prediction input format.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Validated and cleaned data dictionary
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    
    # Required fields for basic prediction
    required_fields = ['Pharmacy_Name', 'Drug_ID']
    missing_fields = [field for field in required_fields if not data.get(field)]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate and set defaults
    validated_data = data.copy()
    
    # String fields
    validated_data['Pharmacy_Name'] = str(data.get('Pharmacy_Name', '')).strip()
    validated_data['Drug_ID'] = str(data.get('Drug_ID', '')).strip()
    validated_data['Province'] = str(data.get('Province', 'Kigali')).strip()
    validated_data['ATC_Code'] = str(data.get('ATC_Code', 'Unknown')).strip()
    
    # Validate province
    if validated_data['Province'] not in RWANDA_PROVINCES and validated_data['Province'] != 'Unknown':
        validated_data['Province'] = 'Kigali'  # Default to Kigali
    
    # Numeric fields with validation
    numeric_fields = {
        'available_stock': (0, 50000, 0),
        'Price_Per_Unit': (0.01, 1000, 30.0),
        'Promotion': (0, 1, 0),
        'Disease_Outbreak': (0, 1, 0),
        'Effectiveness_Rating': (1, 10, 5),
        'Competitor_Count': (0, 50, 3),
        'Holiday_Week': (0, 1, 0)
    }
    
    for field, (min_val, max_val, default) in numeric_fields.items():
        try:
            value = float(data.get(field, default))
            validated_data[field] = max(min_val, min(max_val, value))
        except (ValueError, TypeError):
            validated_data[field] = default
    
    # Date validation
    if 'Date' in data:
        try:
            date_obj = pd.to_datetime(str(data['Date']))
            validated_data['Date'] = date_obj.strftime('%Y-%m-%d')
        except:
            validated_data['Date'] = datetime.now().strftime('%Y-%m-%d')
    else:
        validated_data['Date'] = datetime.now().strftime('%Y-%m-%d')
    
    return validated_data


def validate_pharmacy_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Rwanda pharmacy-specific prediction request.
    
    Args:
        data: Pharmacy request data dictionary
        
    Returns:
        Dictionary with 'valid' boolean and 'message' string
        
    Expected format:
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
        if not isinstance(data, dict):
            return {'valid': False, 'message': 'Request must be a JSON object'}
        
        # Required fields validation
        required_fields = [
            'Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code',
            's-Date', 'E-Date', 'Season'
        ]
        
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return {
                'valid': False, 
                'message': f'Missing required fields: {missing_fields}'
            }
        
        # Validate pharmacy and location data
        pharmacy_name = str(data['Pharmacy_Name']).strip()
        if len(pharmacy_name) < 2:
            return {'valid': False, 'message': 'Pharmacy_Name must be at least 2 characters'}
        
        province = str(data['Province']).strip()
        if province not in RWANDA_PROVINCES:
            return {
                'valid': False, 
                'message': f'Province must be one of: {list(RWANDA_PROVINCES)}'
            }
        
        # Validate drug information
        drug_id = str(data['Drug_ID']).strip()
        if len(drug_id) < 2:
            return {'valid': False, 'message': 'Drug_ID must be at least 2 characters'}
        
        atc_code = str(data['ATC_Code']).strip()
        if atc_code and atc_code not in VALID_ATC_CODES and atc_code != 'Unknown':
            return {
                'valid': False,
                'message': f'ATC_Code must be one of: {list(VALID_ATC_CODES)} or empty'
            }
        
        # Validate Rwanda season
        season = str(data['Season']).strip()
        if season not in RWANDA_SEASONS:
            return {
                'valid': False,
                'message': f'Season must be one of Rwanda seasons: {list(RWANDA_SEASONS)}'
            }
        
        # Validate date range
        try:
            start_date = pd.to_datetime(str(data['s-Date']))
            end_date = pd.to_datetime(str(data['E-Date']))
            
            if start_date > end_date:
                return {'valid': False, 'message': 's-Date must be before or equal to E-Date'}
            
            # Check reasonable date range (allow up to 3 years for long-term planning)
            if (end_date - start_date).days > 1095:  # 3 years
                return {'valid': False, 'message': 'Date range cannot exceed 3 years (1095 days)'}
            
            # Check dates are not too far in the past or future (business-friendly limits)
            now = datetime.now()
            if start_date < (now - timedelta(days=1825)):  # 5 years ago
                return {'valid': False, 'message': 's-Date cannot be more than 5 years in the past'}
            
            if end_date > (now + timedelta(days=1825)):  # 5 years future
                return {'valid': False, 'message': 'E-Date cannot be more than 5 years in the future'}
                
        except Exception as e:
            return {'valid': False, 'message': f'Invalid date format. Use YYYY-MM-DD. Error: {str(e)}'}
        
        # Validate numeric fields
        numeric_validations = {
            'available_stock': (0, 100000, 'Available stock must be between 0 and 100,000'),
            'Price_Per_Unit': (0.01, 10000, 'Price per unit must be between 0.01 and 10,000'),
            'Effectiveness_Rating': (1, 10, 'Effectiveness rating must be between 1 and 10'),
            'Promotion': (0, 1, 'Promotion must be 0 or 1')
        }
        
        for field, (min_val, max_val, error_msg) in numeric_validations.items():
            if field in data:
                try:
                    value = float(data[field])
                    if not (min_val <= value <= max_val):
                        return {'valid': False, 'message': error_msg}
                except (ValueError, TypeError):
                    return {'valid': False, 'message': f'{field} must be a valid number'}
        
        # Validate categorical fields
        if 'Population_Density' in data:
            pop_density = str(data['Population_Density']).strip().lower()
            if pop_density not in POPULATION_DENSITIES:
                return {
                    'valid': False,
                    'message': f'Population_Density must be one of: {list(POPULATION_DENSITIES)}'
                }
        
        if 'Income_Level' in data:
            income_level = str(data['Income_Level']).strip().lower()
            if income_level not in INCOME_LEVELS:
                return {
                    'valid': False,
                    'message': f'Income_Level must be one of: {list(INCOME_LEVELS)}'
                }
        
        # Validate optional date fields
        date_fields = ['expiration_date', 'stock_entry_timestamp']
        for field in date_fields:
            if field in data and data[field]:
                try:
                    pd.to_datetime(str(data[field]))
                except:
                    return {'valid': False, 'message': f'{field} must be in YYYY-MM-DD format'}
        
        # Business logic validation
        if 'expiration_date' in data and data['expiration_date']:
            try:
                exp_date = pd.to_datetime(str(data['expiration_date']))
                if exp_date < start_date:
                    return {'valid': False, 'message': 'Expiration date cannot be before prediction start date'}
            except:
                pass
        
        return {'valid': True, 'message': 'Valid pharmacy request'}
        
    except Exception as e:
        return {'valid': False, 'message': f'Validation error: {str(e)}'}


def validate_batch_prediction_input(records: List[Dict[str, Any]]) -> bool:
    """
    Validate batch prediction input (legacy format).
    
    Args:
        records: List of prediction records
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(records, list):
        raise ValueError("Batch input must be a list of records")
    
    if len(records) == 0:
        raise ValueError("Batch input cannot be empty")
    
    if len(records) > 100:
        raise ValueError("Batch size cannot exceed 100 records")
    
    # Validate each record
    for i, record in enumerate(records):
        try:
            validate_prediction_input(record)
        except ValueError as e:
            raise ValueError(f"Record {i}: {str(e)}")
    
    return True


def validate_pharmacy_batch_request(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate batch pharmacy prediction request.
    
    Args:
        records: List of pharmacy prediction records
        
    Returns:
        Dictionary with 'valid' boolean and 'message' string
    """
    try:
        if not isinstance(records, list):
            return {'valid': False, 'message': 'Batch request must be a list of records'}
        
        if len(records) == 0:
            return {'valid': False, 'message': 'Batch request cannot be empty'}
        
        if len(records) > 50:  # Limit for pharmacy batch processing
            return {'valid': False, 'message': 'Batch size cannot exceed 50 pharmacy records'}
        
        # Validate each pharmacy record
        for i, record in enumerate(records):
            validation_result = validate_pharmacy_request(record)
            if not validation_result['valid']:
                return {
                    'valid': False,
                    'message': f'Record {i+1}: {validation_result["message"]}'
                }
        
        # Check for duplicate pharmacy-drug combinations
        seen_combinations = set()
        for i, record in enumerate(records):
            combination = (
                record.get('Pharmacy_Name', '').strip(),
                record.get('Drug_ID', '').strip(),
                record.get('s-Date', '').strip()
            )
            if combination in seen_combinations:
                return {
                    'valid': False,
                    'message': f'Duplicate pharmacy-drug-date combination found at record {i+1}'
                }
            seen_combinations.add(combination)
        
        return {'valid': True, 'message': 'Valid pharmacy batch request'}
        
    except Exception as e:
        return {'valid': False, 'message': f'Batch validation error: {str(e)}'}


def validate_seasonal_consistency(season: str, date_str: str) -> Dict[str, Any]:
    """
    Validate that the provided season matches the date.
    
    Args:
        season: Rwanda season name
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Dictionary with 'valid' boolean and 'message' string
    """
    try:
        date_obj = pd.to_datetime(date_str)
        month = date_obj.month
        
        # Map months to seasons
        season_months = {
            'Urugaryi': [12, 1, 2],    # Dec-Feb: Short dry season
            'Itumba': [3, 4, 5],       # Mar-May: Long rainy season
            'Icyi': [6, 7, 8],         # Jun-Aug: Long dry season
            'Umuhindo': [9, 10, 11]    # Sep-Nov: Short rainy season
        }
        
        expected_season = None
        for s, months in season_months.items():
            if month in months:
                expected_season = s
                break
        
        if season != expected_season:
            return {
                'valid': False,
                'message': f'Season mismatch: {date_str} is in {expected_season} season, not {season}'
            }
        
        return {'valid': True, 'message': 'Season and date are consistent'}
        
    except Exception as e:
        return {'valid': False, 'message': f'Date validation error: {str(e)}'}


def validate_business_rules(data: Dict[str, Any]) -> List[str]:
    """
    Validate business rules and return list of warnings.
    
    Args:
        data: Pharmacy request data
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    try:
        # Stock level warnings
        stock = data.get('available_stock', 0)
        if stock < 50:
            warnings.append(f"Critical stock level: {stock} units")
        elif stock < 100:
            warnings.append(f"Low stock level: {stock} units")
        
        # Price vs income level warnings
        price = data.get('Price_Per_Unit', 0)
        income_level = data.get('Income_Level', 'medium')
        
        if price > 50 and income_level in ['low', 'medium']:
            warnings.append(f"High price (${price}) may limit accessibility in {income_level} income area")
        
        # Expiration warnings
        if 'expiration_date' in data and data['expiration_date']:
            try:
                exp_date = pd.to_datetime(data['expiration_date'])
                days_to_expiry = (exp_date - datetime.now()).days
                if days_to_expiry < 30:
                    warnings.append(f"Products expire in {days_to_expiry} days")
            except:
                pass
        
        # Seasonal mismatch warnings
        if 's-Date' in data and 'Season' in data:
            season_check = validate_seasonal_consistency(data['Season'], data['s-Date'])
            if not season_check['valid']:
                warnings.append(season_check['message'])
        
    except Exception as e:
        warnings.append(f"Business rule validation error: {str(e)}")
    
    return warnings
