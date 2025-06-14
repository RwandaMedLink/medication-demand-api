"""
Validation utilities for the Rwanda MedLink API.
Based on the actual model training data structure from advanced_demand_prediction.py
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import re


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input data for single prediction endpoint.
    Based on the actual dataset structure used in model training.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Cleaned and validated data dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("Input must be a dictionary")
    
    # Required fields (minimum needed for prediction)
    required_fields = [
        'Pharmacy_Name',
        'Province', 
        'Drug_ID',
        'Date'
    ]
    
    # Check required fields
    missing_fields = []
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == '':
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")
    
    # Validate and clean data
    cleaned_data = {}
    
    # String fields (based on actual dataset columns)
    string_fields = ['Pharmacy_Name', 'Province', 'Drug_ID', 'ATC_Code', 'Season', 
                    'Supply_Chain_Delay', 'Population_Density', 'Income_Level']
    
    for field in string_fields:
        if field in data:
            cleaned_data[field] = str(data[field]).strip()
    
    # Date validation (supports multiple formats as seen in CSV)
    if 'Date' in data:
        cleaned_data['Date'] = validate_date(data['Date'])
    
    # Optional date fields
    for date_field in ['expiration_date', 'stock_entry_timestamp']:
        if date_field in data:
            cleaned_data[date_field] = validate_date(data[date_field])
    
    # Numeric fields with validation (based on actual dataset)
    numeric_fields = {
        'available_stock': {'min': 0, 'type': int, 'default': 0},
        'Price_Per_Unit': {'min': 0, 'type': float, 'default': 0.0},
        'Promotion': {'min': 0, 'max': 1, 'type': int, 'default': 0},
        'Disease_Outbreak': {'min': 0, 'max': 2, 'type': int, 'default': 0},
        'Effectiveness_Rating': {'min': 1, 'max': 5, 'type': int, 'default': 3},
        'Competitor_Count': {'min': 0, 'type': int, 'default': 0},
        'Time_On_Market': {'min': 0, 'type': int, 'default': 1},
        'Holiday_Week': {'min': 0, 'max': 1, 'type': int, 'default': 0}
    }
    
    for field, constraints in numeric_fields.items():
        if field in data:
            cleaned_data[field] = validate_numeric_field(
                data[field], field, constraints
            )
        else:
            # Set default value if not provided
            cleaned_data[field] = constraints['default']
    
    # Validate Province (based on Rwanda provinces)
    valid_provinces = [
        'Kigali', 'Eastern', 'Western', 'Northern', 'Southern'
    ]
    if 'Province' in cleaned_data:
        if cleaned_data['Province'] not in valid_provinces:
            raise ValidationError(f"Invalid Province. Must be one of: {valid_provinces}")
    
    # Validate Season (Rwandan seasons)
    valid_seasons = ['Itumba', 'Urugaryi', 'Icyi', 'Umuhindo']
    if 'Season' in cleaned_data:
        if cleaned_data['Season'] not in valid_seasons:
            cleaned_data['Season'] = 'Urugaryi'  # Default season
    else:
        cleaned_data['Season'] = 'Urugaryi'  # Default season
    
    # Validate Supply Chain Delay
    valid_delays = ['None', 'Low', 'Medium', 'High']
    if 'Supply_Chain_Delay' in cleaned_data:
        if cleaned_data['Supply_Chain_Delay'] not in valid_delays:
            cleaned_data['Supply_Chain_Delay'] = 'None'  # Default
    else:
        cleaned_data['Supply_Chain_Delay'] = 'None'  # Default
    
    # Validate Population Density
    valid_densities = ['low', 'medium', 'high']
    if 'Population_Density' in cleaned_data:
        if cleaned_data['Population_Density'] not in valid_densities:
            cleaned_data['Population_Density'] = 'medium'  # Default
    else:
        cleaned_data['Population_Density'] = 'medium'  # Default
    
    # Validate Income Level
    valid_income_levels = ['lower', 'middle', 'higher']
    if 'Income_Level' in cleaned_data:
        if cleaned_data['Income_Level'] not in valid_income_levels:
            cleaned_data['Income_Level'] = 'middle'  # Default
    else:
        cleaned_data['Income_Level'] = 'middle'  # Default
    
    # Validate ATC Code format (if provided)
    if 'ATC_Code' in cleaned_data and cleaned_data['ATC_Code']:
        if not re.match(r'^[A-Z][0-9]{2}[A-Z]{1,2}$', cleaned_data['ATC_Code']):
            # If invalid ATC code, keep it as is (model might handle it)
            pass
    else:
        cleaned_data['ATC_Code'] = 'M01AB'  # Default ATC code
    
    return cleaned_data


def validate_batch_prediction_input(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate input data for batch prediction endpoint.
    
    Args:
        data: List of input data dictionaries
        
    Returns:
        List of cleaned and validated data dictionaries
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, list):
        raise ValidationError("Input must be a list of dictionaries")
    
    if len(data) == 0:
        raise ValidationError("Input list cannot be empty")
    
    if len(data) > 1000:  # Limit batch size
        raise ValidationError("Batch size cannot exceed 1000 items")
    
    validated_data = []
    for i, item in enumerate(data):
        try:
            validated_item = validate_prediction_input(item)
            validated_data.append(validated_item)
        except ValidationError as e:
            raise ValidationError(f"Item {i}: {str(e)}")
    
    return validated_data


def validate_date(date_input: Any) -> str:
    """
    Validate and format date input.
    Supports multiple formats as seen in the actual dataset.
    
    Args:
        date_input: Date string or datetime object
        
    Returns:
        Formatted date string (YYYY-MM-DD)
        
    Raises:
        ValidationError: If date is invalid
    """
    if isinstance(date_input, datetime):
        return date_input.strftime('%Y-%m-%d')
    
    if isinstance(date_input, str):
        # Try multiple date formats (as seen in the actual CSV)
        formats = [
            '%Y-%m-%d', 
            '%m/%d/%Y',     # CSV format: 1/1/2024
            '%d/%m/%Y', 
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M'  # CSV format: 1/1/2024 10:05
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_input.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        raise ValidationError(f"Invalid date format: {date_input}. Expected formats: YYYY-MM-DD or M/D/YYYY")
    
    raise ValidationError("Date must be a string or datetime object")


def validate_numeric_field(value: Any, field_name: str, constraints: Dict[str, Any]) -> Any:
    """
    Validate numeric field with constraints.
    
    Args:
        value: Value to validate
        field_name: Name of the field
        constraints: Dictionary with 'min', 'max', 'type' keys
        
    Returns:
        Validated and converted value
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Convert to appropriate type
        if constraints['type'] == int:
            converted_value = int(float(value))  # Handle string numbers
        else:
            converted_value = constraints['type'](value)
        
        # Check min constraint
        if 'min' in constraints and converted_value < constraints['min']:
            raise ValidationError(f"{field_name} must be >= {constraints['min']}")
        
        # Check max constraint
        if 'max' in constraints and converted_value > constraints['max']:
            raise ValidationError(f"{field_name} must be <= {constraints['max']}")
        
        return converted_value
    
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid {field_name}: must be a valid {constraints['type'].__name__}")


def validate_pharmacy_name(name: str) -> str:
    """
    Validate pharmacy name format.
    Based on patterns seen in the actual dataset (e.g., "CityMeds 795").
    
    Args:
        name: Pharmacy name string
        
    Returns:
        Validated pharmacy name
        
    Raises:
        ValidationError: If name is invalid
    """
    if not isinstance(name, str) or len(name.strip()) == 0:
        raise ValidationError("Pharmacy name cannot be empty")
    
    # Allow alphanumeric, spaces, and common punctuation (based on actual data)
    if not re.match(r'^[a-zA-Z0-9\s\.\-_&]+$', name.strip()):
        raise ValidationError("Pharmacy name contains invalid characters")
    
    return name.strip()


def validate_drug_id(drug_id: str) -> str:
    """
    Validate drug ID format.
    Based on patterns seen in the actual dataset (e.g., "DICLOFENAC").
    
    Args:
        drug_id: Drug ID string
        
    Returns:
        Validated drug ID
        
    Raises:
        ValidationError: If drug ID is invalid
    """
    if not isinstance(drug_id, str) or len(drug_id.strip()) == 0:
        raise ValidationError("Drug ID cannot be empty")
    
    # Allow alphanumeric and underscore for drug IDs (based on actual data)
    if not re.match(r'^[a-zA-Z0-9_]+$', drug_id.strip()):
        raise ValidationError("Drug ID must contain only letters, numbers, and underscores")
    
    return drug_id.strip().upper()


def sanitize_string_input(value: str, max_length: int = 255) -> str:
    """
    Sanitize string input by removing dangerous characters.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If string is too long
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', value.strip())
    
    if len(sanitized) > max_length:
        raise ValidationError(f"String too long. Maximum {max_length} characters allowed")
    
    return sanitized
