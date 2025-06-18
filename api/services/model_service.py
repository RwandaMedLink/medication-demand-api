"""
Core model service for medication demand prediction
==================================================

This module contains the main business logic for loading and managing
the machine learning model, feature engineering, and making predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import will be updated after config is moved
from config import Config

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service class for handling ML model operations with Rwanda pharmacy-specific patterns.
    
    This class encapsulates all model-related functionality including:
    - Model loading and validation with Rwanda patterns
    - Feature engineering matching advanced_demand_prediction.py
    - Comprehensive pharmacy demand prediction
    - Business intelligence generation
    - Rwanda seasonal pattern analysis
    """
    
    def __init__(self, config=None):
        self.config = config
        self.model = None
        self.label_encoders = {}
        self.feature_mappings = {}
        self.model_metadata = {}
        self.feature_columns = None
        self.model_loaded = False
        self.feature_order = None
        
        # Rwanda-specific mappings (matching advanced_demand_prediction.py)
        self.rwanda_seasons = {
            'Itumba': 1,    # Mar-May: Long rainy season
            'Icyi': 2,      # Jun-Aug: Long dry season
            'Umuhindo': 3,  # Sep-Nov: Short rainy season
            'Urugaryi': 4   # Dec-Feb: Short dry season
        }
        
        # Income level mapping
        self.income_level_mapping = {
            'low': 1,
            'medium': 2,
            'higher': 3,
            'high': 4
        }
        
        # Population density mapping
        self.population_density_mapping = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        
        # Drug category mapping (ATC Code to Drug Category)
        self.drug_category_mapping = {
            'M01AB': 'Anti_Inflammatory',      # Diclofenac, Indomethacin
            'M01AE': 'Propionic_Acid',         # Ibuprofen, Naproxen
            'N02BA': 'Salicylic_Acid',         # Aspirin
            'N02BE': 'Paracetamol_Group',      # Paracetamol fever/pain relief
            'N02BB': 'Paracetamol_Group',      # Alternative paracetamol coding
            'N05B': 'Anxiolytics',             # Diazepam, Lorazepam
            'N05C': 'Sleep_Medications',       # Zolpidem, Zopiclone
            'R03': 'Respiratory_Drugs',        # Salbutamol, Budesonide
            'R06': 'Antihistamines'            # Loratadine, Cetirizine
        }
        
        # Seasonal demand multipliers (Rwanda pharmacy patterns)
        self.seasonal_multipliers = {
            'Paracetamol_Group': {  # Malaria-related medications
                'Itumba': 1.3,     # Moderate malaria season
                'Icyi': 0.8,       # Low transmission
                'Umuhindo': 1.4,   # PEAK malaria season
                'Urugaryi': 1.1    # Moderate levels
            },
            'Respiratory_Drugs': {  # Respiratory medications
                'Itumba': 1.5,     # Humidity-related issues
                'Icyi': 0.7,       # Lowest respiratory problems
                'Umuhindo': 1.6,   # Peak respiratory season
                'Urugaryi': 1.1    # Cold-triggered conditions
            },
            'Antihistamines': {     # Similar to respiratory
                'Itumba': 1.5,
                'Icyi': 0.7,
                'Umuhindo': 1.6,
                'Urugaryi': 1.1
            },
            'Anxiolytics': {        # Mental health medications
                'Itumba': 1.0,
                'Icyi': 1.0,
                'Umuhindo': 1.0,
                'Urugaryi': 1.2     # Holiday stress
            },
            'Sleep_Medications': {  # Similar to anxiolytics
                'Itumba': 1.0,
                'Icyi': 1.0,
                'Umuhindo': 1.0,
                'Urugaryi': 1.2
            },
            'Anti_Inflammatory': {  # Standard seasonal variation
                'Itumba': 1.1,
                'Icyi': 0.9,
                'Umuhindo': 1.2,
                'Urugaryi': 1.0
            },
            'Propionic_Acid': {     # Similar to anti-inflammatory
                'Itumba': 1.1,
                'Icyi': 0.9,
                'Umuhindo': 1.2,
                'Urugaryi': 1.0
            },
            'Salicylic_Acid': {     # Similar to anti-inflammatory
                'Itumba': 1.1,
                'Icyi': 0.9,
                'Umuhindo': 1.2,
                'Urugaryi': 1.0
            },
            'General': {            # Default for other drugs
                'Itumba': 1.0,
                'Icyi': 1.0,
                'Umuhindo': 1.0,
                'Urugaryi': 1.0
            }
        }
        
        # Core 7 pharmacy learning features (from training)
        self.core_pharmacy_features = [
            'Season_Numeric', 'Price_Per_Unit', 'available_stock', 
            'Effectiveness_Rating', 'Promotion', 'Population_Density_Numeric', 
            'Income_Level_Numeric'
        ]
        
        logger.info("Rwanda Pharmacy ModelService initialized with seasonal patterns")
    
    def _extract_feature_order(self):
        """
        Extract the feature order from the loaded model.
        This ensures features are in the correct order for prediction.
        """
        try:
            # Try to get feature order from model attributes
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_order = list(self.model.feature_names_in_)
                logger.info(f"Extracted feature order from model: {self.feature_order}")
            elif 'feature_columns' in self.model_metadata:
                self.feature_order = self.model_metadata['feature_columns']
                logger.info(f"Extracted feature order from metadata: {self.feature_order}")
            else:
                # Use the core pharmacy features as fallback
                self.feature_order = self.core_pharmacy_features.copy()
                logger.warning(f"Using fallback feature order: {self.feature_order}")
            
            # Update feature_columns for backward compatibility
            self.feature_columns = self.feature_order
            
        except Exception as e:
            logger.error(f"Error extracting feature order: {str(e)}")
            # Use core features as final fallback
            self.feature_order = self.core_pharmacy_features.copy()
            self.feature_columns = self.feature_order
            logger.warning(f"Using emergency fallback feature order: {self.feature_order}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the trained model and Rwanda-specific encoders/mappings.
        
        Args:
            model_path: Optional custom path to model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Try multiple model paths (prioritize pharmacy-specific models)
            model_paths = [
                model_path or 'models/pharmacy_linear_regression_label_r2_*.pkl',
                'models/linear_regression_label_r2_*.pkl',
                'models/best_model_*.pkl'
            ]
            
            model_dir = 'models'
            model_found = False
            
            # Find the most recent pharmacy model
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
                pharmacy_models = [f for f in model_files if 'pharmacy' in f.lower()]
                
                if pharmacy_models:
                    # Sort by modification time, get most recent
                    pharmacy_models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, pharmacy_models[0])
                elif model_files:
                    # Fallback to any available model
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
                    model_path = os.path.join(model_dir, model_files[0])
            
            if model_path and os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                    
                    # Handle both formats: direct model or model with metadata
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.model = model_data['model']
                        self.model_metadata = model_data.get('performance', {})
                        logger.info("Loaded pharmacy model with metadata")
                    else:
                        self.model = model_data
                        logger.info("Loaded direct model format")
                    
                    model_found = True
                    logger.info(f"Rwanda pharmacy model loaded from {model_path}")
                    
                except Exception as load_error:
                    logger.warning(f"Failed to load model from {model_path}: {load_error}")
            
            if not model_found:
                logger.warning("No pharmacy model found - operating in fallback mode")
                self.model = None
                self.model_loaded = False
                return False
            
            # Load additional pharmacy-specific files
            self._load_pharmacy_encoders(os.path.dirname(model_path))
            self._extract_feature_order()
            
            self.model_loaded = True
            logger.info("Rwanda pharmacy model system ready")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pharmacy model: {str(e)}")
            self.model = None
            self.model_loaded = False
            return False
    
    def _load_pharmacy_encoders(self, model_dir: str) -> None:
        """Load pharmacy-specific encoders and mappings."""
        try:
            # Load label encoders
            encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                logger.info("Pharmacy label encoders loaded")
            
            # Load feature mappings
            mappings_path = os.path.join(model_dir, 'feature_mappings.pkl')
            if os.path.exists(mappings_path):
                self.feature_mappings = joblib.load(mappings_path)
                logger.info("Pharmacy feature mappings loaded")
            
            # Load model metadata
            metadata_path = os.path.join(model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    additional_metadata = json.load(f)
                    self.model_metadata.update(additional_metadata)
                logger.info("Pharmacy model metadata loaded")
                
        except Exception as e:
            logger.warning(f"Could not load all pharmacy encoders: {e}")
    
    def predict_pharmacy_demand(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive pharmacy demand prediction with Rwanda-specific business intelligence.
        
        Args:
            request_data: Dictionary with pharmacy request fields
            
        Returns:
            Comprehensive prediction results with business intelligence
        """
        try:
            # Validate request
            validation_result = self.validate_pharmacy_request(request_data)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': f"Invalid request: {validation_result['message']}",
                    'pharmacy_info': None
                }
            
            # Extract pharmacy information
            pharmacy_info = {
                'Pharmacy_Name': request_data['Pharmacy_Name'],
                'Province': request_data['Province'],
                'Drug_ID': request_data['Drug_ID'],
                'ATC_Code': request_data.get('ATC_Code', 'Unknown'),
                'Drug_Category': self.drug_category_mapping.get(
                    request_data.get('ATC_Code', ''), 'General'
                )
            }
            
            # Parse date range
            start_date = pd.to_datetime(request_data['s-Date'])
            end_date = pd.to_datetime(request_data['E-Date'])
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            total_days = len(date_range)
            
            # Generate daily predictions
            daily_predictions = []
            total_predicted_demand = 0
            
            for date in date_range:
                # Get season for this date
                season = self._get_rwanda_season(date)
                
                # Create prediction record
                prediction_record = request_data.copy()
                prediction_record['Date'] = date
                prediction_record['Season'] = season
                
                # Get base prediction
                base_prediction, status = self.predict(prediction_record)
                if base_prediction is None:
                    base_prediction = self._fallback_prediction(prediction_record)
                
                # Apply seasonal multiplier
                adjusted_prediction, seasonal_multiplier = self.apply_seasonal_multiplier(
                    base_prediction, season, request_data.get('ATC_Code', '')
                )
                
                # Calculate confidence level
                confidence_level = self._calculate_confidence_level(prediction_record)
                
                daily_predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'season': season,
                    'base_prediction': base_prediction,
                    'seasonal_multiplier': seasonal_multiplier,
                    'final_prediction': adjusted_prediction,
                    'confidence_level': confidence_level
                })
                
                total_predicted_demand += adjusted_prediction
            
            # Calculate summary metrics
            average_daily_demand = total_predicted_demand / total_days if total_days > 0 else 0
            current_stock = request_data.get('available_stock', 0)
            stock_sufficiency_days = current_stock / average_daily_demand if average_daily_demand > 0 else 999
            
            # Generate restock recommendation
            restock_recommendation = self._generate_restock_recommendation(
                current_stock, total_predicted_demand, stock_sufficiency_days
            )
            
            # Generate business intelligence
            business_intelligence = self.generate_business_intelligence(request_data, daily_predictions)
            
            # Compile results
            results = {
                'success': True,
                'pharmacy_info': pharmacy_info,
                'prediction_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'total_days': total_days,
                    'primary_season': self._get_primary_season(date_range)
                },
                'daily_predictions': daily_predictions,
                'summary': {
                    'total_predicted_demand': round(total_predicted_demand),
                    'average_daily_demand': round(average_daily_demand, 2),
                    'current_stock': current_stock,
                    'stock_sufficiency_days': round(stock_sufficiency_days, 1),
                    'restock_recommendation': restock_recommendation
                },
                'business_intelligence': business_intelligence,
                'model_metadata': {
                    'model_used': 'Rwanda Pharmacy Prediction Model',
                    'prediction_confidence': self._get_overall_confidence(daily_predictions),
                    'feature_importance_ranking': self._get_feature_importance(),
                    'seasonal_pattern_applied': True
                }
            }
            
            logger.info(f"Generated pharmacy prediction for {pharmacy_info['Pharmacy_Name']} - {pharmacy_info['Drug_ID']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in pharmacy demand prediction: {str(e)}")
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}",
                'pharmacy_info': request_data.get('Pharmacy_Name', 'Unknown')
            }
    
    def apply_seasonal_multiplier(self, base_prediction: int, season: str, atc_code: str) -> Tuple[int, float]:
        """
        Apply Rwanda seasonal demand multiplier based on drug category.
        
        Args:
            base_prediction: Base prediction value
            season: Rwanda season name
            atc_code: ATC classification code
            
        Returns:
            Tuple of (adjusted_prediction, seasonal_multiplier)
        """
        try:
            # Get drug category
            drug_category = self.drug_category_mapping.get(atc_code, 'General')
            
            # Get seasonal multipliers for this drug category
            category_multipliers = self.seasonal_multipliers.get(drug_category, self.seasonal_multipliers['General'])
            
            # Get multiplier for this season
            seasonal_multiplier = category_multipliers.get(season, 1.0)
            
            # Apply multiplier
            adjusted_prediction = int(base_prediction * seasonal_multiplier)
            
            return adjusted_prediction, seasonal_multiplier
            
        except Exception as e:
            logger.warning(f"Error applying seasonal multiplier: {e}")
            return base_prediction, 1.0
    
    def generate_business_intelligence(self, request_data: Dict[str, Any], daily_predictions: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive business intelligence for pharmacy operations.
        
        Args:
            request_data: Original request data
            daily_predictions: List of daily prediction data
            
        Returns:
            Business intelligence insights dictionary
        """
        try:
            # Market positioning analysis
            market_positioning = self._analyze_market_positioning(request_data)
            
            # Seasonal insights
            seasonal_insights = self._generate_seasonal_insights(request_data, daily_predictions)
            
            # Pricing analysis
            pricing_analysis = self._analyze_pricing_competitiveness(request_data)
            
            # Promotional impact assessment
            promotional_impact = self._assess_promotional_effectiveness(request_data)
            
            # Risk factors identification
            risk_factors = self._identify_risk_factors(request_data)
            
            # Optimization suggestions
            optimization_suggestions = self._generate_optimization_suggestions(request_data, risk_factors)
            
            return {
                'seasonal_insights': seasonal_insights,
                'market_positioning': market_positioning,
                'pricing_analysis': pricing_analysis,
                'promotional_impact': promotional_impact,
                'risk_factors': risk_factors,
                'optimization_suggestions': optimization_suggestions
            }
            
        except Exception as e:
            logger.error(f"Error generating business intelligence: {e}")
            return {'error': 'Could not generate business intelligence'}
    
    def _analyze_market_positioning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market positioning based on demographics."""
        population_density = data.get('Population_Density', 'medium')
        income_level = data.get('Income_Level', 'medium')
        
        positioning = {
            'target_market': f"{income_level.title()} income, {population_density} density area",
            'market_size_indicator': self.population_density_mapping.get(population_density, 2),
            'affordability_score': self.income_level_mapping.get(income_level, 2),
            'market_attractiveness': 'High' if population_density == 'high' and income_level in ['higher', 'high'] else 'Medium'
        }
        
        return positioning
    
    def _generate_seasonal_insights(self, data: Dict[str, Any], predictions: List[Dict]) -> Dict[str, Any]:
        """Generate insights about seasonal demand patterns."""
        atc_code = data.get('ATC_Code', '')
        drug_category = self.drug_category_mapping.get(atc_code, 'General')
        
        # Get peak season for this drug category
        multipliers = self.seasonal_multipliers.get(drug_category, self.seasonal_multipliers['General'])
        peak_season = max(multipliers.items(), key=lambda x: x[1])
        low_season = min(multipliers.items(), key=lambda x: x[1])
        
        insights = {
            'drug_category': drug_category,
            'peak_season': {
                'season': peak_season[0],
                'multiplier': peak_season[1],
                'period': self._get_season_period(peak_season[0])
            },
            'low_season': {
                'season': low_season[0],
                'multiplier': low_season[1],
                'period': self._get_season_period(low_season[0])
            },
            'seasonal_variation': f"{((peak_season[1] - low_season[1]) / low_season[1] * 100):.0f}%",
            'business_implication': self._get_seasonal_business_implication(drug_category)
        }
        
        return insights
    
    def _analyze_pricing_competitiveness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pricing strategy competitiveness."""
        price = data.get('Price_Per_Unit', 30)
        effectiveness = data.get('Effectiveness_Rating', 5)
        income_level = data.get('Income_Level', 'medium')
        
        # Calculate value score
        value_score = effectiveness / (price / 10) if price > 0 else 0
        
        analysis = {
            'price_per_unit': price,
            'effectiveness_rating': effectiveness,
            'value_score': round(value_score, 2),
            'price_positioning': 'High' if price > 40 else 'Medium' if price > 20 else 'Low',
            'affordability_for_market': self._assess_affordability(price, income_level),
            'pricing_recommendation': self._get_pricing_recommendation(price, effectiveness, income_level)
        }
        
        return analysis
    
    def _assess_promotional_effectiveness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess promotional activity effectiveness."""
        promotion = data.get('Promotion', 0)
        season = data.get('Season', 'Itumba')
        population_density = data.get('Population_Density', 'medium')
        
        effectiveness_score = promotion * self.population_density_mapping.get(population_density, 2)
        
        assessment = {
            'current_promotion': bool(promotion),
            'promotional_effectiveness_score': effectiveness_score,
            'optimal_promotional_seasons': self._get_optimal_promotional_seasons(),
            'demographic_suitability': population_density,
            'promotional_recommendation': self._get_promotional_recommendation(promotion, season, population_density)
        }
        
        return assessment
    
    def _identify_risk_factors(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential risk factors for pharmacy operations."""
        risks = []
        
        # Stock level risk
        stock = data.get('available_stock', 0)
        if stock < 100:
            risks.append({
                'type': 'Low Stock Alert',
                'severity': 'High' if stock < 50 else 'Medium',
                'description': f"Current stock ({stock} units) is critically low",
                'recommendation': 'Immediate restock required'
            })
        
        # Expiration risk
        if 'expiration_date' in data:
            try:
                exp_date = pd.to_datetime(data['expiration_date'])
                days_to_expiry = (exp_date - datetime.now()).days
                if days_to_expiry < 30:
                    risks.append({
                        'type': 'Expiration Risk',
                        'severity': 'High' if days_to_expiry < 7 else 'Medium',
                        'description': f"Products expire in {days_to_expiry} days",
                        'recommendation': 'Consider promotional pricing to move inventory'
                    })
            except:
                pass
        
        # Pricing risk
        price = data.get('Price_Per_Unit', 30)
        income_level = data.get('Income_Level', 'medium')
        if price > 50 and income_level in ['low', 'medium']:
            risks.append({
                'type': 'Pricing Accessibility',
                'severity': 'Medium',
                'description': f"High price ({price}) may limit accessibility in {income_level} income area",
                'recommendation': 'Consider tiered pricing or generic alternatives'
            })
        
        return risks
    
    def _generate_optimization_suggestions(self, data: Dict[str, Any], risks: List[Dict]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for pharmacy operations."""
        suggestions = []
        
        # Inventory optimization
        stock = data.get('available_stock', 0)
        if stock > 1000:
            suggestions.append({
                'category': 'Inventory Management',
                'suggestion': 'Consider reducing inventory levels to improve cash flow',
                'impact': 'Cost Reduction',
                'priority': 'Medium'
            })
        
        # Seasonal stocking
        atc_code = data.get('ATC_Code', '')
        drug_category = self.drug_category_mapping.get(atc_code, 'General')
        if drug_category in ['Paracetamol_Group', 'Respiratory_Drugs']:
            suggestions.append({
                'category': 'Seasonal Planning',
                'suggestion': f'Increase inventory before peak seasons for {drug_category}',
                'impact': 'Revenue Optimization',
                'priority': 'High'
            })
        
        # Pricing optimization
        price = data.get('Price_Per_Unit', 30)
        effectiveness = data.get('Effectiveness_Rating', 5)
        if effectiveness >= 8 and price < 30:
            suggestions.append({
                'category': 'Pricing Strategy',
                'suggestion': 'Consider premium pricing for high-effectiveness medications',
                'impact': 'Margin Improvement',
                'priority': 'Medium'
            })
        
        return suggestions
    
    def validate_pharmacy_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pharmacy prediction request."""
        required_fields = ['Pharmacy_Name', 'Drug_ID', 's-Date', 'E-Date', 'Season']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            return {
                'valid': False,
                'message': f"Missing required fields: {missing_fields}"
            }
        
        # Validate season
        season = request_data.get('Season')
        if season not in self.rwanda_seasons:
            return {
                'valid': False,
                'message': f"Invalid season. Must be one of: {list(self.rwanda_seasons.keys())}"
            }
        
        # Validate dates
        try:
            start_date = pd.to_datetime(request_data['s-Date'])
            end_date = pd.to_datetime(request_data['E-Date'])
            if start_date > end_date:
                return {
                    'valid': False,
                    'message': "Start date must be before or equal to end date"
                }
        except:
            return {
                'valid': False,
                'message': "Invalid date format. Use YYYY-MM-DD"
            }
        
        # Validate numeric fields
        numeric_fields = ['available_stock', 'Price_Per_Unit', 'Effectiveness_Rating']
        for field in numeric_fields:
            if field in request_data:
                try:
                    float(request_data[field])
                except:
                    return {
                        'valid': False,
                        'message': f"Invalid numeric value for {field}"
                    }
        
        return {'valid': True, 'message': 'Valid request'}
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[Optional[int], str]:
        """Enhanced predict method handling exact request format."""
        if not self.model_loaded:
            return self._fallback_prediction(input_data), "Prediction using fallback method (model not loaded)"
        
        try:
            # Create features matching training model
            features = self._create_simple_features(input_data)
            if features is None:
                return self._fallback_prediction(input_data), "Feature creation failed - using fallback"
            
            # Create DataFrame with proper feature order
            feature_df = self._create_feature_dataframe(features)
            
            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            prediction = max(0, round(prediction))
            
            return prediction, "Success"
            
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return self._fallback_prediction(input_data), f"Model error - using fallback: {str(e)}"
    
    def _create_simple_features(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Create features matching advanced_demand_prediction training."""
        try:
            features = {}
            
            # Core 7 pharmacy learning features (exactly as trained)
            
            # 1. Season (Rwanda seasonal numeric mapping)
            season = data.get('Season', 'Itumba')
            features['Season_Numeric'] = float(self.rwanda_seasons.get(season, 1))
            
            # 2. Price_Per_Unit (pricing strategy)
            features['Price_Per_Unit'] = float(data.get('Price_Per_Unit', 30.0))
            
            # 3. available_stock (inventory levels)
            features['available_stock'] = float(data.get('available_stock', 0))
            
            # 4. Effectiveness_Rating (customer preference)
            features['Effectiveness_Rating'] = float(data.get('Effectiveness_Rating', 5))
            
            # 5. Promotion (promotional impact)
            features['Promotion'] = float(data.get('Promotion', 0))
            
            # 6. Population_Density (catchment demographics)
            pop_density = data.get('Population_Density', 'medium')
            features['Population_Density_Numeric'] = float(self.population_density_mapping.get(pop_density, 2))
            
            # 7. Income_Level (purchasing power)
            income = data.get('Income_Level', 'medium')
            features['Income_Level_Numeric'] = float(self.income_level_mapping.get(income, 2))
            
            # Derived features (matching training)
            
            # Seasonal multiplier
            atc_code = data.get('ATC_Code', '')
            season = data.get('Season', 'Itumba')
            _, seasonal_mult = self.apply_seasonal_multiplier(1, season, atc_code)
            features['Seasonal_Multiplier'] = float(seasonal_mult)
            
            # Stock turnover ratio
            units_sold_est = features['available_stock'] * 0.1  # Estimation
            features['Stock_Turnover_Ratio'] = units_sold_est / (features['available_stock'] + 1)
            
            # Days until expiry
            if 'expiration_date' in data:
                try:
                    exp_date = pd.to_datetime(data['expiration_date'])
                    current_date = pd.to_datetime(data.get('Date', datetime.now()))
                    days_to_expiry = (exp_date - current_date).days
                    features['Days_Until_Expiry'] = float(max(0, days_to_expiry))
                except:
                    features['Days_Until_Expiry'] = 60.0
            else:
                features['Days_Until_Expiry'] = 60.0
            
            # Days since stock entry
            if 'stock_entry_timestamp' in data:
                try:
                    entry_date = pd.to_datetime(data['stock_entry_timestamp'])
                    current_date = pd.to_datetime(data.get('Date', datetime.now()))
                    days_since_entry = (current_date - entry_date).days
                    features['Days_Since_Stock_Entry'] = float(max(0, days_since_entry))
                except:
                    features['Days_Since_Stock_Entry'] = 30.0
            else:
                features['Days_Since_Stock_Entry'] = 30.0
            
            # Price effectiveness ratio
            features['Price_Effectiveness_Ratio'] = features['Price_Per_Unit'] / (features['Effectiveness_Rating'] + 0.1)
            
            # Promotional demographic impact
            features['Promo_Demo_Impact'] = (features['Promotion'] * 
                                           features['Population_Density_Numeric'] * 
                                           features['Income_Level_Numeric'])
            
            # Categorical features (using label encoding for compatibility)
            features['Pharmacy_Name'] = float(abs(hash(str(data.get('Pharmacy_Name', 'default')))) % 1000)
            features['Province'] = float(abs(hash(str(data.get('Province', 'default')))) % 100)
            features['Drug_ID'] = float(abs(hash(str(data.get('Drug_ID', 'default')))) % 1000)
            features['ATC_Code'] = float(abs(hash(str(data.get('ATC_Code', 'default')))) % 100)
            
            # Drug category
            drug_category = self.drug_category_mapping.get(data.get('ATC_Code', ''), 'General')
            features['Drug_Category'] = float(abs(hash(drug_category)) % 100)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return None
    
    def _get_rwanda_season(self, date) -> str:
        """Get Rwanda season for a given date."""
        month = date.month
        if month in [12, 1, 2]:
            return 'Urugaryi'  # Short dry season
        elif month in [3, 4, 5]:
            return 'Itumba'    # Long rainy season
        elif month in [6, 7, 8]:
            return 'Icyi'      # Long dry season
        else:  # 9, 10, 11
            return 'Umuhindo'  # Short rainy season
    
    def _get_season_period(self, season: str) -> str:
        """Get period description for Rwanda season."""
        periods = {
            'Urugaryi': 'December-February (Short Dry)',
            'Itumba': 'March-May (Long Rainy)',
            'Icyi': 'June-August (Long Dry)',
            'Umuhindo': 'September-November (Short Rainy)'
        }
        return periods.get(season, 'Unknown')
    
    def _get_seasonal_business_implication(self, drug_category: str) -> str:
        """Get business implication for drug category seasonality."""
        implications = {
            'Paracetamol_Group': 'Stock up before malaria seasons (Umuhindo peak, Icyi low)',
            'Respiratory_Drugs': 'Peak demand during rainy seasons due to respiratory issues',
            'Antihistamines': 'Similar to respiratory - plan for humid seasons',
            'Anxiolytics': 'Stable demand with slight increase during holiday stress',
            'Sleep_Medications': 'Consistent demand with holiday season increase',
            'Anti_Inflammatory': 'Moderate seasonal variation - plan accordingly'
        }
        return implications.get(drug_category, 'Monitor seasonal patterns for optimal stocking')
    
    def _fallback_prediction(self, input_data: Dict[str, Any]) -> int:
        """
        Provide a reasonable prediction when the ML model is not available.
        This uses simple business rules based on Rwanda healthcare patterns.
        """
        try:
            # Base demand estimation
            base_demand = 100
            
            # Factor in available stock (if low stock, higher demand likely)
            stock = input_data.get('available_stock', 500)
            if stock < 100:
                base_demand += 50
            elif stock < 300:
                base_demand += 20
            
            # Factor in price (higher price, lower demand)
            price = input_data.get('Price_Per_Unit', 30)
            if price > 40:
                base_demand -= 30
            elif price < 20:
                base_demand += 20
            
            # Factor in promotion
            promotion = input_data.get('Promotion', 0)
            if promotion:
                base_demand += 40
            
            # Factor in disease outbreak
            outbreak = input_data.get('Disease_Outbreak', 0)
            if outbreak:
                base_demand += 60
            
            # Factor in effectiveness rating
            effectiveness = input_data.get('Effectiveness_Rating', 5)
            if effectiveness >= 8:
                base_demand += 30
            elif effectiveness <= 3:
                base_demand -= 20
            
            # Factor in holiday week
            holiday = input_data.get('Holiday_Week', 0)
            if holiday:
                base_demand += 25
            
            # Factor in province (Kigali typically has higher demand)
            province = input_data.get('Province', '')
            if province == 'Kigali':
                base_demand += 20
            
            # Factor in population density
            density = input_data.get('Population_Density', 'medium')
            if density == 'high':
                base_demand += 15
            elif density == 'low':
                base_demand -= 10
            
            # Ensure minimum reasonable demand
            final_demand = max(10, base_demand)
            
            logger.info(f"Fallback prediction: {final_demand} (base: {100}, final: {final_demand})")
            return final_demand
            
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return 50  # Default safe value
    
    def _calculate_confidence_level(self, data: Dict[str, Any]) -> float:
        """Calculate confidence level for a prediction."""
        try:
            confidence = 0.5  # Base confidence
            
            # Increase confidence if we have good stock data
            if data.get('available_stock', 0) > 0:
                confidence += 0.2
            
            # Increase confidence if model is loaded
            if self.model_loaded:
                confidence += 0.2
            
            # Increase confidence for complete data
            required_fields = ['Price_Per_Unit', 'Effectiveness_Rating', 'Population_Density', 'Income_Level']
            complete_fields = sum(1 for field in required_fields if data.get(field) is not None)
            confidence += (complete_fields / len(required_fields)) * 0.1
            
            return min(1.0, confidence)
        except:
            return 0.6  # Default moderate confidence
    
    def _generate_restock_recommendation(self, current_stock: int, predicted_demand: float, sufficiency_days: float) -> str:
        """Generate restock recommendation based on stock and demand."""
        if sufficiency_days < 7:
            return "URGENT: Restock immediately - less than 1 week supply remaining"
        elif sufficiency_days < 14:
            return "HIGH PRIORITY: Restock within 3-5 days - less than 2 weeks supply"
        elif sufficiency_days < 30:
            return "MEDIUM PRIORITY: Plan restock within 1-2 weeks"
        elif sufficiency_days < 60:
            return "LOW PRIORITY: Monitor and plan restock in 3-4 weeks"
        else:
            return "ADEQUATE: Current stock sufficient for 2+ months"
    
    def _get_primary_season(self, date_range) -> str:
        """Get the primary season for a date range."""
        season_counts = {}
        for date in date_range:
            season = self._get_rwanda_season(date)
            season_counts[season] = season_counts.get(season, 0) + 1
        
        return max(season_counts.items(), key=lambda x: x[1])[0]
    
    def _get_overall_confidence(self, daily_predictions: List[Dict]) -> float:
        """Calculate overall confidence from daily predictions."""
        if not daily_predictions:
            return 0.6
        
        confidences = [pred.get('confidence_level', 0.6) for pred in daily_predictions]
        return sum(confidences) / len(confidences)
    
    def _get_feature_importance(self) -> List[str]:
        """Get feature importance ranking."""
        if hasattr(self.model, 'feature_importances_') and self.feature_order:
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_order, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return [f[0] for f in feature_importance[:5]]  # Top 5 features
        else:
            return self.core_pharmacy_features[:5]  # Fallback to core features
    
    def _create_feature_dataframe(self, features: Dict[str, float]) -> pd.DataFrame:
        """Create properly ordered DataFrame for model prediction."""
        if self.feature_order:
            # Use model's expected feature order
            ordered_features = {col: features.get(col, 0.0) for col in self.feature_order}
            return pd.DataFrame([ordered_features])
        else:
            # Use core features as fallback
            ordered_features = {col: features.get(col, 0.0) for col in self.core_pharmacy_features}
            return pd.DataFrame([ordered_features])
    
    def _assess_affordability(self, price: float, income_level: str) -> str:
        """Assess affordability based on price and income level."""
        income_factor = self.income_level_mapping.get(income_level, 2)
        
        if price > 50 and income_factor <= 2:
            return "Low - Price may be too high for market"
        elif price > 30 and income_factor <= 1:
            return "Very Low - Price significantly exceeds market capacity"
        elif price < 20 and income_factor >= 3:
            return "High - Good value proposition for market"
        else:
            return "Moderate - Appropriately priced for market"
    
    def _get_pricing_recommendation(self, price: float, effectiveness: float, income_level: str) -> str:
        """Get pricing strategy recommendation."""
        income_factor = self.income_level_mapping.get(income_level, 2)
        value_ratio = effectiveness / (price / 10) if price > 0 else 0
        
        if value_ratio > 2 and income_factor >= 3:
            return "Consider premium pricing - high value in affluent market"
        elif value_ratio < 1 and income_factor <= 2:
            return "Reduce price or improve value proposition"
        elif price > 40 and income_factor <= 2:
            return "Consider tiered pricing or generic alternatives"
        else:
            return "Current pricing strategy appears appropriate"
    
    def _get_optimal_promotional_seasons(self) -> List[str]:
        """Get optimal seasons for promotions."""
        return ['Umuhindo', 'Itumba']  # Rainy seasons typically have higher demand
    
    def _get_promotional_recommendation(self, promotion: int, season: str, population_density: str) -> str:
        """Get promotional strategy recommendation."""
        if not promotion and season in ['Umuhindo', 'Itumba'] and population_density == 'high':
            return "Consider launching promotion - optimal conditions for high impact"
        elif promotion and season in ['Icyi'] and population_density == 'low':
            return "Current promotion may have limited impact - consider reducing spend"
        elif not promotion and population_density == 'high':
            return "Consider seasonal promotion to capture market share"
        else:
            return "Current promotional strategy appears appropriate"
