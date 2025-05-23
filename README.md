# Rwanda MedLink - Medication Demand Prediction System

This project provides a comprehensive system for predicting medication demand, analyzing key sales drivers, and generating restock recommendations for healthcare facilities in Rwanda.

## Overview

The Rwanda MedLink system uses machine learning to analyze historical medication sales data and identify patterns that influence demand. It can:

1. **Predict future medication demand** based on multiple factors (seasonality, promotions, etc.)
2. **Analyze key sales drivers** to understand what factors most influence medication sales
3. **Generate restock recommendations** with prioritization based on current stock levels
4. **Visualize sales patterns** across different dimensions (time, location, drug type)

## Dataset

The system works with medication sales data that includes the following key fields:
- `Drug_ID`: Medication identifier
- `ATC_Code`: Anatomical Therapeutic Chemical classification
- `Date`: Transaction date
- `Province`: Geographic location 
- `Health_Center`: Healthcare facility
- `units_sold`: Quantity sold (target variable)
- Other features: price, availability, promotions, seasonality, etc.

## Files Included

- `medication_demand_prediction.py`: Main prediction pipeline with feature engineering
- `advanced_demand_prediction.py`: Advanced prediction model with hyperparameter tuning
- `sales_drivers_analysis.py`: Detailed analysis of factors influencing medication sales
- `restock_recommendation_system.py`: System for generating actionable restock recommendations

## Requirements

Required Python packages:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
tabulate
```

## Usage Instructions

### 1. Basic Demand Prediction

Run the main prediction script:

```bash
python medication_demand_prediction.py
```

This will:
- Load and preprocess the dataset
- Engineer relevant features
- Train multiple models (Linear Regression, Random Forest, SVM)
- Evaluate model performance
- Generate initial restock recommendations

### 2. Advanced Demand Prediction

For more advanced modeling with hyperparameter tuning:

```bash
python advanced_demand_prediction.py
```

This includes:
- More sophisticated feature engineering
- Multiple model comparison (LR, Ridge, Lasso, RF, GB, SVM)
- Hyperparameter tuning via GridSearchCV
- Detailed model evaluation and visualization
- Feature importance analysis

### 3. Sales Drivers Analysis

To analyze what factors drive medication sales:

```bash
python sales_drivers_analysis.py
```

This will generate visualizations and insights about:
- Seasonal impact on drug sales
- Promotion effectiveness
- Price elasticity
- Geographic patterns
- Key sales drivers

### 4. Restock Recommendations

To generate actionable restock recommendations:

```bash
python restock_recommendation_system.py [data_path] [model_path] [days_to_predict]
```

Parameters:
- `data_path`: Path to the dataset (default: 'synthetic_pharma_sales.csv')
- `model_path`: Path to the saved model (optional, will search in 'models' directory)
- `days_to_predict`: Number of days to predict ahead (default: 30)

This will:
- Load the trained model and dataset
- Generate predictions for future demand
- Calculate recommended restock quantities
- Prioritize items based on days of stock remaining
- Generate an interactive HTML report
- Print a summary of restock recommendations

## Output Files

The system generates several output files:

### Models
- `models/best_model_*.pkl`: Saved trained model

### Visualizations
- `figures/feature_correlations.png`: Feature correlations with sales
- `figures/feature_importance.png`: Most important features for prediction
- `figures/seasonal_impact.png`: Seasonal patterns in drug sales
- `figures/promotion_impact.png`: Effect of promotions on sales
- `figures/geographic_sales.png`: Sales patterns by province
- And more...

### Reports
- `restock_recommendations.csv`: Detailed restock recommendations
- `reports/restock_report.html`: Interactive HTML report with visualizations

## Analysis Workflow

For a complete analysis, follow this workflow:

1. Run `medication_demand_prediction.py` for initial data exploration and modeling
2. Run `sales_drivers_analysis.py` to understand key factors influencing sales
3. Run `advanced_demand_prediction.py` to create a more sophisticated predictive model
4. Run `restock_recommendation_system.py` to generate actionable restock recommendations

## Example Use Cases

1. **Inventory Planning**: Predict demand for the next 30 days to optimize stock levels
2. **Marketing Optimization**: Identify which drugs respond best to promotions
3. **Seasonal Planning**: Prepare for seasonal variations in medication demand
4. **Budget Allocation**: Prioritize restock budget based on criticality and demand
5. **Supply Chain Management**: Anticipate potential shortages before they occur

## Customization

The system can be customized by:
- Modifying feature engineering in each script
- Adjusting hyperparameter search spaces
- Changing the safety stock levels in restock calculations
- Modifying priority thresholds for restock recommendations

## License

This project is provided for educational and demonstration purposes.
