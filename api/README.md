# Rwanda MedLink API

A sophisticated Flask-based REST API for medication demand prediction in Rwanda's healthcare system. This API provides machine learning-powered predictions to help healthcare facilities optimize medication inventory and improve patient care.

## 🚀 Features

- **Machine Learning Predictions**: Advanced ML models for medication demand forecasting
- **Modular Architecture**: Clean, maintainable codebase with separation of concerns
- **RESTful API**: Standard HTTP methods and status codes
- **Comprehensive Validation**: Input validation and error handling
- **Performance Monitoring**: Request timing and performance metrics
- **Security**: Built-in security headers and rate limiting
- **Professional Web Interface**: User-friendly web forms for predictions
- **Batch Processing**: Support for bulk predictions
- **Health Monitoring**: Health check and status endpoints
- **Configurable**: Environment-specific configurations
- **Extensive Logging**: Comprehensive logging and error tracking

## 📁 Project Structure

```
api/
├── __init__.py                 # Package initialization
├── app_factory.py             # Flask application factory
├── main.py                    # Application entry point
├── config.py                  # Configuration classes
├── middleware.py              # Custom middleware components
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── routes/                    # API routes and blueprints
│   └── __init__.py           # Route definitions and blueprints
│
├── services/                  # Business logic services
│   ├── __init__.py           # Service package
│   └── model_service.py      # ML model service
│
└── utils/                     # Utility modules
    ├── __init__.py           # Utility package
    ├── validators.py         # Input validation
    ├── responses.py          # Response formatting
    ├── error_handlers.py     # Error handling
    └── helpers.py            # Helper functions
```

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd rwanda_medlink_model
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   cd api
   pip install -r requirements.txt
   ```

4. **Set environment variables** (optional):
   ```bash
   # Windows
   set FLASK_ENV=development
   set FLASK_DEBUG=True
   set FLASK_HOST=0.0.0.0
   set FLASK_PORT=5000
   
   # Linux/Mac
   export FLASK_ENV=development
   export FLASK_DEBUG=True
   export FLASK_HOST=0.0.0.0
   export FLASK_PORT=5000
   ```

## 🚀 Running the Application

### Development Mode

```bash
cd api
python main.py
```

### Production Mode

```bash
# Set production environment
export FLASK_ENV=production  # or set FLASK_ENV=production on Windows
export FLASK_DEBUG=False

# Run the application
python main.py
```

### Using Flask CLI

```bash
export FLASK_APP=main.py
flask run --host=0.0.0.0 --port=5000
```

## 📡 API Endpoints

### Health and Status

- **GET** `/health` - Health check
- **GET** `/info` - Application information
- **GET** `/api/health/status` - Detailed system status

### Model Management

- **POST** `/api/model/load` - Load ML model
- **GET** `/api/model/info` - Get model information and metrics

### Predictions

- **POST** `/api/predict` - Single prediction
- **POST** `/api/predict/batch` - Batch predictions

### Web Interface

- **GET** `/` - Main web interface
- **GET** `/api/web/predict` - Prediction form
- **GET** `/api/web/batch` - Batch prediction form

## 📝 API Usage Examples

### Single Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
  "Pharmacy_Name": "CityMeds 795",
  "Province": "Kigali",
  "Drug_ID": "DICLOFENAC",
  "ATC_Code": "M01AB",
  "Date": "2024-01-01",
  "available_stock": 470,
  "expiration_date": "2024-02-28",
  "stock_entry_timestamp": "2023-12-06",
  "Price_Per_Unit": 33.04,
  "Promotion": 1,
  "Season": "Urugaryi",
  "Disease_Outbreak": 1,
  "Supply_Chain_Delay": "Medium",
  "Effectiveness_Rating": 5,
  "Competitor_Count": 4,
  "Time_On_Market": 47,
  "Population_Density": "high",
  "Income_Level": "higher",
  "Holiday_Week": 1
}'
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
        {
            "Pharmacy_Name": "CityMeds 795",
            "Province": "Kigali",
            "Drug_ID": "DICLOFENAC",
            "Date": "2024-01-01",
            "available_stock": 470,
            "Price_Per_Unit": 33.04
        },
        {
            "Pharmacy_Name": "HealthPlus 234",
            "Province": "Kigali",
            "Drug_ID": "PARACETAMOL",
            "Date": "2024-01-02",
            "available_stock": 250,
            "Price_Per_Unit": 15.50
        },
        {
            "Pharmacy_Name": "MediCare 456",
            "Province": "Northern",
            "Drug_ID": "IBUPROFEN",
            "Date": "2024-01-03",
            "available_stock": 180,
            "Price_Per_Unit": 25.75
        }
    ]
  }'l
```

### Model Information

```bash
curl -X GET http://localhost:5000/api/model/info
```

## 🔧 Configuration

The API supports multiple configuration environments:

- **Development**: Debug mode enabled, verbose logging
- **Production**: Optimized for performance, security enabled
- **Testing**: Configured for unit testing

Configuration is managed through environment variables and the `config.py` module.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Environment mode |
| `FLASK_DEBUG` | `True` | Debug mode |
| `FLASK_HOST` | `0.0.0.0` | Host address |
| `FLASK_PORT` | `5000` | Port number |
| `MODEL_PATH` | `../models/` | Path to ML models |
| `LOG_LEVEL` | `INFO` | Logging level |

## 🔒 Security Features

- **Input Validation**: Comprehensive input validation and sanitization
- **Rate Limiting**: Configurable request rate limiting
- **Security Headers**: OWASP recommended security headers
- **CORS**: Configurable Cross-Origin Resource Sharing
- **Error Handling**: Secure error responses (no sensitive data exposure)

## 📊 Monitoring and Logging

- **Request Timing**: Automatic request performance monitoring
- **Error Tracking**: Comprehensive error logging and tracking
- **Health Checks**: System health monitoring endpoints
- **Performance Metrics**: Memory and CPU usage monitoring (optional)

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-flask pytest-cov

# Run tests
pytest

# Run tests with coverage
pytest --cov=api --cov-report=html
```

## 🔄 Development Workflow

1. **Make changes** to the codebase
2. **Run tests** to ensure functionality
3. **Check code quality** with linting tools
4. **Test the API** using the web interface or curl commands
5. **Review logs** for any issues

### Code Quality Tools

```bash
# Format code
black api/

# Sort imports
isort api/

# Lint code
flake8 api/
```

## 📈 Performance Considerations

- **Model Caching**: ML models are cached in memory
- **Request Timing**: All requests are timed and logged
- **Batch Processing**: Efficient batch prediction support
- **Memory Monitoring**: Optional memory usage tracking
- **Rate Limiting**: Prevents API abuse

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Ensure model files exist in the `models/` directory
2. **Import errors**: Check that all dependencies are installed
3. **Port conflicts**: Change the port number if 5000 is in use
4. **Memory issues**: Monitor memory usage for large batch predictions

### Debug Mode

Enable debug mode for development:

```bash
export FLASK_DEBUG=True
python main.py
```

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add tests for new functionality
3. Update documentation for any API changes
4. Use the provided code quality tools
5. Follow Python PEP 8 style guidelines

## 📄 License

This project is part of the Rwanda MedLink medication demand prediction system.

## 📞 Support

For issues and questions:
- Check the logs in the `logs/` directory
- Review the health check endpoints
- Consult the API documentation
- Check configuration settings
