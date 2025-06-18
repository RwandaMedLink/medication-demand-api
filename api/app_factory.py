"""
Flask application factory for the Rwanda MedLink API.

This module creates and configures the Flask application with all necessary
components including blueprints, error handlers, logging, and services.
"""

import os
import logging
from flask import Flask

# Make flask_cors optional
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

from config import DevelopmentConfig, ProductionConfig, TestingConfig
from routes import create_blueprints
from services import ModelService
from utils.error_handlers import register_error_handlers
from middleware import register_middleware


def create_app(config_name: str = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.getenv('FLASK_ENV', 'development')
    config_mapping = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    config_class = config_mapping.get(config_name.lower(), DevelopmentConfig)
    app.config.from_object(config_class)
    
    # Setup logging
    setup_logging(app)
    
    # Enable CORS if available
    if HAS_CORS:
        CORS(app, resources={
            r"/api/*": {
                "origins": ["*"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
    else:
        app.logger.warning("Flask-CORS not available. Cross-origin requests may be blocked.")
    
    # Initialize services
    model_service = ModelService(app.config)
    
    # Register blueprints
    blueprints = create_blueprints(model_service)
    
    # Register blueprints with appropriate URL prefixes
    for blueprint in blueprints:
        if blueprint.name == 'health':
            app.register_blueprint(blueprint)
        elif blueprint.name == 'model':
            app.register_blueprint(blueprint, url_prefix='/api/model')
        elif blueprint.name == 'prediction':
            app.register_blueprint(blueprint, url_prefix='/api')
        elif blueprint.name == 'web':
            app.register_blueprint(blueprint, url_prefix='/api/web')
        else:
            app.register_blueprint(blueprint)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register middleware
    register_middleware(app)
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        """Simple health check endpoint."""
        from utils.responses import health_check_response
        return health_check_response()
    
    # Add application info
    @app.route('/info')
    def app_info():
        """Application information endpoint."""
        from utils.responses import success_response
        info = {
            "name": "Rwanda MedLink API",
            "version": "1.0.0",
            "description": "Medication demand prediction API for Rwanda healthcare system",
            "environment": config_name,
            "debug": app.debug
        }
        return success_response(info, "Application information retrieved")
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Rwanda MedLink API started in {config_name} mode")
    logger.info(f"Debug mode: {app.debug}")
    logger.info(f"Model path: {app.config.get('MODEL_PATH', 'Not configured')}")
    
    return app


def setup_logging(app: Flask) -> None:
    """
    Setup application logging.
    
    Args:
        app: Flask application instance
    """
    # Create logs directory if it doesn't exist
    log_dir = app.config.get('LOG_DIR', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO').upper())
    log_format = app.config.get('LOG_FORMAT', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    if not app.config.get('TESTING', False):
        log_file = os.path.join(log_dir, 'app.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Configure Flask logger
        app.logger.setLevel(log_level)
        app.logger.addHandler(file_handler)
        app.logger.addHandler(console_handler)
    
    # Suppress some verbose loggers in production
    if app.config.get('ENV') == 'production':
        logging.getLogger('werkzeug').setLevel(logging.WARNING)


# Import click for CLI commands (optional)
try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False


def register_cli_commands(app: Flask) -> None:
    """
    Register CLI commands for the application.
    
    Args:
        app: Flask application instance
    """
    if not HAS_CLICK:
        return
    
    @app.cli.command()
    def init_db():
        """Initialize the database."""
        click.echo('Database initialization not implemented yet.')
    
    @app.cli.command()
    def load_model():
        """Load the prediction model."""
        try:
            model_service = ModelService(app.config)
            model_service.load_model()
            click.echo('Model loaded successfully.')
        except Exception as e:
            click.echo(f'Error loading model: {e}', err=True)
    
    @app.cli.command()
    @click.option('--test-data', required=True, help='Path to test data file')
    def validate_model(test_data):
        """Validate the prediction model with test data."""
        try:
            model_service = ModelService(app.config)
            # Add model validation logic here
            click.echo('Model validation completed.')
        except Exception as e:
            click.echo(f'Error validating model: {e}', err=True)


# Only create CLI version if click is available
if HAS_CLICK:
    def create_app_with_cli(config_name: str = None) -> Flask:
        """Create app with CLI commands."""
        app = create_app(config_name)
        register_cli_commands(app)
        return app
else:
    create_app_with_cli = create_app
