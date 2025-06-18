"""
Main entry point for the Rwanda MedLink API.

This module provides the main entry point for running the Flask application
using the application factory pattern. It supports both Gunicorn and CLI usage.
"""

import os
import sys
import logging
from flask import Flask

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_factory import create_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# Create the app at the module level so Gunicorn can find it
try:
    config_name = os.getenv('FLASK_ENV', 'development')
    app = create_app(config_name)
except Exception as e:
    logging.error(f"Failed to create the Flask app: {e}", exc_info=True)
    app = None  # Prevent Gunicorn from starting with a broken app


def main():
    """Main entry point for the application when run via CLI."""
    try:
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

        logging.info("Starting Rwanda MedLink API")
        logging.info(f"Configuration: {config_name}")
        logging.info(f"Host: {host}")
        logging.info(f"Port: {port}")
        logging.info(f"Debug mode: {debug}")

        if app is None:
            raise RuntimeError("Flask app could not be created.")

        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logging.error(f"Failed to start the application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
