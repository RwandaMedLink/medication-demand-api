"""
Main entry point for the Rwanda MedLink API.

This module provides the main entry point for running the Flask application
using the application factory pattern.
"""

import os
import sys
import logging
from flask import Flask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_factory import create_app


def main():
    """Main entry point for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )

    try:
        config_name = os.getenv('FLASK_ENV', 'development')
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 5000))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

        logging.info(f"Starting Rwanda MedLink API")
        logging.info(f"Configuration: {config_name}")
        logging.info(f"Host: {host}")
        logging.info(f"Port: {port}")
        logging.info(f"Debug mode: {debug}")

        app = create_app(config_name)

        # Run the application
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logging.error(f"Failed to start the application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
