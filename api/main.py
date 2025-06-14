"""
Main entry point for the Rwanda MedLink API.

This module provides the main entry point for running the Flask application
using the application factory pattern.
"""

import os
import sys
from flask import Flask

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_factory import create_app


def main():
    """Main entry point for the application."""
    # Get configuration from environment
    config_name = os.getenv('FLASK_ENV', 'development')
    
    # Create the Flask application
    app = create_app(config_name)
    
    # Get host and port from environment or use defaults
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Run the application
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
