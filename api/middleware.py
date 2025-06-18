"""
Middleware components for the Rwanda MedLink API.
"""

import time
import logging
from functools import wraps
from flask import request, g, current_app
from typing import Callable, Any

logger = logging.getLogger(__name__)


def request_timing_middleware(app):
    """
    Middleware to track request processing time.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def before_request():
        """Record request start time."""
        g.start_time = time.time()
        g.request_id = f"{int(time.time() * 1000)}"
        logger.info(f"Request {g.request_id} started: {request.method} {request.path}")
    
    @app.after_request
    def after_request(response):
        """Log request completion and timing."""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            logger.info(f"Request {g.request_id} completed: {response.status_code} in {duration:.4f}s")
            response.headers['X-Response-Time'] = f"{duration:.4f}s"
            response.headers['X-Request-ID'] = g.request_id
        return response


def request_logging_middleware(app):
    """
    Middleware for detailed request logging.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def log_request_info():
        """Log detailed request information."""
        logger.debug(f"Request details: {request.method} {request.url}")
        logger.debug(f"Headers: {dict(request.headers)}")
        logger.debug(f"Remote address: {request.remote_addr}")
        logger.debug(f"User agent: {request.headers.get('User-Agent', 'Unknown')}")
        
        # Log request body for POST/PUT requests (be careful with sensitive data)
        if request.method in ['POST', 'PUT'] and request.is_json:
            logger.debug(f"Request body size: {len(request.get_data())} bytes")


def security_headers_middleware(app):
    """
    Middleware to add security headers to responses.
    
    Args:
        app: Flask application instance
    """
    
    @app.after_request
    def add_security_headers(response):
        """Add security headers to all responses."""
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = "default-src 'self'"
        
        # Strict Transport Security (for HTTPS)
        if request.is_secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response


def cors_middleware(app):
    """
    Custom CORS middleware for fine-grained control.
    
    Args:
        app: Flask application instance
    """
    
    @app.after_request
    def handle_cors(response):
        """Handle CORS headers."""
        origin = request.headers.get('Origin')
        
        # Allow specific origins in production
        allowed_origins = app.config.get('ALLOWED_ORIGINS', ['*'])
        
        if origin and (origin in allowed_origins or '*' in allowed_origins):
            response.headers['Access-Control-Allow-Origin'] = origin
        
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Max-Age'] = '3600'
        
        return response


def rate_limiting_middleware(app):
    """
    Simple rate limiting middleware.
    
    Args:
        app: Flask application instance
    """
    # This is a basic implementation - consider using Flask-Limiter for production
    request_counts = {}
    
    @app.before_request
    def check_rate_limit():
        """Check request rate limits."""
        if not app.config.get('ENABLE_RATE_LIMITING', False):
            return
        
        client_ip = request.remote_addr
        current_time = int(time.time())
        window_start = current_time - 60  # 1-minute window
        
        # Clean old entries
        request_counts[client_ip] = [
            timestamp for timestamp in request_counts.get(client_ip, [])
            if timestamp > window_start
        ]
        
        # Check if limit exceeded
        max_requests = app.config.get('RATE_LIMIT_PER_MINUTE', 60)
        if len(request_counts.get(client_ip, [])) >= max_requests:
            from flask import abort
            abort(429)  # Too Many Requests
        
        # Add current request
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        request_counts[client_ip].append(current_time)


def error_tracking_middleware(app):
    """
    Middleware for error tracking and monitoring.
    
    Args:
        app: Flask application instance
    """
    
    @app.teardown_request
    def track_errors(exception):
        """Track and log errors."""
        if exception:
            logger.error(f"Request failed with exception: {str(exception)}")
            logger.error(f"Request URL: {request.url}")
            logger.error(f"Request method: {request.method}")
            logger.error(f"User agent: {request.headers.get('User-Agent', 'Unknown')}")
            
            # In production, you might want to send this to an external service
            # like Sentry, DataDog, etc.


def request_validation_middleware(app):
    """
    Middleware for request validation.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def validate_request():
        """Validate incoming requests."""
        # Content-Type validation for JSON endpoints
        if request.method in ['POST', 'PUT'] and request.path.startswith('/api/'):
            content_type = request.headers.get('Content-Type', '')
            if not (content_type.startswith('application/json') or request.is_json):
                from flask import abort
                abort(400, description="Content-Type must be application/json")
        
        # Request size validation
        max_content_length = app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)  # 16MB
        if request.content_length and max_content_length and request.content_length > max_content_length:
            from flask import abort
            abort(413, description="Request entity too large")


def performance_monitoring_middleware(app):
    """
    Middleware for performance monitoring.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def start_performance_monitoring():
        """Start performance monitoring."""
        g.perf_start = time.perf_counter()
        g.memory_start = None
        
        # Memory monitoring (optional, requires psutil)
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            g.memory_start = process.memory_info().rss
        except ImportError:
            pass
    
    @app.after_request
    def end_performance_monitoring(response):
        """End performance monitoring and log metrics."""
        if hasattr(g, 'perf_start'):
            duration = time.perf_counter() - g.perf_start
            
            # Log slow requests
            slow_request_threshold = app.config.get('SLOW_REQUEST_THRESHOLD', 1.0)
            if duration > slow_request_threshold:
                logger.warning(f"Slow request detected: {request.path} took {duration:.4f}s")
            
            # Memory usage monitoring
            if hasattr(g, 'memory_start') and g.memory_start:
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    memory_end = process.memory_info().rss
                    memory_diff = memory_end - g.memory_start
                    
                    if memory_diff > 10 * 1024 * 1024:  # 10MB increase
                        logger.warning(f"High memory usage detected: {memory_diff / 1024 / 1024:.2f}MB increase")
                except ImportError:
                    pass
        
        return response


def register_middleware(app):
    """
    Register all middleware components.
    
    Args:
        app: Flask application instance
    """
    # Core middleware
    request_timing_middleware(app)
    security_headers_middleware(app)
    error_tracking_middleware(app)
    
    # Optional middleware based on configuration
    if app.config.get('ENABLE_REQUEST_LOGGING', False):
        request_logging_middleware(app)
    
    if app.config.get('ENABLE_RATE_LIMITING', False):
        rate_limiting_middleware(app)
    
    if app.config.get('ENABLE_PERFORMANCE_MONITORING', False):
        performance_monitoring_middleware(app)
    
    if app.config.get('ENABLE_REQUEST_VALIDATION', True):
        request_validation_middleware(app)
    
    logger.info("Middleware components registered successfully")


# Decorator for function-level middleware
def with_timing(func: Callable) -> Callable:
    """
    Decorator to add timing to specific functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.debug(f"Function {func.__name__} executed in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
            raise
    return wrapper


def with_caching(cache_key: str = None, ttl: int = 300):
    """
    Decorator to add caching to specific functions.
    
    Args:
        cache_key: Cache key (if None, generates from function name and args)
        ttl: Time to live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Simple in-memory cache implementation
            # In production, consider using Redis or Memcached
            cache = getattr(current_app, '_simple_cache', {})
            if not hasattr(current_app, '_simple_cache'):
                current_app._simple_cache = {}
                cache = current_app._simple_cache
            
            # Generate cache key
            key = cache_key or f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            if key in cache:
                cached_data, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            logger.debug(f"Cache miss for {func.__name__}, result cached")
        
            return result
        return wrapper
    return decorator
