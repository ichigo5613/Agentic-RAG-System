# backend/utils/error_handler.py
from functools import wraps
from flask import request, jsonify
from typing import List, Optional
import traceback

from backend.utils.logger import logger

def handle_exception(e: Exception):
    """Global exception handler"""
    error_type = type(e).__name__
    error_message = str(e)
    traceback_str = traceback.format_exc()
    
    logger.error(f"Unhandled exception: {error_type}: {error_message}", extra={
        "error_type": error_type,
        "error_message": error_message,
        "traceback": traceback_str,
        "endpoint": request.endpoint if request else "unknown",
        "method": request.method if request else "unknown"
    })
    
    # Return appropriate HTTP status
    if error_type == "ValueError":
        status_code = 400
    elif error_type == "PermissionError":
        status_code = 403
    elif error_type == "FileNotFoundError":
        status_code = 404
    elif error_type == "ConnectionError":
        status_code = 503
    else:
        status_code = 500
    
    response = {
        "error": error_type,
        "message": error_message,
        "request_id": request.headers.get("X-Request-ID", "unknown") if request else "unknown"
    }
    
    # Only include traceback in debug mode
    import os
    if os.getenv("DEBUG", "False").lower() == "true":
        response["traceback"] = traceback_str
    
    return jsonify(response), status_code

def validate_request(required_fields: List[str] = None, optional_fields: List[str] = None):
    """Decorator to validate request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    "error": "Invalid Content-Type",
                    "message": "Request must be JSON"
                }), 400
            
            data = request.json
            
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({
                        "error": "Missing required fields",
                        "missing": missing_fields,
                        "required": required_fields
                    }), 400
            
            if optional_fields:
                invalid_fields = [
                    field for field in data.keys() 
                    if field not in required_fields + optional_fields
                ]
                if invalid_fields:
                    return jsonify({
                        "error": "Invalid fields",
                        "invalid": invalid_fields,
                        "allowed": required_fields + optional_fields
                    }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

class RAGError(Exception):
    """Custom exception for RAG system errors"""
    def __init__(self, message: str, code: str = "RAG_ERROR", details: Optional[dict] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)