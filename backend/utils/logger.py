# backend/utils/logger.py
import logging
import sys
from datetime import datetime
from typing import Dict, Any, List
import json
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat()
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup structured JSON logging"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('logs/agentic_rag.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create main application logger
logger = setup_logger("agentic_rag", "INFO")

def log_agent_step(agent_name: str, step: str, data: Dict[str, Any]):
    """Log agent step with structured data"""
    logger.info(f"Agent Step: {agent_name} - {step}", extra={
        "agent": agent_name,
        "step": step,
        "data": data
    })

def log_query_processing(query: str, results: Dict[str, Any]):
    """Log query processing details"""
    logger.info("Query Processing", extra={
        "query": query,
        "chunks_retrieved": len(results.get("chunks", [])),
        "sources": results.get("sources", []),
        "processing_time": results.get("processing_time", 0)
    })