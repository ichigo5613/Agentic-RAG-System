#Agentic RAG System/backend/utils.py
import os
import json
import logging
from typing import Any, Dict
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_json(data: Dict, filename: str):
    """Save data as JSON file"""
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filename: str) -> Dict:
    """Load data from JSON file"""
    filepath = os.path.join('data', filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def format_context(chunks: list, sources: list) -> str:
    """Format context for display"""
    formatted = []
    for i, chunk in enumerate(chunks):
        source = sources[i] if i < len(sources) else "Unknown"
        formatted.append(f"📄 Source: {source}\n{chunk[:200]}...")
    return "\n\n".join(formatted)

def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_file(file_path: str, allowed_extensions: set) -> bool:
    """Validate file extension"""
    ext = os.path.splitext(file_path)[1].lower()[1:]  # Remove dot
    return ext in allowed_extensions