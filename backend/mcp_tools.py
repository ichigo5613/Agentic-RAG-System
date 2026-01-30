#Agentic RAG System/backend/mcp_tools.py
import requests
from datetime import datetime
import json
from typing import Dict, Optional

class MCPTools:
    """Simple MCP tools implementation"""
    
    def web_search(self, query: str) -> str:
        """Mock web search (in real system, use actual API)"""
        # This is a mock implementation
        # In production, integrate with DuckDuckGo, Serper, or other APIs
        return f"Mock search results for: {query}\n- Result 1: Information about {query}\n- Result 2: Related to {query}"
    
    def calculate(self, expression: str) -> str:
        """Simple calculator"""
        try:
            # Extract numbers and operators
            import re
            # Simple evaluation (be careful with security)
            safe_expr = re.sub(r'[^0-9+\-*/().\s]', '', expression)
            if safe_expr:
                result = eval(safe_expr, {"__builtins__": {}})
                return f"{expression} = {result}"
            else:
                return "Invalid calculation expression"
        except:
            return "Could not calculate"
    
    def get_datetime(self) -> str:
        """Get current date and time"""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    def read_file(self, filepath: str) -> str:
        """Read file content (simplified)"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()[:1000]  # Limit content
        except:
            return f"Could not read file: {filepath}"
    
    def call_api(self, url: str, method: str = "GET", 
                data: Optional[Dict] = None) -> str:
        """Make API call"""
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                return f"Unsupported method: {method}"
            
            return f"Status: {response.status_code}\nResponse: {response.text[:500]}"
        except Exception as e:
            return f"API call failed: {str(e)}"