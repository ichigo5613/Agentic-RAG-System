#Agentic RAG System/backend/ollama_client.py
import requests
import json
from typing import List, Dict, Optional
from config import Config

class OllamaClient:
    def __init__(self):
        self.host = Config.OLLAMA_HOST
        self.model = Config.OLLAMA_MODEL
        self.timeout = 120  # 2 minutes
    
    def generate(self, prompt: str, system: Optional[str] = None, 
                 temperature: float = 0.1) -> str:
        """Generate text using Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1000
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"
    
    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Chat completion using Ollama"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Error: {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            return f"Connection error: {str(e)}"
    
    def extract_json(self, text: str) -> Dict:
        """Extract JSON from LLM response"""
        try:
            # Find JSON in text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            pass
        return {}
    
    def test_connection(self) -> bool:
        """Test Ollama connection"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if self.model in models:
                    print(f"✅ Ollama connected with model: {self.model}")
                    return True
                else:
                    print(f"⚠️ Model {self.model} not found in Ollama")
                    print(f"   Available models: {models}")
                    return False
        except:
            print("❌ Cannot connect to Ollama")
            return False