# backend/config.py - FIXED VERSION
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Flask Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "agentic-rag-secret-key-2024")
    MAX_CONTENT_LENGTH: int = 100 * 1024 * 1024  # 100MB
    
    # Ollama Settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma2:2b")
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_MAX_TOKENS: int = 2000
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIR: str = "./storage/chroma_db"
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "agentic_rag_docs")
    
    # File Upload Settings
    UPLOAD_FOLDER: str = "./storage/uploads"
    # FIXED: Using default_factory for mutable set
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {'pdf', 'docx', 'txt', 'xlsx', 'xls', 'pptx', 'md'})
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    TOP_K_RERANK: int = 3
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Agent Settings
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_TIMEOUT: int = 60
    ENABLE_QUERY_DECOMPOSITION: bool = True
    ENABLE_HYDE: bool = True
    ENABLE_RERANKING: bool = True
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = "./logs/agentic_rag.log"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.CHROMA_PERSIST_DIR,
            "./logs",
            "./storage",
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")

# Initialize config
config = Config()
config.create_directories()