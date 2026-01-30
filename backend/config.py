#Agentic RAG System/backend/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Ollama
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
    
    # Milvus
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agentic_rag")
    
    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2
    
    # File Upload
    UPLOAD_FOLDER = "./storage/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'xlsx', 'xls', 'pptx'}
    
    # Agent Settings
    MAX_CONTEXT_CHUNKS = 5
    AGENT_TEMPERATURE = 0.1
    
    # Create directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("./storage/processed", exist_ok=True)