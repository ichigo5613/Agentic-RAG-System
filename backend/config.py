# backend/config.py - UPDATED FOR MILVUS
import os
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Any
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
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # bge-small-en-v1.5 = 384 dimensions
    
    # Milvus Vector Database Settings
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER: str = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME", "default")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "agentic_rag_docs")
    
    # Milvus Index Settings
    MILVUS_INDEX_TYPE: str = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
    MILVUS_METRIC_TYPE: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    MILVUS_INDEX_NLIST: int = int(os.getenv("MILVUS_INDEX_NLIST", "128"))
    MILVUS_SEARCH_NPROBE: int = int(os.getenv("MILVUS_SEARCH_NPROBE", "10"))
    
    # File Upload Settings
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./storage/uploads")
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {'pdf', 'docx', 'txt', 'xlsx', 'xls', 'pptx', 'md'})
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    TOP_K_RERANK: int = int(os.getenv("TOP_K_RERANK", "3"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Agent Settings
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "60"))
    ENABLE_QUERY_DECOMPOSITION: bool = os.getenv("ENABLE_QUERY_DECOMPOSITION", "True").lower() == "true"
    ENABLE_HYDE: bool = os.getenv("ENABLE_HYDE", "True").lower() == "true"
    ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "True").lower() == "true"
    ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "True").lower() == "true"
    
    # Cache Settings
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    MAX_CACHE_SIZE: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/agentic_rag.log")
    
    @property
    def milvus_connection_params(self) -> Dict[str, Any]:
        """Get Milvus connection parameters"""
        params = {
            "host": self.MILVUS_HOST,
            "port": self.MILVUS_PORT,
            "db_name": self.MILVUS_DB_NAME
        }
        
        if self.MILVUS_USER:
            params["user"] = self.MILVUS_USER
        if self.MILVUS_PASSWORD:
            params["password"] = self.MILVUS_PASSWORD
            
        return params
    
    @property
    def milvus_index_params(self) -> Dict[str, Any]:
        """Get Milvus index parameters"""
        return {
            "metric_type": self.MILVUS_METRIC_TYPE,
            "index_type": self.MILVUS_INDEX_TYPE,
            "params": {"nlist": self.MILVUS_INDEX_NLIST}
        }
    
    @property
    def milvus_search_params(self) -> Dict[str, Any]:
        """Get Milvus search parameters"""
        return {
            "metric_type": self.MILVUS_METRIC_TYPE,
            "params": {"nprobe": self.MILVUS_SEARCH_NPROBE}
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.UPLOAD_FOLDER,
            "./storage",
            "./logs",
            "./storage/uploads",
            "./storage/temp"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("🤖 Agentic RAG System Configuration")
        print("="*60)
        
        print(f"\n🔧 Backend Settings:")
        print(f"   • Flask Debug: {self.DEBUG}")
        print(f"   • Max Upload: {self.MAX_CONTENT_LENGTH/(1024*1024)} MB")
        
        print(f"\n🤖 LLM Settings:")
        print(f"   • Ollama Host: {self.OLLAMA_HOST}")
        print(f"   • Model: {self.OLLAMA_MODEL}")
        print(f"   • Temperature: {self.OLLAMA_TEMPERATURE}")
        
        print(f"\n🗄️  Vector Database (Milvus):")
        print(f"   • Host: {self.MILVUS_HOST}:{self.MILVUS_PORT}")
        print(f"   • Collection: {self.MILVUS_COLLECTION_NAME}")
        print(f"   • DB Name: {self.MILVUS_DB_NAME}")
        print(f"   • Index: {self.MILVUS_INDEX_TYPE} ({self.MILVUS_METRIC_TYPE})")
        
        print(f"\n🔤 Embedding Model:")
        print(f"   • Model: {self.EMBEDDING_MODEL}")
        print(f"   • Dimension: {self.EMBEDDING_DIMENSION}")
        
        print(f"\n📄 Document Processing:")
        print(f"   • Chunk Size: {self.CHUNK_SIZE}")
        print(f"   • Chunk Overlap: {self.CHUNK_OVERLAP}")
        print(f"   • Allowed Extensions: {', '.join(self.ALLOWED_EXTENSIONS)}")
        
        print(f"\n🔍 Retrieval Settings:")
        print(f"   • Top K Retrieval: {self.TOP_K_RETRIEVAL}")
        print(f"   • Top K Rerank: {self.TOP_K_RERANK}")
        print(f"   • Similarity Threshold: {self.SIMILARITY_THRESHOLD}")
        print(f"   • HyDE Enabled: {self.ENABLE_HYDE}")
        print(f"   • Multi-Query Enabled: {self.ENABLE_MULTI_QUERY}")
        
        print(f"\n🤖 Agent Settings:")
        print(f"   • Query Decomposition: {self.ENABLE_QUERY_DECOMPOSITION}")
        print(f"   • Max Iterations: {self.AGENT_MAX_ITERATIONS}")
        
        print(f"\n💾 Cache Settings:")
        print(f"   • TTL: {self.CACHE_TTL}s")
        print(f"   • Max Size: {self.MAX_CACHE_SIZE}")
        
        print(f"\n📊 Logging:")
        print(f"   • Level: {self.LOG_LEVEL}")
        print(f"   • File: {self.LOG_FILE}")
        print("="*60 + "\n")

# Initialize config
config = Config()
config.create_directories()
config.print_config_summary()