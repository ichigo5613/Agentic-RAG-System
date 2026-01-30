#Agentic RAG System/backend/milvus_client.py
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from config import Config
import uuid

class MilvusClient:
    def __init__(self):
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT
        )
        
        # Initialize embedding model
        print(f"🔧 Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Create or get collection
        self.collection = self._get_or_create_collection()
        print(f"✅ Connected to Milvus collection: {Config.COLLECTION_NAME}")
    
    def _get_or_create_collection(self):
        """Create collection if it doesn't exist"""
        if Config.COLLECTION_NAME in Collection.list():
            collection = Collection(Config.COLLECTION_NAME)
            collection.load()
            return collection
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, description="Agentic RAG documents")
        collection = Collection(Config.COLLECTION_NAME, schema)
        
        # Create index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("embedding", index_params)
        
        return collection
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def add_documents(self, texts: List[str], metadata_list: List[Dict]):
        """Add documents to collection"""
        if not texts:
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Prepare data
        data = [
            ids,
            embeddings,
            texts,
            metadata_list
        ]
        
        # Insert into collection
        self.collection.insert(data)
        self.collection.flush()
        
        print(f"📚 Added {len(texts)} documents to Milvus")
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 50}
        }
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        
        # Format results
        documents = []
        scores = []
        metadata = []
        
        for hits in results:
            for hit in hits:
                documents.append(hit.entity.get("text"))
                scores.append(hit.score)
                metadata.append(hit.entity.get("metadata"))
        
        return {
            "documents": documents,
            "scores": scores,
            "metadata": metadata
        }
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Hybrid search (vector + keyword)"""
        # This is a simplified version
        # In production, you'd want to use Milvus's hybrid search capabilities
        return self.search(query, top_k)
    
    def count_documents(self) -> int:
        """Count documents in collection"""
        return self.collection.num_entities
    
    def list_documents(self) -> List[str]:
        """List all unique document sources"""
        try:
            # Get distinct sources from metadata
            results = self.collection.query(
                expr="",
                output_fields=["metadata"],
                limit=1000
            )
            
            sources = set()
            for result in results:
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown")
                sources.add(source)
            
            return list(sources)
        except:
            return []
    
    def clear_collection(self):
        """Clear all documents from collection"""
        self.collection.drop()
        print("🗑️ Cleared Milvus collection")
        # Recreate collection
        self.collection = self._get_or_create_collection()
    
    def test_connection(self) -> bool:
        """Test Milvus connection"""
        try:
            return self.collection.is_empty is not None
        except:
            return False