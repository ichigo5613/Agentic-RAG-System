# backend/core/milvus_store.py
# from typing import List, Dict, Any, Optional, Tuple
# import uuid
# from datetime import datetime
# import json
# import numpy as np

# from pymilvus import (
#     connections,
#     FieldSchema, CollectionSchema, DataType,
#     Collection, utility
# )
# from langchain.vectorstores import VectorStore
# from langchain.docstore.document import Document as LangchainDocument
# from langchain_huggingface import HuggingFaceEmbeddings

# from backend.config import config
# from backend.models.llm_client import LLMClient
# from backend.utils.logger import logger

# backend/core/milvus_store.py - FIXED IMPORTS
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
import json
import numpy as np

from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

# Updated imports for LangChain compatibility
try:
    from langchain.vectorstores.base import VectorStore
    from langchain_core.documents import Document as LangchainDocument
except ImportError:
    # Fallback for older versions
    try:
        from langchain.schema.vectorstore import VectorStore
        from langchain.schema.document import Document as LangchainDocument
    except ImportError:
        # Minimal implementation without LangChain
        class VectorStore:
            pass
        class LangchainDocument:
            def __init__(self, page_content, metadata):
                self.page_content = page_content
                self.metadata = metadata

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import config
from backend.models.llm_client import LLMClient
from backend.utils.logger import logger

class MilvusVectorStore(VectorStore):
    """LangChain compatible Milvus vector store"""
    
    def __init__(self):
        # Connect to Milvus
        self._connect_to_milvus()
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM client for HyDE
        self.llm_client = LLMClient()
        
        # Initialize or get collection
        self.collection = self._initialize_collection()
        
        logger.info(f"✅ Initialized Milvus Vector Store with collection: {config.MILVUS_COLLECTION_NAME}")
    
    def _connect_to_milvus(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                user=config.MILVUS_USER if config.MILVUS_USER else None,
                password=config.MILVUS_PASSWORD if config.MILVUS_PASSWORD else None,
                db_name=config.MILVUS_DB_NAME
            )
            logger.info(f"✅ Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus: {str(e)}")
            raise
    
    def _initialize_collection(self) -> Collection:
        """Initialize or get existing collection"""
        # Check if collection exists
        if utility.has_collection(config.MILVUS_COLLECTION_NAME):
            logger.info(f"📚 Collection '{config.MILVUS_COLLECTION_NAME}' already exists")
            collection = Collection(config.MILVUS_COLLECTION_NAME)
            
            # Load collection for searching (new Milvus API)
            try:
                collection.load()
                logger.info(f"📂 Loaded collection '{config.MILVUS_COLLECTION_NAME}'")
            except Exception as e:
                logger.warning(f"⚠️ Could not load collection: {e}")
            
            return collection
        
        # Create new collection
        logger.info(f"🆕 Creating new collection: {config.MILVUS_COLLECTION_NAME}")
        
        # Define schema with LangChain compatibility
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIMENSION),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(fields=fields, description="Agentic RAG Document Chunks")
        collection = Collection(name=config.MILVUS_COLLECTION_NAME, schema=schema)
        
        # Create index
        index_params = {
            "metric_type": config.MILVUS_METRIC_TYPE,
            "index_type": config.MILVUS_INDEX_TYPE,
            "params": {"nlist": config.MILVUS_INDEX_NLIST}
        }
        
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"📊 Created index: {config.MILVUS_INDEX_TYPE}")
        
        # Load collection
        collection.load()
        logger.info(f"✅ Collection '{config.MILVUS_COLLECTION_NAME}' created and loaded")
        
        return collection
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[str]:
        """LangChain compatible method to add texts"""
        return self.add_documents(texts, metadatas or [])
    
    def add_documents(self, texts: List[str], metadata_list: List[Dict]) -> List[str]:
        """Add documents to Milvus"""
        if not texts:
            return []
        
        logger.info(f"🔄 Generating embeddings for {len(texts)} documents...")
        
        # Generate embeddings
        try:
            embeddings = self.embedding_model.embed_documents(texts)
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {str(e)}")
            raise
        
        # Prepare data for insertion
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Create data in correct format for Milvus
        data = [
            ids,  # id
            embeddings,  # embedding
            texts,  # text
            [json.dumps(meta) for meta in metadata_list],  # metadata
            [meta.get("source", "Unknown") for meta in metadata_list],  # source
            [meta.get("chunk_id", i) for i, meta in enumerate(metadata_list)],  # chunk_id
            [datetime.utcnow().isoformat() for _ in range(len(texts))]  # created_at
        ]
        
        # Insert data
        try:
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"✅ Successfully inserted {len(texts)} documents into Milvus")
            return ids
        except Exception as e:
            logger.error(f"❌ Failed to insert documents into Milvus: {str(e)}")
            # Try alternative insertion method
            return self._insert_alternative_method(texts, embeddings, metadata_list, ids)
    
    def _insert_alternative_method(self, texts, embeddings, metadata_list, ids):
        """Alternative insertion method if primary fails"""
        try:
            # Insert one by one
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata_list)):
                data = [
                    [ids[i]],  # id
                    [embedding],  # embedding
                    [text],  # text
                    [json.dumps(meta)],  # metadata
                    [meta.get("source", "Unknown")],  # source
                    [meta.get("chunk_id", i)],  # chunk_id
                    [datetime.utcnow().isoformat()]  # created_at
                ]
                self.collection.insert(data)
            
            self.collection.flush()
            logger.info(f"✅ Inserted {len(texts)} documents using alternative method")
            return ids
        except Exception as e:
            logger.error(f"❌ Alternative insertion also failed: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List[LangchainDocument]:
        """LangChain compatible similarity search"""
        results = self.search(query, top_k=k, filters=filter)
        
        documents = []
        for doc, meta in zip(results["documents"], results["metadata"]):
            # Parse metadata from JSON string
            if isinstance(meta, str):
                try:
                    meta_dict = json.loads(meta)
                except:
                    meta_dict = {"source": "Unknown"}
            else:
                meta_dict = meta
            
            documents.append(LangchainDocument(
                page_content=doc,
                metadata=meta_dict
            ))
        
        return documents
    
    def search(self, query: str, top_k: int = None, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Basic similarity search"""
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Prepare search parameters
            search_params = {
                "metric_type": config.MILVUS_METRIC_TYPE,
                "params": {"nprobe": config.MILVUS_SEARCH_NPROBE}
            }
            
            # Build filter expression
            expr = self._build_filter_expression(filters) if filters else None
            
            # Execute search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "metadata", "source", "chunk_id"]
            )
            
            # Process results
            documents = []
            similarities = []
            metadata_list = []
            
            for hits in results:
                for hit in hits:
                    documents.append(hit.entity.get("text"))
                    similarities.append(hit.score)
                    metadata_list.append(hit.entity.get("metadata", "{}"))
            
            return {
                "documents": documents,
                "similarities": similarities,
                "metadata": metadata_list
            }
            
        except Exception as e:
            logger.error(f"❌ Milvus search failed: {str(e)}")
            return {"documents": [], "similarities": [], "metadata": []}
    
    def _build_filter_expression(self, filters: Dict) -> str:
        """Build Milvus filter expression"""
        expressions = []
        for key, value in filters.items():
            if key == "source":
                expressions.append(f'source == "{value}"')
            elif key == "chunk_id":
                expressions.append(f'chunk_id == {value}')
        
        return " and ".join(expressions) if expressions else ""
    
    # LangChain required methods
    def from_documents(self, documents: List[LangchainDocument], **kwargs):
        """Create from LangChain documents"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)
    
    def from_texts(self, texts: List[str], metadatas: List[Dict], **kwargs):
        """Create from texts"""
        return self.add_texts(texts, metadatas, **kwargs)
    
    def count_documents(self) -> int:
        """Count total documents"""
        try:
            return self.collection.num_entities
        except:
            return 0
    
    def list_documents(self) -> List[str]:
        """List unique document sources"""
        try:
            # Get distinct sources
            results = self.collection.query(
                expr="",
                output_fields=["source"],
                limit=10000
            )
            
            sources = set()
            for item in results:
                if item.get("source"):
                    sources.add(item["source"])
            
            return list(sources)
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def clear_collection(self):
        """Clear the entire collection"""
        try:
            utility.drop_collection(config.MILVUS_COLLECTION_NAME)
            logger.info(f"🗑️ Dropped collection: {config.MILVUS_COLLECTION_NAME}")
            
            # Reinitialize collection
            self.collection = self._initialize_collection()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to Milvus"""
        try:
            # Simple connection test
            connections.get_connection_addr("default")
            
            # Check collection exists and has entities
            count = self.count_documents()
            logger.info(f"✅ Milvus connection test successful. Documents: {count}")
            return True
        except Exception as e:
            logger.error(f"❌ Milvus connection test failed: {str(e)}")
            return False