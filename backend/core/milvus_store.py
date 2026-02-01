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
        
        # Get schema fields to understand expected structure
        schema_fields = self.collection.schema.fields
        field_names = [field.name for field in schema_fields]
        logger.info(f"📋 Milvus schema fields: {field_names}")
        
        # Prepare base data with common fields
        data = [
            ids,  # id field
            embeddings,  # embedding field
            texts,  # text field
            [json.dumps(meta) for meta in metadata_list],  # metadata field
            [meta.get("source", "Unknown") for meta in metadata_list],  # source field
            [meta.get("chunk_id", i) for i, meta in enumerate(metadata_list)],  # chunk_id field
            [datetime.utcnow().isoformat() for _ in range(len(texts))]  # created_at field
        ]
        
        # Check if we need additional fields based on schema
        if len(field_names) > 7:
            logger.warning(f"⚠️ Schema expects {len(field_names)} fields, but we have 7. Adding placeholder fields.")
            
            # Add placeholder data for missing fields
            for i in range(7, len(field_names)):
                field_name = field_names[i]
                logger.info(f"➕ Adding placeholder for field: {field_name}")
                
                # Add default values based on field type
                field_type = schema_fields[i].dtype
                if field_type in [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]:
                    data.append([0] * len(texts))
                elif field_type in [DataType.FLOAT, DataType.DOUBLE]:
                    data.append([0.0] * len(texts))
                elif field_type in [DataType.BOOL]:
                    data.append([True] * len(texts))
                else:
                    data.append([""] * len(texts))
        
        # Insert data
        try:
            logger.info(f"📊 Inserting {len(texts)} documents with {len(data)} fields")
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"✅ Successfully inserted {len(texts)} documents into Milvus")
            return ids
        except Exception as e:
            logger.error(f"❌ Failed to insert documents into Milvus: {str(e)}")
            
            # Debug: Print actual schema
            logger.error(f"📋 Schema details: {[(f.name, f.dtype) for f in schema_fields]}")
            logger.error(f"📋 Data structure: {[type(d).__name__ for d in data]}")
            
            # Try alternative insertion method
            return self._insert_alternative_method(texts, embeddings, metadata_list, ids, schema_fields)

    def _insert_alternative_method(self, texts, embeddings, metadata_list, ids, schema_fields):
        """Alternative insertion method if primary fails"""
        try:
            field_names = [field.name for field in schema_fields]
            
            # Insert one by one
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata_list)):
                # Prepare single document data
                base_data = [
                    [ids[i]],  # id
                    [embedding],  # embedding
                    [text],  # text
                    [json.dumps(meta)],  # metadata
                    [meta.get("source", "Unknown")],  # source
                    [meta.get("chunk_id", i)],  # chunk_id
                    [datetime.utcnow().isoformat()]  # created_at
                ]
                
                # Add placeholder fields if needed
                if len(field_names) > 7:
                    for j in range(7, len(field_names)):
                        field_name = field_names[j]
                        field_type = schema_fields[j].dtype
                        
                        if field_type in [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]:
                            base_data.append([0])
                        elif field_type in [DataType.FLOAT, DataType.DOUBLE]:
                            base_data.append([0.0])
                        elif field_type in [DataType.BOOL]:
                            base_data.append([True])
                        else:
                            base_data.append([""])
                
                self.collection.insert(base_data)
            
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

    def advanced_search(self, 
                    query: str, 
                    use_hyde: bool = True,
                    use_multi_query: bool = True) -> Dict[str, Any]:
        """Advanced search with multiple retrieval strategies"""
        all_results = []
        
        # 1. Basic similarity search
        basic_results = self.search(query)
        if basic_results["documents"]:
            all_results.append(basic_results)
        
        # 2. HyDE (Hypothetical Document Embeddings)
        if use_hyde:
            hyde_results = self._hyde_search(query)
            if hyde_results["documents"]:
                all_results.append(hyde_results)
        
        # 3. Multi-query retrieval
        if use_multi_query and len(all_results) > 0:
            multi_query_results = self._multi_query_search(query)
            if multi_query_results["documents"]:
                all_results.append(multi_query_results)
        
        # Combine and deduplicate results
        combined_results = self._combine_results(all_results)
        
        # Rerank if enabled
        if config.ENABLE_RERANKING and combined_results["documents"]:
            combined_results = self._rerank_results(query, combined_results)
        
        logger.info(f"Advanced search retrieved {len(combined_results['documents'])} unique chunks")
        return combined_results

    def _hyde_search(self, query: str) -> Dict[str, Any]:
        """Hypothetical Document Embeddings search"""
        try:
            # Generate hypothetical answer
            hyde_prompt = f"""
            Based on the following query, write a hypothetical answer 
            that would be found in a relevant document:
            
            Query: {query}
            
            Hypothetical Answer:
            """
            
            hypothetical_answer = self.llm_client.generate(hyde_prompt)
            
            # Search with hypothetical answer
            return self.search(hypothetical_answer)
        except Exception as e:
            logger.warning(f"HyDE search failed: {str(e)}")
            return {"documents": [], "similarities": [], "metadata": []}

    def _multi_query_search(self, query: str) -> Dict[str, Any]:
        """Generate multiple queries from the original query"""
        try:
            multi_query_prompt = f"""
            Given the following user query, generate 3 different ways 
            this query could be expressed for document search:
            
            Original Query: {query}
            
            Generate 3 alternative search queries (one per line):
            1.
            2.
            3.
            """
            
            response = self.llm_client.generate(multi_query_prompt)
            
            # Parse alternative queries
            alternative_queries = [
                line.strip()[3:] if line.strip().startswith(("1.", "2.", "3.")) 
                else line.strip()
                for line in response.split('\n')
                if line.strip()
            ]
            
            # Add original query
            alternative_queries = [query] + alternative_queries[:3]
            
            # Search with each query and combine
            all_documents = []
            all_similarities = []
            all_metadata = []
            
            for alt_query in alternative_queries:
                results = self.search(alt_query)
                all_documents.extend(results["documents"])
                all_similarities.extend(results["similarities"])
                all_metadata.extend(results["metadata"])
            
            return {
                "documents": all_documents,
                "similarities": all_similarities,
                "metadata": all_metadata
            }
        except Exception as e:
            logger.warning(f"Multi-query search failed: {str(e)}")
            return {"documents": [], "similarities": [], "metadata": []}

    def _combine_results(self, results_list: List[Dict]) -> Dict[str, Any]:
        """Combine and deduplicate results from multiple searches"""
        seen_documents = set()
        combined_documents = []
        combined_similarities = []
        combined_metadata = []
        
        for results in results_list:
            for doc, sim, meta in zip(
                results["documents"], 
                results["similarities"], 
                results["metadata"]
            ):
                if doc not in seen_documents:
                    seen_documents.add(doc)
                    combined_documents.append(doc)
                    combined_similarities.append(sim)
                    combined_metadata.append(meta)
        
        return {
            "documents": combined_documents,
            "similarities": combined_similarities,
            "metadata": combined_metadata
        }

    def _rerank_results(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Simple reranking based on relevance"""
        try:
            # Simple reranking: prioritize chunks that contain query terms
            reranked_data = []
            
            for doc, sim, meta in zip(
                results["documents"], 
                results["similarities"], 
                results["metadata"]
            ):
                # Calculate relevance score
                query_terms = query.lower().split()
                doc_lower = doc.lower()
                
                # Count query term matches
                term_matches = sum(1 for term in query_terms if term in doc_lower)
                
                # Combined score: similarity + term matches
                relevance_score = sim + (term_matches * 0.1)
                
                reranked_data.append({
                    "document": doc,
                    "similarity": sim,
                    "metadata": meta,
                    "relevance_score": relevance_score
                })
            
            # Sort by relevance score
            reranked_data.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Take top K after reranking
            top_k = min(config.TOP_K_RERANK, len(reranked_data))
            reranked_data = reranked_data[:top_k]
            
            return {
                "documents": [item["document"] for item in reranked_data],
                "similarities": [item["similarity"] for item in reranked_data],
                "metadata": [item["metadata"] for item in reranked_data],
                "relevance_scores": [item["relevance_score"] for item in reranked_data]
            }
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return results