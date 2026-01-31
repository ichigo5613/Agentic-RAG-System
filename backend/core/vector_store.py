# backend/core/vector_store.py - FIXED IMPORTS
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from backend.config import config
from backend.models.llm_client import LLMClient
from backend.utils.logger import logger

class EnhancedVectorStore:
    def __init__(self):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize retriever with advanced features
        self.base_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": config.TOP_K_RETRIEVAL,
                "score_threshold": config.SIMILARITY_THRESHOLD
            }
        )
        
        # Initialize LLM for compression
        self.llm_client = LLMClient()
        
        logger.info(f"Initialized Enhanced Vector Store with collection: {config.COLLECTION_NAME}")
    
    def add_documents(self, texts: List[str], metadata_list: List[Dict]) -> List[str]:
        """Add documents to vector store with enhanced metadata"""
        if not texts:
            return []
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Enhance metadata with timestamps
        enhanced_metadata = []
        for i, meta in enumerate(metadata_list):
            enhanced_meta = meta.copy()
            enhanced_meta.update({
                "id": ids[i],
                "created_at": datetime.utcnow().isoformat(),
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": len(texts[i])
            })
            enhanced_metadata.append(enhanced_meta)
        
        # Add to vector store
        self.vector_store.add_texts(
            texts=texts,
            metadatas=enhanced_metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(texts)} documents to ChromaDB")
        return ids
    
    def search(self, query: str, top_k: int = None, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Basic similarity search"""
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=top_k,
                filter=filters
            )
            
            documents = [doc.page_content for doc, score in results]
            scores = [score for doc, score in results]
            metadata = [doc.metadata for doc, score in results]
            
            # Convert distance to similarity score (assuming cosine distance)
            similarities = [1 - float(score) for score in scores]
            
            return {
                "documents": documents,
                "similarities": similarities,
                "metadata": metadata,
                "raw_scores": scores
            }
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"documents": [], "similarities": [], "metadata": []}
    
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
        
        # 3. Multi-query retrieval (simplified implementation)
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
    
    def count_documents(self) -> int:
        """Count total documents in collection"""
        try:
            return self.vector_store._collection.count()
        except:
            return 0
    
    def list_documents(self) -> List[str]:
        """List unique document sources"""
        try:
            all_metadata = self.vector_store.get()["metadatas"]
            sources = set(meta.get("source", "Unknown") for meta in all_metadata)
            return list(sources)
        except:
            return []
    
    def clear_collection(self):
        """Clear the entire collection"""
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.CHROMA_PERSIST_DIR,
                embedding_function=self.embedding_model
            )
            logger.info("Cleared vector store collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to vector store"""
        try:
            count = self.count_documents()
            logger.info(f"Vector store connection test successful. Documents: {count}")
            return True
        except Exception as e:
            logger.error(f"Vector store connection test failed: {str(e)}")
            return False