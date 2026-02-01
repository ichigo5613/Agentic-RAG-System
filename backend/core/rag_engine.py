# backend/core/rag_engine.py
from typing import List, Dict, Any, Optional
import time
from functools import lru_cache

from backend.config import config
from backend.utils.logger import logger, log_query_processing
from backend.agents.orchestrator import AgentOrchestrator
# from backend.core.vector_store import EnhancedVectorStore
from backend.core.milvus_store import MilvusVectorStore
from backend.models.llm_client import LLMClient

class RAGEngine:
    def __init__(self):
        # Initialize components
        self.llm_client = LLMClient()
        self.vector_store = MilvusVectorStore()
        
        # Initialize agent orchestrator
        self.agent_orchestrator = AgentOrchestrator(
            vector_store=self.vector_store,
            llm_client=self.llm_client
        )
        
        # Initialize cache
        self.cache = {}
        
        logger.info("Initialized RAG Engine with Agentic capabilities")
    
    def query(self, 
              query: str, 
              use_agentic: bool = True,
              top_k: Optional[int] = None) -> Dict[str, Any]:
        """Main query method with agentic capabilities"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query}_{use_agentic}_{top_k}"
        if cache_key in self.cache:
            logger.debug(f"Using cached response for query: {query[:50]}...")
            cached_result = self.cache[cache_key]
            cached_result["cached"] = True
            return cached_result
        
        try:
            if use_agentic:
                # Use agentic workflow
                result = self.agent_orchestrator.process_query(query, use_agentic=True)
            else:
                # Use simple RAG
                results = self.vector_store.advanced_search(query)
                answer = self._generate_simple_answer(query, results)
                result = {
                    "query": query,
                    "answer": answer,
                    "citations": self._extract_citations(results),
                    "agentic": False
                }
            
            # Add processing metrics
            result["processing_time"] = time.time() - start_time
            result["cached"] = False
            
            # Log processing
            log_query_processing(query, result)
            
            # Cache the result
            if len(self.cache) >= config.MAX_CACHE_SIZE:
                # Remove oldest entry (simple LRU)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", extra={"error": str(e)})
            
            # Return error response
            return {
                "query": query,
                "answer": f"I encountered an error while processing your query. Please try again. Error: {str(e)[:100]}",
                "citations": [],
                "processing_time": time.time() - start_time,
                "error": str(e),
                "agentic": use_agentic
            }
    
    def _generate_simple_answer(self, query: str, results: Dict[str, Any]) -> str:
        """Generate answer without agentic processing"""
        if not results.get("documents"):
            return "I couldn't find relevant information in the documents. Please try a different query."
        
        # Format context
        context_parts = []
        for i, doc in enumerate(results["documents"][:3], 1):  # Use top 3
            context_parts.append(f"[Excerpt {i}]: {doc}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based on the following document excerpts, answer the question.

Question: {query}

Document Excerpts:
{context}

Answer concisely based only on the excerpts:"""
        
        return self.llm_client.generate(prompt)
    
    def _extract_citations(self, results: Dict[str, Any]) -> List[Dict]:
        """Extract citation information"""
        citations = []
        for i, meta in enumerate(results.get("metadata", []), 1):
            citations.append({
                "id": i,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", ""),
                "chunk": i
            })
        return citations
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        logger.info("Cleared RAG Engine cache")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            "llm_connected": self.llm_client.test_connection(),
            "vector_store_connected": self.vector_store.test_connection(),
            "documents_count": self.vector_store.count_documents(),
            "cache_size": len(self.cache),
            "ollama_model": config.OLLAMA_MODEL,
            "embedding_model": config.EMBEDDING_MODEL
        }