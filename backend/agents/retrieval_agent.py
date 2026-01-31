# backend/agents/retrieval_agent.py
from typing import List, Dict, Any
import time

from backend.config import config
from backend.utils.logger import logger

class RetrievalAgent:
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant information for a query"""
        start_time = time.time()
        
        try:
            # Use advanced search with multiple techniques
            results = self.vector_store.advanced_search(
                query,
                use_hyde=config.ENABLE_HYDE,
                use_multi_query=True
            )
            
            # Apply additional filtering if needed
            if results["documents"]:
                results = self._filter_low_quality(results)
            
            elapsed = time.time() - start_time
            
            logger.debug(f"Retrieval completed in {elapsed:.2f}s", extra={
                "query": query,
                "chunks_retrieved": len(results["documents"]),
                "elapsed_time": elapsed
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            return {"documents": [], "similarities": [], "metadata": []}
    
    def _filter_low_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out low-quality retrieval results"""
        filtered_docs = []
        filtered_sims = []
        filtered_meta = []
        
        for doc, sim, meta in zip(
            results["documents"],
            results["similarities"],
            results["metadata"]
        ):
            # Filter criteria
            if self._is_high_quality(doc, sim):
                filtered_docs.append(doc)
                filtered_sims.append(sim)
                filtered_meta.append(meta)
        
        return {
            "documents": filtered_docs,
            "similarities": filtered_sims,
            "metadata": filtered_meta
        }
    
    def _is_high_quality(self, document: str, similarity: float) -> bool:
        """Determine if a retrieved document is high quality"""
        # Check similarity threshold
        if similarity < config.SIMILARITY_THRESHOLD:
            return False
        
        # Check document length
        if len(document.strip()) < 20:  # Too short
            return False
        
        # Check for meaningful content
        if document.strip().lower().startswith(("http://", "https://", "www.")):
            return False
        
        # Check for common low-quality patterns
        low_quality_patterns = [
            "[removed]", "[deleted]", "loading...", "please wait",
            "error:", "404", "not found", "page not available"
        ]
        
        doc_lower = document.lower()
        if any(pattern in doc_lower for pattern in low_quality_patterns):
            return False
        
        return True