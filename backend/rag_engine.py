#Agentic RAG System/backend/rag_engine.py
from typing import Dict, List
from config import Config
from ollama_client import OllamaClient

class RAGEngine:
    def __init__(self, milvus_client):
        self.milvus_client = milvus_client
        self.ollama_client = OllamaClient()
        
        # Test connections
        self.ollama_client.test_connection()
    
    def retrieve_context(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve relevant context from vector store"""
        # Search for relevant documents
        results = self.milvus_client.search(query, top_k)
        
        if not results["documents"]:
            return {"chunks": [], "sources": []}
        
        # Format context for LLM
        chunks = []
        sources = set()
        
        for doc, meta in zip(results["documents"], results["metadata"]):
            source = meta.get("source", "Unknown")
            chunks.append(f"[Source: {source}]\n{doc}")
            sources.add(source)
        
        return {
            "chunks": chunks,
            "sources": list(sources),
            "raw_results": results
        }
    
    def generate_answer(self, query: str, context: Dict) -> str:
        """Generate answer using retrieved context"""
        if not context["chunks"]:
            return "I couldn't find relevant information in the documents."
        
        # Format context
        context_text = "\n\n".join(context["chunks"])
        
        # Create prompt
        prompt = f"""Based on the following documents, answer the question.

Documents:
{context_text}

Question: {query}

Instructions:
1. Answer based ONLY on the provided documents
2. If information is not in the documents, say so
3. Be concise and factual
4. Mention the source documents when appropriate

Answer:"""
        
        # Generate response
        system_prompt = "You are a helpful assistant that answers questions based on provided documents."
        answer = self.ollama_client.generate(prompt, system_prompt, Config.AGENT_TEMPERATURE)
        
        return answer
    
    def summarize_documents(self) -> str:
        """Summarize all documents"""
        # Get sample of documents
        results = self.milvus_client.search("summary overview", top_k=10)
        
        if not results["documents"]:
            return "No documents to summarize."
        
        # Combine documents
        combined_text = "\n\n".join(results["documents"][:5])
        
        prompt = f"""Please provide a concise summary of the following documents:

{combined_text}

Summary:"""
        
        return self.ollama_client.generate(prompt, temperature=0.2)