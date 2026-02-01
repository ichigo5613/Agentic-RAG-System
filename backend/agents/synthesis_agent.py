# backend/agents/synthesis_agent.py
from typing import Dict, Any, List
import re

from backend.utils.logger import logger

class SynthesisAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def synthesize(self, query: str, retrieval_results: Dict[str, Any]) -> str:
        """Synthesize answer from retrieved information"""
        if not retrieval_results.get("documents"):
            return self._generate_no_info_response(query)
        
        try:
            # Format context for LLM
            context = self._format_context(retrieval_results)
            
            # Generate answer
            answer = self._generate_answer(query, context)
            
            # Add citations
            answer_with_citations = self._add_citations(answer, retrieval_results)
            
            logger.debug(f"Synthesized answer of length {len(answer_with_citations)}")
            return answer_with_citations
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {str(e)}")
            return f"I encountered an error while processing your query. Please try again. Error: {str(e)[:100]}"
    
    def _format_context(self, results: Dict[str, Any]) -> str:
        """Format retrieval results into context string"""
        context_parts = []
        
        documents = results.get("documents", [])
        metadata_list = results.get("metadata", [])
        
        for i, doc in enumerate(documents, 1):
            if i-1 < len(metadata_list):
                meta = metadata_list[i-1]
                # Parse metadata if it's a JSON string
                if isinstance(meta, str):
                    try:
                        import json
                        meta = json.loads(meta)
                    except:
                        meta = {"source": "Unknown"}
            else:
                meta = {"source": "Unknown"}
            
            source = meta.get("source", "Document") if isinstance(meta, dict) else "Unknown"
            page = meta.get("page", "") if isinstance(meta, dict) else ""
            page_info = f" (page {page})" if page else ""
            
            context_parts.append(
                f"[Document {i}: {source}{page_info}]\n"
                f"{doc}\n"
            )
        
        return "\n".join(context_parts)

    def _add_citations(self, answer: str, results: Dict[str, Any]) -> str:
        """Add formal citations to answer"""
        metadata_list = results.get("metadata", [])
        if not metadata_list:
            return answer
        
        # Add citation references
        citations_section = "\n\n**References:**\n"
        for i, meta in enumerate(metadata_list, 1):
            # Parse metadata if it's a JSON string
            if isinstance(meta, str):
                try:
                    import json
                    meta = json.loads(meta)
                except:
                    meta = {"source": "Unknown Document"}
            
            source = meta.get("source", "Unknown Document") if isinstance(meta, dict) else "Unknown Document"
            page = meta.get("page", "") if isinstance(meta, dict) else ""
            page_info = f", page {page}" if page else ""
            
            citations_section += f"{i}. {source}{page_info}\n"
        
        return answer + citations_section
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM"""
        system_prompt = """You are an expert document analysis assistant. 
        Answer the user's question based ONLY on the provided document excerpts.
        
        Guidelines:
        1. Base your answer strictly on the provided context
        2. Be concise but comprehensive
        3. If information is not in the context, say so
        4. Use clear, professional language
        5. Include relevant details and specifics
        6. Reference document sources in your answer
        
        Format your answer with:
        - A clear, direct answer first
        - Supporting details from the documents
        - Citations like [Document X] where relevant"""
        
        user_prompt = f"""Question: {query}

        Context from documents:
        {context}

        Answer based on the documents:"""
        
        return self.llm_client.generate(user_prompt, system_prompt)
    
    def _add_citations(self, answer: str, results: Dict[str, Any]) -> str:
        """Add formal citations to answer"""
        if not results.get("metadata"):
            return answer
        
        # Add citation references
        citations_section = "\n\n**References:**\n"
        for i, meta in enumerate(results["metadata"], 1):
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", "")
            page_info = f", page {page}" if page else ""
            
            citations_section += f"{i}. {source}{page_info}\n"
        
        return answer + citations_section
    
    def _generate_no_info_response(self, query: str) -> str:
        """Generate response when no information is found"""
        no_info_responses = [
            f"I couldn't find specific information about '{query}' in the uploaded documents. "
            "You might want to upload relevant documents or rephrase your question.",
            
            f"The documents don't contain information about '{query}'. "
            "Please make sure you've uploaded relevant files or try asking about different topics.",
            
            f"Based on the available documents, I don't have information about '{query}'. "
            "Consider uploading documents that might contain this information."
        ]
        
        import random
        return random.choice(no_info_responses)