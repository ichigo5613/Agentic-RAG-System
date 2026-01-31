# backend/agents/query_router.py
from typing import Dict, Any
from enum import Enum

from backend.utils.logger import logger

class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    CONVERSATIONAL = "conversational"
    FACTUAL = "factual"

class QueryRouterAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def route(self, query: str) -> str:
        """Route query to appropriate processing path"""
        try:
            # Simple rule-based routing first
            if self._is_conversational(query):
                return "conversational"
            elif self._is_simple_factual(query):
                return "simple"
            elif self._is_complex_query(query):
                return "complex"
            else:
                # Use LLM for ambiguous cases
                return self._llm_route(query)
        except Exception as e:
            logger.warning(f"Query routing failed: {str(e)}")
            return "retrieve"  # Default to retrieval
    
    def _is_conversational(self, query: str) -> bool:
        """Check if query is conversational/greeting"""
        conversational_phrases = [
            "hello", "hi", "hey", "how are you", "good morning",
            "good afternoon", "good evening", "thank you", "thanks"
        ]
        query_lower = query.lower().strip()
        
        # Check for greetings
        if any(phrase in query_lower for phrase in conversational_phrases):
            return True
        
        # Check for short, non-question queries
        if len(query.split()) <= 3 and not query.endswith('?'):
            return True
        
        return False
    
    def _is_simple_factual(self, query: str) -> bool:
        """Check if query is simple factual"""
        # Simple questions usually start with who, what, when, where
        simple_starters = ["who ", "what ", "when ", "where ", "which "]
        query_lower = query.lower()
        
        if any(query_lower.startswith(starter) for starter in simple_starters):
            # Check if it's a short question
            if len(query.split()) <= 10:
                return True
        
        return False
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex (needs decomposition)"""
        complex_indicators = [
            "compare", "analyze", "explain", "describe", "summarize",
            "advantages and disadvantages", "pros and cons", "difference between",
            "similarities between", "how to", "step by step", "list of"
        ]
        
        query_lower = query.lower()
        
        # Check for complex indicators
        if any(indicator in query_lower for indicator in complex_indicators):
            return True
        
        # Check for multiple questions in one query
        if query.count('?') > 1 or " and " in query_lower:
            return True
        
        # Long queries are often complex
        if len(query.split()) > 15:
            return True
        
        return False
    
    def _llm_route(self, query: str) -> str:
        """Use LLM for routing decisions"""
        prompt = f"""
        Analyze the following query and determine the best processing approach:
        
        Query: "{query}"
        
        Options:
        1. "simple" - Simple factual question that can be answered directly
        2. "complex" - Complex query that needs decomposition into sub-questions
        3. "conversational" - Greeting or conversational query
        4. "retrieve" - Query that needs document retrieval
        
        Respond with only one of: simple, complex, conversational, retrieve
        
        Decision:
        """
        
        try:
            decision = self.llm_client.generate(prompt).strip().lower()
            
            # Validate decision
            valid_decisions = ["simple", "complex", "conversational", "retrieve"]
            if decision in valid_decisions:
                logger.debug(f"LLM routed query to: {decision}")
                return decision
            else:
                return "retrieve"  # Default
        except:
            return "retrieve"  # Default on error