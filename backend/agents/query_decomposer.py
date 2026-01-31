# backend/agents/query_decomposer.py
from typing import List
import json

from backend.utils.logger import logger

class QueryDecomposerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def decompose(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        try:
            prompt = f"""
            Decompose the following complex query into independent sub-questions 
            that can be answered separately and then combined:
            
            Original Query: "{query}"
            
            Instructions:
            1. Break down the query into 2-4 logical sub-questions
            2. Each sub-question should be self-contained
            3. Ensure sub-questions cover all aspects of the original query
            4. Return as a JSON array of strings
            
            Example format:
            {{
                "sub_queries": ["sub-question 1", "sub-question 2", ...]
            }}
            
            Response:
            """
            
            response = self.llm_client.generate(prompt)
            
            # Try to parse JSON
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed = json.loads(json_str)
                    sub_queries = parsed.get("sub_queries", [])
                else:
                    # Fallback: split by lines
                    sub_queries = [
                        line.strip()[3:] if line.strip().startswith(("1.", "2.", "3.", "4.")) 
                        else line.strip()
                        for line in response.split('\n')
                        if line.strip() and not line.strip().startswith(("{", "}"))
                    ]
            except json.JSONDecodeError:
                # Alternative parsing
                sub_queries = self._extract_subqueries_fallback(response)
            
            # Clean and validate sub-queries
            valid_sub_queries = []
            for q in sub_queries:
                q_clean = q.strip().strip('"\'')
                if q_clean and len(q_clean) > 5:  # Minimum length
                    valid_sub_queries.append(q_clean)
            
            # If no valid sub-queries, return original query
            if not valid_sub_queries:
                valid_sub_queries = [query]
            
            logger.debug(f"Decomposed query into {len(valid_sub_queries)} sub-queries")
            return valid_sub_queries
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {str(e)}")
            return [query]  # Fallback to original query
    
    def _extract_subqueries_fallback(self, text: str) -> List[str]:
        """Fallback method to extract sub-queries"""
        lines = text.strip().split('\n')
        sub_queries = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and JSON markers
            if not line or line in ['{', '}', '[', ']']:
                continue
            
            # Remove numbering and quotes
            if line.startswith(('1.', '2.', '3.', '4.', '- ', '* ')):
                line = line[2:].strip()
            
            line = line.strip('"\'')
            
            if line and not line.startswith('{') and not line.startswith('['):
                sub_queries.append(line)
        
        return sub_queries