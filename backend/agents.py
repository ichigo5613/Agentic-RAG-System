#Agentic RAG System/backend/agents.py
from typing import Dict, List, Any
from config import Config
from ollama_client import OllamaClient
from mcp_tools import MCPTools

class AgentOrchestrator:
    def __init__(self):
        self.ollama = OllamaClient()
        self.tools = MCPTools()
        
    def process_query(self, query: str, use_tools: bool = True, 
                     rag_engine: Any = None) -> Dict:
        """Process query using agentic workflow"""
        thought_process = []
        tools_used = []
        
        # Step 1: Query Analysis
        thought_process.append("🔍 Analyzing query...")
        query_type, needs_tools = self._analyze_query(query)
        
        # Step 2: Context Retrieval
        context = None
        if rag_engine:
            thought_process.append("📚 Retrieving relevant context...")
            context = rag_engine.retrieve_context(query, Config.MAX_CONTEXT_CHUNKS)
        
        # Step 3: Tool Usage (if needed)
        tool_results = {}
        if use_tools and needs_tools:
            thought_process.append("🛠️ Using tools...")
            tool_results = self._use_tools(query, query_type)
            tools_used = list(tool_results.keys())
        
        # Step 4: Answer Synthesis
        thought_process.append("🧠 Synthesizing answer...")
        answer = self._synthesize_answer(
            query=query,
            query_type=query_type,
            context=context,
            tool_results=tool_results
        )
        
        return {
            "answer": answer,
            "thought_process": thought_process,
            "tools_used": tools_used,
            "context_used": bool(context and context["chunks"]),
            "query_type": query_type
        }
    
    def _analyze_query(self, query: str) -> tuple:
        """Determine query type and needs"""
        prompt = f"""Analyze this query and determine:
1. Query type: [factual, analytical, calculation, summary, other]
2. Needs tools: [yes/no]
3. Brief reasoning

Query: {query}

Respond in JSON format:
{{
    "type": "string",
    "needs_tools": "boolean",
    "reasoning": "string"
}}"""
        
        response = self.ollama.generate(prompt)
        analysis = self.ollama.extract_json(response)
        
        # Default values
        query_type = analysis.get("type", "factual")
        needs_tools = analysis.get("needs_tools", False)
        
        return query_type, needs_tools
    
    def _use_tools(self, query: str, query_type: str) -> Dict:
        """Use appropriate tools based on query type"""
        results = {}
        
        if query_type == "calculation":
            results["calculator"] = self.tools.calculate(query)
        elif "search" in query.lower() or "web" in query.lower():
            results["web_search"] = self.tools.web_search(query)
        elif "time" in query.lower() or "date" in query.lower():
            results["datetime"] = self.tools.get_datetime()
        
        return results
    
    def _synthesize_answer(self, query: str, query_type: str, 
                          context: Dict, tool_results: Dict) -> str:
        """Synthesize final answer"""
        # Prepare context text
        context_text = ""
        if context and context["chunks"]:
            context_text = "\n\n".join(context["chunks"])
        
        # Prepare tool results text
        tool_text = ""
        if tool_results:
            tool_text = "\n".join([f"{k}: {v}" for k, v in tool_results.items()])
        
        # Create synthesis prompt
        prompt = f"""Synthesize an answer for this query:

Query: {query}
Query Type: {query_type}

Available Information:
{'='*40}
Document Context:
{context_text if context_text else 'No document context available'}

{'='*40}
Tool Results:
{tool_text if tool_text else 'No tool results'}

{'='*40}
Instructions:
1. Use all available information
2. If using documents, cite sources
3. If using tools, mention them
4. Be clear and concise
5. If conflicting information, note it

Answer:"""
        
        system_prompt = """You are an AI assistant that synthesizes information from multiple sources.
        Integrate document context with tool results to provide comprehensive answers."""
        
        return self.ollama.generate(prompt, system_prompt, Config.AGENT_TEMPERATURE)


class QueryAnalyzerAgent:
    def analyze(self, query: str) -> Dict:
        """Simple query analyzer"""
        if "calculate" in query.lower() or any(op in query for op in ['+', '-', '*', '/']):
            return {"type": "calculation", "needs_tools": True}
        elif "search" in query.lower() or "find" in query.lower():
            return {"type": "search", "needs_tools": True}
        elif "summar" in query.lower():
            return {"type": "summary", "needs_tools": False}
        else:
            return {"type": "factual", "needs_tools": False}


class SynthesisAgent:
    def __init__(self, ollama_client):
        self.ollama = ollama_client
    
    def synthesize(self, query: str, context: str, tools: Dict = None) -> str:
        """Synthesize answer from multiple sources"""
        prompt = f"""Based on the following information, answer: {query}

Context from documents:
{context}

Tool results: {tools if tools else 'None'}

Answer:"""
        
        return self.ollama.generate(prompt)