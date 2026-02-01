# backend/agents/orchestrator.py
from typing import List, Dict, Any, List, Optional
import time
from enum import Enum
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import config
from backend.utils.logger import logger, log_agent_step
from .query_router import QueryRouterAgent
from .query_decomposer import QueryDecomposerAgent
from .retrieval_agent import RetrievalAgent
from .synthesis_agent import SynthesisAgent

class AgentState(dict):
    """State for agent workflow"""
    pass

class AgentOrchestrator:
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm_client = llm_client
        
        # Initialize agents
        self.query_router = QueryRouterAgent(llm_client)
        self.query_decomposer = QueryDecomposerAgent(llm_client)
        self.retrieval_agent = RetrievalAgent(vector_store, llm_client)
        self.synthesis_agent = SynthesisAgent(llm_client)
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        logger.info("Initialized Agent Orchestrator")
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("decompose_query", self._decompose_query)
        workflow.add_node("retrieve_information", self._retrieve_information)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        workflow.add_node("direct_answer", self._direct_answer)
        
        # Set entry point
        workflow.set_entry_point("route_query")
        
        # Define edges based on routing
        workflow.add_conditional_edges(
            "route_query",
            self._decide_next_step,
            {
                "decompose": "decompose_query",
                "direct": "direct_answer",
                "retrieve": "retrieve_information"
            }
        )
        
        # Connect decompose to retrieve
        workflow.add_edge("decompose_query", "retrieve_information")
        
        # Connect retrieve to synthesize
        workflow.add_edge("retrieve_information", "synthesize_answer")
        
        # Connect direct answer to end
        workflow.add_edge("direct_answer", END)
        workflow.add_edge("synthesize_answer", END)
        
        return workflow.compile()
    
    def _route_query(self, state: AgentState) -> AgentState:
        """Route query to appropriate path"""
        query = state.get("query", "")
        
        log_agent_step("Orchestrator", "Routing Query", {"query": query})
        
        route_decision = self.query_router.route(query)
        state["route_decision"] = route_decision
        
        # Ensure thought_process is a list
        if "thought_process" not in state:
            state["thought_process"] = []
        elif isinstance(state["thought_process"], str):
            state["thought_process"] = [state["thought_process"]]
        
        state["thought_process"].append(f"Routing: {route_decision}")
        
        return state
    
    def _decide_next_step(self, state: AgentState) -> str:
            """Decide next step based on routing"""
            route = state.get("route_decision", "retrieve")
            
            if route == "simple" or route == "conversational":
                return "direct"
            elif route == "complex":
                return "decompose"
            else:
                return "retrieve"
    
    def _decompose_query(self, state: AgentState) -> AgentState:
        """Decompose complex query into sub-queries"""
        query = state.get("query", "")
        
        log_agent_step("Orchestrator", "Decomposing Query", {"query": query})
        
        sub_queries = self.query_decomposer.decompose(query)
        state["sub_queries"] = sub_queries
        state["thought_process"].append(f"Decomposed into {len(sub_queries)} sub-queries")
        
        return state
    
    def _retrieve_information(self, state: AgentState) -> AgentState:
        """Retrieve information based on query/sub-queries"""
        query = state.get("query", "")
        sub_queries = state.get("sub_queries", [])
        
        log_agent_step("Orchestrator", "Retrieving Information", {
            "query": query,
            "sub_queries": sub_queries
        })
        
        # If we have sub-queries, retrieve for each
        if sub_queries:
            all_results = []
            for sub_q in sub_queries:
                results = self.retrieval_agent.retrieve(sub_q)
                all_results.append(results)
            
            # Combine results
            combined_results = self._combine_retrieval_results(all_results)
            state["retrieval_results"] = combined_results
        else:
            # Single query retrieval
            results = self.retrieval_agent.retrieve(query)
            state["retrieval_results"] = results
        
        state["thought_process"].append(
            f"Retrieved {len(state['retrieval_results'].get('documents', []))} chunks"
        )
        
        return state
    
    def _synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesize final answer from retrieved information"""
        query = state.get("query", "")
        results = state.get("retrieval_results", {})
        
        log_agent_step("Orchestrator", "Synthesizing Answer", {
            "query": query,
            "chunks_count": len(results.get("documents", []))
        })
        
        answer = self.synthesis_agent.synthesize(query, results)
        state["answer"] = answer
        
        # Add citations
        state["citations"] = self._extract_citations(results)
        state["thought_process"].append("Synthesized final answer with citations")
        
        return state
    
    def _direct_answer(self, state: AgentState) -> AgentState:
        """Generate direct answer without retrieval"""
        query = state.get("query", "")
        
        log_agent_step("Orchestrator", "Direct Answer", {"query": query})
        
        # Use LLM for direct conversation
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=query)
        ]
        
        answer = self.llm_client.chat([{"role": "system", "content": "You are a helpful assistant."},
                                      {"role": "user", "content": query}])
        
        state["answer"] = answer
        state["thought_process"].append("Generated direct answer (no retrieval needed)")
        
        return state
    
    def _combine_retrieval_results(self, results_list: List[Dict]) -> Dict:
        """Combine results from multiple retrievals"""
        combined = {"documents": [], "similarities": [], "metadata": []}
        
        for results in results_list:
            combined["documents"].extend(results.get("documents", []))
            combined["similarities"].extend(results.get("similarities", []))
            combined["metadata"].extend(results.get("metadata", []))
        
        return combined
    
    def _extract_citations(self, results: Dict) -> List[Dict]:
        """Extract citation information from results"""
        citations = []
        for i, (doc, meta) in enumerate(zip(
            results.get("documents", []),
            results.get("metadata", [])
        )):
            citations.append({
                "id": i + 1,
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", 1),
                "snippet": doc[:200] + "..." if len(doc) > 200 else doc
            })
        return citations
    
    def process_query(self, query: str, use_agentic: bool = True) -> Dict[str, Any]:
        """Main method to process a query"""
        start_time = time.time()
        
        if not use_agentic:
            # Simple RAG fallback
            results = self.vector_store.advanced_search(query)
            answer = self.synthesis_agent.synthesize(query, results)
            
            return {
                "query": query,
                "answer": answer,
                "citations": self._extract_citations(results),
                "processing_time": time.time() - start_time,
                "agentic": False
            }
        
        try:
            # Initialize state with proper thought_process
            initial_state = AgentState({
                "query": query,
                "thought_process": [],  # Initialize as empty list
                "agentic": True
            })
            
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Extract results - handle missing thought_process
            response = {
                "query": query,
                "answer": final_state.get("answer", "No answer generated."),
                "thought_process": final_state.get("thought_process", []),  # Default to empty list
                "citations": final_state.get("citations", []),
                "sub_queries": final_state.get("sub_queries", []),
                "route_decision": final_state.get("route_decision", "unknown"),
                "processing_time": time.time() - start_time,
                "agentic": True
            }
            
            logger.info(f"Agentic query processed in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Agentic processing failed: {str(e)}")
            
            # Fallback to simple RAG
            results = self.vector_store.advanced_search(query)
            answer = self.synthesis_agent.synthesize(query, results)
            
            return {
                "query": query,
                "answer": answer,
                "citations": self._extract_citations(results),
                "processing_time": time.time() - start_time,
                "agentic": False,
                "error": str(e)
            }
    # In the process_query method, add debug logging:

    def process_query(self, query: str, use_agentic: bool = True) -> Dict[str, Any]:
        """Main method to process a query"""
        start_time = time.time()
        
        logger.info(f"🔍 Processing query: '{query}' (agentic={use_agentic})")
        
        if not use_agentic:
            logger.info("🔄 Using simple RAG (non-agentic)")
            # Simple RAG fallback
            results = self.vector_store.advanced_search(query)
            logger.info(f"📚 Retrieved {len(results.get('documents', []))} chunks")
            answer = self.synthesis_agent.synthesize(query, results)
            
            return {
                "query": query,
                "answer": answer,
                "citations": self._extract_citations(results),
                "processing_time": time.time() - start_time,
                "agentic": False
            }
        
        try:
            # Initialize state with proper thought_process
            initial_state = AgentState({
                "query": query,
                "thought_process": [],
                "agentic": True
            })
            
            logger.info("🚀 Starting agentic workflow...")
            
            # Execute workflow
            final_state = self.workflow.invoke(initial_state)
            
            logger.info(f"✅ Workflow completed. Final state keys: {list(final_state.keys())}")
            
            # Extract results
            response = {
                "query": query,
                "answer": final_state.get("answer", "No answer generated."),
                "thought_process": final_state.get("thought_process", []),
                "citations": final_state.get("citations", []),
                "sub_queries": final_state.get("sub_queries", []),
                "route_decision": final_state.get("route_decision", "unknown"),
                "processing_time": time.time() - start_time,
                "agentic": True
            }
            
            logger.info(f"Agentic query processed in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Agentic processing failed: {str(e)}", exc_info=True)
            
            # Fallback to simple RAG
            logger.info("🔄 Falling back to simple RAG...")
            results = self.vector_store.advanced_search(query)
            answer = self.synthesis_agent.synthesize(query, results)
            
            return {
                "query": query,
                "answer": answer,
                "citations": self._extract_citations(results),
                "processing_time": time.time() - start_time,
                "agentic": False,
                "error": str(e)
            }