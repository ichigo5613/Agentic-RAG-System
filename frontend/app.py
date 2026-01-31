# frontend/app.py
import streamlit as st
import requests
import json
import time
import os
from typing import List, Dict, Any
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
BACKEND_URL = "http://127.0.0.1:5000"

# Define allowed extensions in frontend
ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt', 'xlsx', 'xls', 'pptx', 'md']

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'agent_mode' not in st.session_state:
    st.session_state.agent_mode = True
if 'show_thought_process' not in st.session_state:
    st.session_state.show_thought_process = True
if 'api_status' not in st.session_state:
    st.session_state.api_status = "unknown"
if 'backend_model' not in st.session_state:
    st.session_state.backend_model = "Unknown"
if 'backend_status' not in st.session_state:
    st.session_state.backend_status = "unknown"
if 'last_api_check' not in st.session_state:
    st.session_state.last_api_check = 0

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 6px solid #4f46e5;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 6px solid #10b981;
    }
    .agent-thinking {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
        border-left: 6px solid #f59e0b;
        font-style: italic;
        font-size: 0.9em;
    }
    .citation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #6b7280;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        font-size: 0.85em;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    .status-offline {
        background-color: #ef4444;
    }
    .debug-output {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
        white-space: pre-wrap;
        margin: 10px 0;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if backend API is running - COMPLETE FIX with caching"""
    # Store result in session state to avoid repeated calls
    if 'last_api_check' in st.session_state:
        elapsed = time.time() - st.session_state.last_api_check
        if elapsed < 5:  # Cache for 5 seconds
            return st.session_state.api_status == "online"
    
    try:
        # Test the connection
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Store model info in session state
            st.session_state.backend_model = data.get('models', {}).get('ollama', 'Unknown')
            st.session_state.backend_status = data.get('status', 'unknown')
            st.session_state.api_status = "online"
            st.session_state.last_api_check = time.time()
            return True
        else:
            st.session_state.api_status = "offline"
            st.session_state.last_api_check = time.time()
            return False
            
    except requests.exceptions.ConnectionError:
        st.session_state.api_status = "offline"
        st.session_state.last_api_check = time.time()
        return False
    except Exception as e:
        # Log error but don't crash
        print(f"Backend check error: {e}")
        st.session_state.api_status = "offline"
        st.session_state.last_api_check = time.time()
        return False

def debug_backend_connection():
    """Debug backend connection issues"""
    debug_output = []
    debug_output.append("\n" + "="*60)
    debug_output.append("🔍 Debugging Backend Connection")
    debug_output.append("="*60)
    
    debug_output.append(f"Frontend BACKEND_URL: {BACKEND_URL}")
    
    try:
        debug_output.append(f"\n1. Testing {BACKEND_URL}/health ...")
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        debug_output.append(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            debug_output.append(f"✅ Backend is working!")
            debug_output.append(f"   Status: {data.get('status')}")
            debug_output.append(f"   Model: {data.get('models', {}).get('ollama')}")
            debug_output.append(f"   LLM Connected: {data.get('components', {}).get('llm')}")
            return True, "\n".join(debug_output)
        else:
            debug_output.append(f"❌ Unexpected status: {response.status_code}")
            debug_output.append(f"   Response: {response.text[:200]}")
            return False, "\n".join(debug_output)
            
    except requests.exceptions.ConnectionError as e:
        debug_output.append(f"❌ Connection Error: {e}")
        debug_output.append("\nPossible solutions:")
        debug_output.append("1. Is backend running? Check with: python backend/app.py")
        debug_output.append("2. Try different URL: http://localhost:5000 or http://127.0.0.1:5000")
        debug_output.append("3. Check Windows Firewall")
        return False, "\n".join(debug_output)
    except Exception as e:
        debug_output.append(f"❌ Error: {e}")
        return False, "\n".join(debug_output)

# frontend/app.py - UPDATE UPLOAD FUNCTION
def upload_file(file):
    """Upload file to backend with better timeout handling"""
    try:
        # Check file size first
        file_size = len(file.getvalue())
        MAX_SIZE = 50 * 1024 * 1024  # 50MB
        
        if file.name.lower().endswith('.pdf') and file_size > MAX_SIZE:
            st.error(f"PDF file too large ({file_size/(1024*1024):.1f}MB). Maximum is 50MB.")
            return None
        
        files = {'file': (file.name, file.getvalue(), file.type)}
        
        # Adjust timeout based on file size
        if file_size > 10 * 1024 * 1024:  # >10MB
            timeout = 300  # 5 minutes for large files
        else:
            timeout = 120  # 2 minutes for smaller files
        
        response = requests.post(
            f"{BACKEND_URL}/upload", 
            files=files, 
            timeout=timeout
        )
        return response
    except requests.exceptions.Timeout:
        st.error(f"Upload timed out. The file might be too large. Try a smaller file.")
        return None
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None
        
def send_query(query, agent_mode=True):
    """Send query to backend"""
    endpoint = "/query" if agent_mode else "/quick_query"
    payload = {
        "query": query,
        "use_agentic": agent_mode
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}{endpoint}",
            json=payload,
            timeout=120
        )
        return response
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return None

def get_system_status():
    """Get system status from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def clear_documents():
    """Clear all documents from backend"""
    try:
        response = requests.post(f"{BACKEND_URL}/clear", timeout=10)
        return response.status_code == 200
    except:
        return False

def get_documents_list():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []

# Sidebar
with st.sidebar:
    st.title("🤖 Agentic RAG System")
    
    # Debug button
    if st.button("🐛 Debug Connection", type="secondary", use_container_width=True):
        debug_success, debug_info = debug_backend_connection()
        st.session_state.debug_info = debug_info
        st.session_state.show_debug = True
        st.rerun()
    
    # API Status
    st.subheader("🔌 Connection Status")
    api_online = check_api_status()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        status_class = "status-online" if api_online else "status-offline"
        st.markdown(f'<div class="status-indicator {status_class}"></div>', unsafe_allow_html=True)
    with col2:
        status_text = "✅ Online" if api_online else "❌ Offline"
        model_info = f" ({st.session_state.backend_model})" if api_online else ""
        st.write(f"**Backend:** {status_text}{model_info}")
    
    if not api_online:
        st.warning("Backend is offline. Start it with: `python backend/app.py`")
        st.code("cd backend\npython app.py", language="bash")
    
    st.divider()
    
    # Mode Selection
    st.subheader("⚙️ Agent Mode")
    agent_mode = st.toggle(
        "Enable Agentic Workflow", 
        value=st.session_state.agent_mode,
        help="When enabled, uses multi-agent system for complex reasoning"
    )
    st.session_state.agent_mode = agent_mode
    
    if agent_mode:
        show_thought = st.toggle(
            "Show Thought Process", 
            value=st.session_state.show_thought_process,
            help="Display agent reasoning steps"
        )
        st.session_state.show_thought_process = show_thought
    
    st.divider()
    
    # Document Management
    st.subheader("📚 Document Management")
    
    # Document list
    documents = get_documents_list() if api_online else []
    if documents:
        st.write(f"**Uploaded Documents ({len(documents)}):**")
        for doc in documents[:5]:
            st.write(f"📄 {doc}")
        if len(documents) > 5:
            st.write(f"*... and {len(documents) - 5} more*")
    else:
        st.info("No documents uploaded")
    
    # Clear button
    if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
        if clear_documents():
            st.session_state.uploaded_files = []
            st.success("Documents cleared!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to clear documents")
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 System Info", use_container_width=True):
            status_data = get_system_status()
            if status_data:
                st.session_state.system_info = status_data
                st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    st.divider()
    
    # System Info (collapsible)
    if 'system_info' in st.session_state:
        with st.expander("📈 System Information"):
            info = st.session_state.system_info
            
            st.metric("Documents", info.get("storage", {}).get("vector_store", {}).get("documents", 0))
            st.metric("Cache Size", info.get("storage", {}).get("cache", {}).get("size", 0))
            
            llm_status = info.get("models", {}).get("llm", {}).get("status", "unknown")
            st.write(f"**LLM:** {llm_status}")
            if info.get("models", {}).get("llm", {}).get("model"):
                st.write(f"**Model:** {info['models']['llm']['model']}")
    
    # Debug Info (collapsible)
    if hasattr(st.session_state, 'show_debug') and st.session_state.show_debug:
        with st.expander("🔧 Debug Output", expanded=True):
            st.markdown(f'<div class="debug-output">{st.session_state.debug_info}</div>', unsafe_allow_html=True)
            if st.button("Clear Debug", use_container_width=True):
                del st.session_state.debug_info
                del st.session_state.show_debug
                st.rerun()

# Main Content
st.title("🤖 Agentic RAG System")
st.caption("Intelligent document analysis with multi-agent reasoning")

# Create tabs
tab_chat, tab_upload, tab_analyze = st.tabs(["💬 Chat", "📤 Upload", "📊 Analyze"])

# Tab 1: Chat Interface
with tab_chat:
    # Mode indicator
    if st.session_state.agent_mode:
        st.success("🤖 **Agent Mode**: Multi-agent reasoning enabled")
    else:
        st.info("🔍 **Basic Mode**: Direct retrieval and response")
    
    # Chat container
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            elif message.get("type") == "thinking" and st.session_state.show_thought_process:
                st.markdown(f"""
                <div class="chat-message agent-thinking">
                    <strong>🤔 Thinking:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            else:
                # Assistant message
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show citations if available
                if message.get("citations"):
                    with st.expander(f"📚 Sources ({len(message['citations'])})"):
                        for citation in message["citations"]:
                            st.markdown(f"""
                            <div class="citation-box">
                                <strong>Source {citation.get('id', '?')}:</strong> {citation.get('source', 'Unknown')}<br>
                                <em>{citation.get('snippet', '')}</em>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    st.divider()
    
    col_input, col_send = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your question...",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask about your documents...",
            disabled=not api_online
        )
    
    with col_send:
        send_disabled = not api_online or not user_input.strip()
        send_button = st.button(
            "📤 Send", 
            type="primary", 
            use_container_width=True,
            disabled=send_disabled
        )
    
    # Handle send
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Show thinking message if in agent mode
        if st.session_state.agent_mode and st.session_state.show_thought_process:
            st.session_state.chat_history.append({
                "role": "assistant",
                "type": "thinking",
                "content": "🤔 Analyzing query, decomposing if needed, retrieving information...",
                "timestamp": time.time()
            })
        
        st.rerun()
        
        # Process query
        with st.spinner("🤖 Processing..." if st.session_state.agent_mode else "🔍 Searching..."):
            response = send_query(user_input, st.session_state.agent_mode)
            
            # Remove thinking message if it exists
            if st.session_state.agent_mode and st.session_state.show_thought_process:
                st.session_state.chat_history.pop()
            
            if response and response.status_code == 200:
                result = response.json()
                
                # Prepare assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": result.get("answer", "No answer generated."),
                    "timestamp": time.time(),
                    "processing_time": result.get("processing_time"),
                    "cached": result.get("cached", False)
                }
                
                # Add citations if available
                if result.get("citations"):
                    assistant_msg["citations"] = result["citations"]
                
                # Add thought process for agent mode
                if st.session_state.agent_mode and result.get("thought_process"):
                    for thought in result["thought_process"]:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "type": "thinking",
                            "content": f"🔧 {thought}",
                            "timestamp": time.time()
                        })
                
                # Add assistant response
                st.session_state.chat_history.append(assistant_msg)
                
                # Add performance info
                if result.get("processing_time"):
                    perf_msg = f"⏱️ Response time: {result['processing_time']}s"
                    if result.get("cached"):
                        perf_msg += " (cached)"
                    
                    st.session_state.chat_history.append({
                        "role": "system",
                        "type": "info",
                        "content": perf_msg,
                        "timestamp": time.time()
                    })
                
            else:
                error_msg = "Connection error"
                if response:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", "Unknown error")
                    except:
                        error_msg = f"HTTP {response.status_code}"
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {error_msg}",
                    "timestamp": time.time()
                })
        
        st.rerun()
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("📋 Copy Last", use_container_width=True, disabled=not st.session_state.chat_history):
            if st.session_state.chat_history:
                last_msg = st.session_state.chat_history[-1]
                if last_msg["role"] == "assistant":
                    st.code(last_msg["content"], language=None)
    
    with col3:
        if st.button("💾 Export Chat", use_container_width=True, disabled=not st.session_state.chat_history):
            chat_data = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="Download JSON",
                data=chat_data,
                file_name=f"chat_history_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

# Tab 2: Upload Documents
with tab_upload:
    st.header("📤 Upload Documents")
    
    if not api_online:
        st.warning("Backend must be running to upload documents.")
        st.info("Start the backend server first: `python backend/app.py`")
    else:
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
        if uploaded_files:
            st.write(f"**Selected {len(uploaded_files)} file(s):**")
            
            file_info = []
            for file in uploaded_files:
                size_kb = len(file.getvalue()) / 1024
                file_info.append({
                    "Name": file.name,
                    "Type": file.type,
                    "Size": f"{size_kb:.1f} KB"
                })
            
            st.dataframe(pd.DataFrame(file_info), use_container_width=True)
            
            # Process button
            if st.button("🚀 Process & Upload Files", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_container = st.empty()
                results = []
                
                for i, file in enumerate(uploaded_files):
                    status_container.write(f"Processing **{file.name}**...")
                    
                    response = upload_file(file)
                    
                    if response and response.status_code == 200:
                        result = response.json()
                        results.append({
                            "file": file.name,
                            "status": "✅ Success",
                            "chunks": result.get("chunks_processed", 0),
                            "message": result.get("message", "")
                        })
                    else:
                        error_msg = "Unknown error"
                        if response:
                            try:
                                error_data = response.json()
                                error_msg = error_data.get("error", "Upload failed")
                            except:
                                error_msg = f"HTTP {response.status_code}"
                        
                        results.append({
                            "file": file.name,
                            "status": "❌ Failed",
                            "chunks": 0,
                            "message": error_msg
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Show results
                status_container.empty()
                st.divider()
                st.subheader("📊 Processing Results")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary
                success_count = sum(1 for r in results if r["status"] == "✅ Success")
                total_chunks = sum(r["chunks"] for r in results)
                
                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count} file(s) with {total_chunks} total chunks!")
                    st.balloons()
                    
                    # Update document list
                    documents = get_documents_list()
                    st.info(f"**Total documents in system:** {len(documents)}")
                
                # Clear button
                if st.button("🔄 Process More Files", use_container_width=True):
                    st.rerun()

# Tab 3: Analyze & Debug
with tab_analyze:
    st.header("📊 System Analysis")
    
    if not api_online:
        st.warning("Connect to backend to view system analysis.")
    else:
        # System status
        with st.spinner("Fetching system status..."):
            status_data = get_system_status()
        
        if status_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                docs_count = status_data.get("storage", {}).get("vector_store", {}).get("documents", 0)
                st.metric("📚 Documents", docs_count)
            
            with col2:
                cache_size = status_data.get("storage", {}).get("cache", {}).get("size", 0)
                st.metric("💾 Cache", cache_size)
            
            with col3:
                llm_status = status_data.get("models", {}).get("llm", {}).get("status", "unknown")
                status_icon = "✅" if llm_status == "connected" else "❌"
                st.metric("🤖 LLM", f"{status_icon} {llm_status}")
            
            st.divider()
            
            # Model Information
            st.subheader("🔧 Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**LLM Model:**")
                model_name = status_data.get("models", {}).get("llm", {}).get("model", st.session_state.backend_model)
                st.code(model_name)
                
                st.write("**Embedding Model:**")
                st.code(status_data.get("models", {}).get("embeddings", {}).get("model", "N/A"))
            
            with col2:
                st.write("**Vector Store:**")
                st.code(status_data.get("storage", {}).get("vector_store", {}).get("type", "N/A"))
                
                st.write("**Collection:**")
                st.code(status_data.get("storage", {}).get("vector_store", {}).get("collection", "N/A"))
            
            st.divider()
            
            # Test Queries
            st.subheader("🧪 Test Queries")
            
            test_queries = [
                "What are the main topics in the documents?",
                "Summarize the key points from all documents",
                "Find information about specific topics",
                "Compare different concepts mentioned"
            ]
            
            test_cols = st.columns(2)
            for idx, query in enumerate(test_queries):
                with test_cols[idx % 2]:
                    if st.button(f"🔍 {query[:30]}...", use_container_width=True):
                        with st.spinner("Testing..."):
                            response = send_query(query, st.session_state.agent_mode)
                            
                            if response and response.status_code == 200:
                                result = response.json()
                                
                                with st.expander(f"Test: {query}"):
                                    st.write("**Answer:**")
                                    st.write(result.get("answer", "No answer"))
                                    
                                    st.write("**Performance:**")
                                    st.write(f"Time: {result.get('processing_time', 0):.2f}s")
                                    st.write(f"Cached: {result.get('cached', False)}")
                            else:
                                st.error("Test failed")
            
            st.divider()
            
            # Debug Tools
            st.subheader("🐛 Debug Tools")
            
            debug_query = st.text_input("Debug query:", "Test the system")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Simple Query", use_container_width=True):
                    with st.spinner("Processing..."):
                        response = send_query(debug_query, agent_mode=False)
                        if response:
                            st.json(response.json())
            
            with col2:
                if st.button("Agentic Query", use_container_width=True):
                    with st.spinner("Processing with agents..."):
                        response = send_query(debug_query, agent_mode=True)
                        if response:
                            st.json(response.json())

# Footer
st.divider()
st.caption(
    f"🤖 Agentic RAG System | "
    f"Powered by Ollama ({st.session_state.backend_model}) & ChromaDB | "
    f"Multi-Agent Reasoning System | "
    f"Backend: {st.session_state.backend_status}"
)