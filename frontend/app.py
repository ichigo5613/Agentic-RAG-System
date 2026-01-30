#Agentic RAG System/frontend/app.py
import streamlit as st
import requests
import json
import time
import os
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API URL
BACKEND_URL = "http://localhost:5000"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'agent_mode' not in st.session_state:
    st.session_state.agent_mode = True

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .agent-thinking {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_file(file):
    """Upload file to backend"""
    try:
        files = {'file': (file.name, file.getvalue())}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        return response
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def send_query(query, use_agent=True):
    """Send query to backend"""
    endpoint = "/agent_query" if use_agent else "/query"
    payload = {"query": query}
    if use_agent:
        payload["use_tools"] = st.session_state.get("use_tools", True)
    
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

def get_documents():
    """Get list of uploaded documents"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except:
        return []
    return []

def clear_documents():
    """Clear all documents"""
    try:
        response = requests.post(f"{BACKEND_URL}/clear", timeout=10)
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Backend status
    st.subheader("System Status")
    if check_backend():
        st.success("✅ Backend Connected")
    else:
        st.error("❌ Backend Offline")
        st.info("Start the backend: `python backend/app.py`")
    
    # Mode selection
    st.subheader("Mode")
    agent_mode = st.toggle("🤖 Agent Mode", value=True)
    st.session_state.agent_mode = agent_mode
    
    if agent_mode:
        use_tools = st.toggle("🛠️ Enable Tools", value=True)
        st.session_state.use_tools = use_tools
    
    # Document management
    st.subheader("📚 Documents")
    
    # List current documents
    documents = get_documents()
    if documents:
        st.write("Uploaded files:")
        for doc in documents[:5]:  # Show first 5
            st.write(f"• {doc}")
        if len(documents) > 5:
            st.write(f"... and {len(documents) - 5} more")
    else:
        st.info("No documents uploaded yet")
    
    # Clear button
    if st.button("🗑️ Clear All Documents", type="secondary"):
        if clear_documents():
            st.session_state.uploaded_files = []
            st.success("Documents cleared!")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Failed to clear documents")
    
    st.divider()
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("📝 Summarize Documents"):
        with st.spinner("Summarizing..."):
            try:
                response = requests.post(f"{BACKEND_URL}/query", 
                                       json={"query": "Summarize all documents"},
                                       timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"**Document Summary:**\n\n{result['answer']}",
                        "type": "summary"
                    })
                    st.rerun()
            except:
                st.error("Summarization failed")
    
    if st.button("🔍 Search Test"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": "Search for important information"
        })
        st.rerun()

# Main content area
st.title("🤖 Agentic RAG System")
st.caption("Upload documents and chat with AI agents using local LLMs")

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload", "📊 Status"])

# Tab 1: Chat Interface
with tab1:
    # Display mode
    mode_text = "🤖 **Agent Mode**" if st.session_state.agent_mode else "🔍 **Basic RAG Mode**"
    st.write(mode_text)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            elif message.get("type") == "thinking":
                st.markdown(f"""
                <div class="chat-message agent-thinking">
                    <strong>🤔 Thinking:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"📄 {source}")
    
    # Chat input
    st.divider()
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask about your documents..."
        )
    
    with col2:
        send_button = st.button("📤 Send", type="primary", use_container_width=True)
    
    # Handle send
    if send_button and user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Add thinking message in agent mode
        if st.session_state.agent_mode:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Analyzing query and retrieving information...",
                "type": "thinking"
            })
        
        st.rerun()
        
        # Remove thinking message
        if st.session_state.agent_mode:
            st.session_state.chat_history.pop()
        
        # Get response
        with st.spinner("Thinking..." if st.session_state.agent_mode else "Searching..."):
            response = send_query(user_input, st.session_state.agent_mode)
            
            if response and response.status_code == 200:
                result = response.json()
                
                # Add assistant response
                assistant_msg = {
                    "role": "assistant",
                    "content": result["answer"]
                }
                
                # Add additional info for agent mode
                if st.session_state.agent_mode:
                    if result.get("thought_process"):
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": " | ".join(result["thought_process"]),
                            "type": "thinking"
                        })
                    
                    if result.get("tools_used"):
                        assistant_msg["content"] += f"\n\n**Tools used:** {', '.join(result['tools_used'])}"
                
                if result.get("sources"):
                    assistant_msg["sources"] = result["sources"]
                
                st.session_state.chat_history.append(assistant_msg)
                
            else:
                error_msg = response.json().get("error", "Unknown error") if response else "Connection error"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {error_msg}"
                })
        
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# Tab 2: Upload Documents
with tab2:
    st.header("📤 Upload Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'xlsx', 'xls', 'pptx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word, Text, Excel, PowerPoint"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s):")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        # Process button
        if st.button("🚀 Process Files", type="primary"):
            progress_bar = st.progress(0)
            success_count = 0
            
            for i, file in enumerate(uploaded_files):
                st.write(f"Processing {file.name}...")
                
                response = upload_file(file)
                
                if response and response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {file.name}: {result.get('message', 'Processed')}")
                    if file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(file.name)
                    success_count += 1
                else:
                    error = response.json().get("error", "Unknown error") if response else "Upload failed"
                    st.error(f"❌ {file.name}: {error}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if success_count > 0:
                st.balloons()
                st.success(f"Successfully processed {success_count} file(s)!")
                
                # Refresh document list
                documents = get_documents()
                st.info(f"Total documents in system: {len(documents)}")

# Tab 3: System Status
with tab3:
    st.header("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Chat Messages", len(st.session_state.chat_history))
    
    with col2:
        st.metric("Uploaded Files", len(st.session_state.uploaded_files))
    
    with col3:
        if check_backend():
            st.metric("Backend", "✅ Online")
        else:
            st.metric("Backend", "❌ Offline")
    
    st.divider()
    
    # System information
    st.subheader("System Information")
    
    if check_backend():
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Ollama Status:**")
                    if health_data.get("ollama_connected"):
                        st.success("✅ Connected")
                    else:
                        st.error("❌ Disconnected")
                
                with col2:
                    st.write("**Milvus Status:**")
                    if health_data.get("milvus_connected"):
                        st.success("✅ Connected")
                    else:
                        st.error("❌ Disconnected")
                
                st.write(f"**Documents in Vector Store:** {health_data.get('documents_count', 0)}")
                
        except:
            st.error("Could not fetch system details")
    
    # Quick test
    st.divider()
    st.subheader("Quick Test")
    
    test_query = st.text_input("Test query:", "What is in the documents?")
    
    if st.button("Run Test"):
        with st.spinner("Testing..."):
            response = send_query(test_query, False)
            
            if response and response.status_code == 200:
                result = response.json()
                st.success("✅ System is working!")
                
                with st.expander("Test Results"):
                    st.write(f"**Query:** {result.get('query')}")
                    st.write(f"**Answer:** {result.get('answer')}")
                    st.write(f"**Context Chunks:** {result.get('context_chunks')}")
                    
                    if result.get('sources'):
                        st.write("**Sources:**")
                        for source in result['sources']:
                            st.write(f"- {source}")
            else:
                st.error("❌ Test failed")

# Footer
st.divider()
st.caption("Agentic RAG System | Powered by Ollama (phi3:mini) & Milvus | Local AI Assistant")