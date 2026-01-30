#Agentic RAG System/backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from config import Config
from document_processor import DocumentProcessor
from milvus_client import MilvusClient
from rag_engine import RAGEngine
from agents import AgentOrchestrator
import json

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for frontend

# Initialize components
print("🚀 Initializing Agentic RAG System...")
doc_processor = DocumentProcessor()
milvus_client = MilvusClient()
rag_engine = RAGEngine(milvus_client)
agent_orchestrator = AgentOrchestrator()
print("✅ System initialized!")

# Helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return jsonify({
        "message": "Agentic RAG System API",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "upload": "POST /upload",
            "query": "POST /query",
            "agent_query": "POST /agent_query",
            "documents": "GET /documents",
            "clear": "POST /clear"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "ollama_connected": rag_engine.ollama_client.test_connection(),
        "milvus_connected": milvus_client.test_connection(),
        "documents_count": milvus_client.count_documents()
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process documents"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {Config.ALLOWED_EXTENSIONS}"}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process document
        print(f"📄 Processing: {filename}")
        chunks, metadata = doc_processor.process_document(filepath, filename)
        
        # Store in Milvus
        print(f"💾 Storing {len(chunks)} chunks...")
        milvus_client.add_documents(chunks, metadata)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "chunks": len(chunks),
            "message": f"Processed {len(chunks)} chunks from {filename}"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Basic RAG query"""
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 3)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Get context from RAG engine
        context = rag_engine.retrieve_context(query, top_k)
        
        # Generate answer
        answer = rag_engine.generate_answer(query, context)
        
        return jsonify({
            "query": query,
            "answer": answer,
            "context_chunks": len(context['chunks']),
            "sources": context['sources']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agent_query', methods=['POST'])
def agent_query():
    """Agentic workflow query"""
    try:
        data = request.json
        query = data.get('query')
        use_tools = data.get('use_tools', True)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        print(f"🤖 Agent processing query: {query}")
        
        # Use agent orchestrator
        result = agent_orchestrator.process_query(
            query=query,
            use_tools=use_tools,
            rag_engine=rag_engine
        )
        
        return jsonify({
            "query": query,
            "answer": result['answer'],
            "thought_process": result['thought_process'],
            "tools_used": result['tools_used'],
            "context_used": result['context_used']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        documents = milvus_client.list_documents()
        return jsonify({
            "documents": documents,
            "count": len(documents)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear all documents from vector store"""
    try:
        milvus_client.clear_collection()
        
        # Clear uploaded files
        import shutil
        if os.path.exists(Config.UPLOAD_FOLDER):
            shutil.rmtree(Config.UPLOAD_FOLDER)
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        return jsonify({
            "success": True,
            "message": "All documents cleared"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Direct vector search"""
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 5)
        
        results = milvus_client.search(query, top_k)
        
        return jsonify({
            "query": query,
            "results": results['documents'],
            "scores": results['scores'],
            "metadata": results['metadata']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🌐 Starting Flask server on http://localhost:5000")
    print(f"🤖 Using Ollama model: {Config.OLLAMA_MODEL}")
    print(f"🗄️  Milvus collection: {Config.COLLECTION_NAME}")
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)