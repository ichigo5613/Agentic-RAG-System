# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import uuid
from werkzeug.utils import secure_filename
from typing import Dict, Any

from backend.config import config
from backend.core.document_processor import AdvancedDocumentProcessor
from backend.core.rag_engine import RAGEngine
from backend.utils.logger import logger
from backend.utils.error_handler import handle_exception, validate_request

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(config)
CORS(app)

# Global error handler
@app.errorhandler(Exception)
def global_error_handler(e):
    return handle_exception(e)

# Initialize components
logger.info("🚀 Starting Agentic RAG System...")

try:
    doc_processor = AdvancedDocumentProcessor()
    rag_engine = RAGEngine()
    logger.info("✅ System components initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize system components: {str(e)}")
    raise

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Agentic RAG System API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "GET /health": "System health check",
            "POST /upload": "Upload and process documents",
            "POST /query": "Query with agentic capabilities",
            "POST /quick_query": "Simple query without agents",
            "GET /documents": "List uploaded documents",
            "POST /clear": "Clear all documents",
            "GET /status": "Detailed system status"
        },
        "supported_formats": list(config.ALLOWED_EXTENSIONS),
        "max_upload_size_mb": config.MAX_CONTENT_LENGTH / (1024 * 1024)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = rag_engine.get_system_status()
        
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "llm": "connected" if status["llm_connected"] else "disconnected",
                "vector_store": "connected" if status["vector_store_connected"] else "disconnected",
                "documents": status["documents_count"],
                "cache": status["cache_size"]
            },
            "models": {
                "ollama": config.OLLAMA_MODEL,
                "embeddings": config.EMBEDDING_MODEL
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)[:100]
        }), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process document"""
    # Validate request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Supported: {list(config.ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"Processing uploaded file: {filename}")
        
        # Process document
        chunks, metadata = doc_processor.process_document(filepath, filename)
        
        # Add to vector store
        chunk_ids = rag_engine.vector_store.add_documents(chunks, metadata)
        
        # Calculate metrics
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "file_size_mb": round(file_size_mb, 2),
            "chunks_processed": len(chunks),
            "chunk_ids": chunk_ids[:5],  # Return first 5 IDs
            "message": f"Successfully processed {filename} into {len(chunks)} chunks"
        })
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        
        # Clean up uploaded file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            "error": f"Document processing failed: {str(e)[:200]}"
        }), 500

@app.route('/query', methods=['POST'])
@validate_request(['query'])
def query_documents():
    """Query documents with agentic capabilities"""
    data = request.json
    query_text = data.get('query', '').strip()
    use_agentic = data.get('use_agentic', True)
    top_k = data.get('top_k', config.TOP_K_RETRIEVAL)
    
    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    logger.info(f"Processing query: {query_text[:50]}...")
    
    try:
        # Process query
        result = rag_engine.query(
            query=query_text,
            use_agentic=use_agentic,
            top_k=top_k
        )
        
        # Prepare response
        response = {
            "query": query_text,
            "answer": result.get("answer", ""),
            "agentic": result.get("agentic", False),
            "processing_time": round(result.get("processing_time", 0), 2),
            "cached": result.get("cached", False),
            "citations": result.get("citations", []),
            "chunks_retrieved": len(result.get("citations", [])),
            "thought_process": result.get("thought_process", []) if use_agentic else None
        }
        
        # Add error if present
        if "error" in result:
            response["error"] = result["error"]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return jsonify({
            "error": f"Query processing failed: {str(e)[:200]}",
            "query": query_text
        }), 500

@app.route('/quick_query', methods=['POST'])
@validate_request(['query'])
def quick_query():
    """Simple query without agentic processing"""
    data = request.json
    query_text = data.get('query', '').strip()
    
    if not query_text:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    logger.info(f"Processing quick query: {query_text[:50]}...")
    
    try:
        # Use simple RAG without agents
        result = rag_engine.query(
            query=query_text,
            use_agentic=False
        )
        
        response = {
            "query": query_text,
            "answer": result.get("answer", ""),
            "processing_time": round(result.get("processing_time", 0), 2),
            "cached": result.get("cached", False),
            "citations": result.get("citations", []),
            "agentic": False
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Quick query failed: {str(e)}")
        return jsonify({
            "error": f"Query failed: {str(e)[:200]}",
            "query": query_text
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all uploaded documents"""
    try:
        documents = rag_engine.vector_store.list_documents()
        count = rag_engine.vector_store.count_documents()
        
        return jsonify({
            "documents": documents,
            "count": count,
            "total_chunks": count
        })
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        return jsonify({"error": str(e)[:100]}), 500

@app.route('/clear', methods=['POST'])
def clear_documents():
    """Clear all documents and reset system"""
    try:
        # Clear vector store
        rag_engine.vector_store.clear_collection()
        
        # Clear uploads directory
        import shutil
        if os.path.exists(config.UPLOAD_FOLDER):
            shutil.rmtree(config.UPLOAD_FOLDER)
            os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        
        # Clear cache
        rag_engine.clear_cache()
        
        logger.info("Cleared all documents and caches")
        
        return jsonify({
            "success": True,
            "message": "All documents and caches cleared successfully"
        })
    except Exception as e:
        logger.error(f"Failed to clear documents: {str(e)}")
        return jsonify({"error": str(e)[:100]}), 500

@app.route('/status', methods=['GET'])
def system_status():
    """Get detailed system status"""
    try:
        status = rag_engine.get_system_status()
        
        return jsonify({
            "system": {
                "status": "operational",
                "version": "1.0.0",
                "timestamp": time.time()
            },
            "models": {
                "llm": {
                    "model": config.OLLAMA_MODEL,
                    "status": "connected" if status["llm_connected"] else "disconnected",
                    "host": config.OLLAMA_HOST
                },
                "embeddings": {
                    "model": config.EMBEDDING_MODEL,
                    "status": "loaded"
                }
            },
            "storage": {
                "vector_store": {
                    "type": "ChromaDB",
                    "status": "connected" if status["vector_store_connected"] else "disconnected",
                    "documents": status["documents_count"],
                    "collection": config.COLLECTION_NAME
                },
                "cache": {
                    "size": status["cache_size"],
                    "max_size": config.MAX_CACHE_SIZE
                }
            },
            "settings": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k_retrieval": config.TOP_K_RETRIEVAL,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "agentic_enabled": config.ENABLE_QUERY_DECOMPOSITION
            }
        })
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        return jsonify({
            "error": str(e)[:100],
            "status": "degraded"
        }), 500

@app.route('/test', methods=['POST'])
def test_endpoint():
    """Test endpoint for debugging"""
    data = request.json
    test_type = data.get('type', 'echo')
    
    if test_type == 'echo':
        return jsonify({
            "message": "Echo test successful",
            "received": data
        })
    elif test_type == 'llm':
        try:
            test_query = data.get('query', 'Hello, are you working?')
            response = rag_engine.llm_client.generate(test_query)
            
            return jsonify({
                "test": "llm",
                "query": test_query,
                "response": response,
                "success": True
            })
        except Exception as e:
            return jsonify({
                "test": "llm",
                "error": str(e),
                "success": False
            }), 500
    else:
        return jsonify({
            "error": f"Unknown test type: {test_type}",
            "available_tests": ["echo", "llm"]
        }), 400

if __name__ == '__main__':
    logger.info(f"🌐 Starting Flask server on http://localhost:5000")
    logger.info(f"🤖 Using Ollama model: {config.OLLAMA_MODEL}")
    logger.info(f"🔧 Embedding model: {config.EMBEDDING_MODEL}")
    logger.info(f"🗄️ ChromaDB collection: {config.COLLECTION_NAME}")
    logger.info(f"📁 Upload folder: {config.UPLOAD_FOLDER}")
    
    app.run(
        debug=config.DEBUG,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )