from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import threading

from config import Config
from utils.document_hash import DocumentHashManager
from services.vector_store_service import VectorStoreService
from services.rag_service import RAGService

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure directories exist
Config.ensure_directories()

# Initialize services
hash_manager = DocumentHashManager(Config.DOCUMENTS_FOLDER, Config.METADATA_FILE)
vector_store_service = VectorStoreService(Config, hash_manager)
rag_service = RAGService(Config, vector_store_service)

# ============= API ENDPOINTS =============

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        "initialized": rag_service.is_initialized,
        "message": "System is ready" if rag_service.is_initialized else "System is initializing"
    })

@app.route('/progress', methods=['GET'])
def get_progress():
    """Get initialization progress"""
    return jsonify(rag_service.get_progress())

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question"""
    if not rag_service.is_initialized:
        return jsonify({
            "error": "System not initialized yet. Please wait."
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request"}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400
        
        result = rag_service.ask(question)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing question: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/rebuild', methods=['POST'])
def rebuild_database():
    """Force rebuild of the vector database"""
    if rag_service.is_initialized:
        return jsonify({
            "error": "System is running. Stop it first or restart the service."
        }), 400
    
    try:
        threading.Thread(
            target=lambda: rag_service.initialize(force_rebuild=True),
            daemon=True
        ).start()
        
        return jsonify({
            "message": "Database rebuild started",
            "status": "rebuilding"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Delete the vector database"""
    if rag_service.is_initialized:
        return jsonify({
            "error": "Stop the system first before clearing cache"
        }), 400
    
    try:
        vector_store_service.clear_database()
        return jsonify({"message": "Cache cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the system"""
    try:
        rag_service.shutdown()
        return jsonify({"message": "System shut down successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "initialized": rag_service.is_initialized
    })

# Start initialization in background
def initialize_system():
    try:
        rag_service.initialize()
    except Exception as e:
        print(f"Failed to initialize: {e}", flush=True)

# ============= MAIN =============

if __name__ == '__main__':
    print("=" * 50, flush=True)
    print("RAG API Server Starting...", flush=True)
    print(f"Documents folder: {Config.DOCUMENTS_FOLDER}", flush=True)
    print(f"ChromaDB path: {Config.CHROMA_DB_PATH}", flush=True)
    print(f"Server: http://{Config.HOST}:{Config.PORT}", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.flush()
    
    initialization_thread = threading.Thread(target=initialize_system, daemon=True)
    initialization_thread.start()

   # Start Flask server
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG
    )