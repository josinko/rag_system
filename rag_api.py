# rag_api.py - Flask API for RAG System (Centralized Server)

from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import threading
import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import time # NEW: For retry delay
import requests # NEW: For Ollama health check

# Force output to flush immediately (important for Windows service logging)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

app = Flask(__name__)
CORS(app)  # Enable CORS for WPF application

# Global constants and variables
OLLAMA_URL = "http://localhost:11434" # Default Ollama API host
MAX_RETRIES = 3 
RETRY_DELAY = 10 # Delay in seconds between retries for vectorstore creation

qa_chain = None
is_initialized = False
initialization_progress = {
    "status": "not_started", 
    "message": "Waiting to initialize", 
    "progress": 0
}

BASE_DIR = Path(__file__).resolve().parent

# Redefine constants using the absolute path
DOCUMENTS_FOLDER = BASE_DIR / 'documents'
CHROMA_DB_PATH = BASE_DIR / 'chroma_db'

# --- Ollama Health Check ---

def check_ollama_status():
    """Checks if the Ollama server is running and reachable."""
    try:
        # Check /api/tags endpoint as a simple health check
        print(f"Checking Ollama server at {OLLAMA_URL}...", flush=True)
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print("Ollama server is running.", flush=True)
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Ollama server not reachable at {OLLAMA_URL}. Ensure it is running: {e}", flush=True)
        return False

def get_documents_hash(documents_folder):
    """Calculate hash of all PDF files to detect changes"""
    pdf_files = sorted(Path(documents_folder).glob("*.pdf"))
    
    if not pdf_files:
        return None
    
    hash_data = []
    for pdf_file in pdf_files:
        # Include filename, size, and modification time
        stat = pdf_file.stat()
        hash_data.append(f"{pdf_file.name}:{stat.st_size}:{stat.st_mtime}")
    
    # Create hash from all file info
    combined = "|".join(hash_data)
    return hashlib.md5(combined.encode()).hexdigest()

def save_documents_hash(hash_value):
    """Save the current documents hash"""
    metadata_file = CHROMA_DB_PATH / "metadata.json" # Use the absolute path constant
    metadata = {
        "documents_hash": hash_value,
        "last_updated": datetime.now().isoformat()
    }
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

def load_documents_hash():
    """Load the saved documents hash"""
    metadata_file = CHROMA_DB_PATH / "metadata.json" # Use the absolute path constant
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get("documents_hash")
    except Exception as e:
        print(f"Could not load metadata: {e}", flush=True)
    return None

def should_rebuild_database(documents_folder):
    """Determine if database needs to be rebuilt"""
    chroma_db_path = CHROMA_DB_PATH
    
    # Check if database exists
    if not os.path.exists(chroma_db_path) or not os.listdir(chroma_db_path):
        print("No existing database found", flush=True)
        return True, "no_database"
    
    # Check if documents hash changed
    current_hash = get_documents_hash(documents_folder)
    if current_hash is None:
        print("No PDF files found", flush=True)
        return True, "no_documents"
    
    saved_hash = load_documents_hash()
    if saved_hash is None:
        print("No metadata found, rebuilding for safety", flush=True)
        return True, "no_metadata"
    
    if current_hash != saved_hash:
        print("Documents have changed, rebuilding database", flush=True)
        return True, "documents_changed"
    
    print("Documents unchanged, using existing database", flush=True)
    return False, "up_to_date"

def build_vectorstore_from_scratch(documents_folder):
    """Build vector store from documents with retry logic for robustness."""
    print(f"Loading PDFs from '{documents_folder}'...", flush=True)
    loader = PyPDFDirectoryLoader(str(documents_folder))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages", flush=True)
    
    global initialization_progress
    
    # 1. SPLITTING
    initialization_progress = {
        "status": "splitting", 
        "message": f"Loaded {len(documents)} pages, splitting into chunks...", 
        "progress": 25
    }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks", flush=True)

    # 2. CHECK FOR ZERO CHUNKS (THE FIX)
    print("Initializing embeddings model...", flush=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if not chunks:
        print("WARNING: No document chunks created. Initializing an EMPTY vector store.", flush=True)
        # Create an empty, persistent Chroma instance if no documents exist, 
        # to ensure the system can still initialize and create the retriever.
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embeddings
        )
        # We save the hash of 'None' to mark that no documents were present
        save_documents_hash(None) 
        return vectorstore, embeddings
    
    # 3. VECTOR STORE CREATION (Original Logic with Retry)
    initialization_progress = {
        "status": "vectorstore", 
        "message": f"Building vector database from {len(chunks)} chunks...", 
        "progress": 70
    }

    vectorstore = None
    for attempt in range(MAX_RETRIES):
        print(f"Creating vector store, Attempt {attempt + 1}/{MAX_RETRIES}...", flush=True)
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_PATH)
            )
            print("Vector store created successfully!", flush=True)
            break # Success, exit the loop
        except Exception as e:
            # Added a specific check for the underlying Chroma error for better logging
            if "list index out of range" in str(e):
                print("FATAL ERROR: Indexing failed due to incomplete embeddings/chunks.", flush=True)
            
            print(f"Error during vector store creation (Attempt {attempt + 1}): {str(e)}", flush=True)
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...", flush=True)
                time.sleep(RETRY_DELAY)
            else:
                raise Exception(f"Failed to create vector store after {MAX_RETRIES} attempts. Last error: {str(e)}")
    
    if vectorstore is None:
         raise Exception("Vector store creation failed, resulting in an uninitialized RAG component.")
    
    # Save the current documents hash
    current_hash = get_documents_hash(documents_folder)
    save_documents_hash(current_hash)
    
    return vectorstore, embeddings

def load_existing_vectorstore():
    """Load existing vector store from disk"""
    print("Loading existing vector database...", flush=True)
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=embeddings
        )
        print("Vector store loaded from disk!", flush=True)
        return vectorstore, embeddings
    except KeyError as e:
        # Catch the specific error often related to version incompatibility/corruption
        if str(e) == "'_type'":
            print("FATAL ERROR: Chroma metadata is incompatible/corrupted. Forcing rebuild.", flush=True)
            # Trigger rebuild logic by raising an exception that initialization will catch
            raise RuntimeError("Chroma metadata corruption detected. Force rebuild.")
        else:
            raise # Re-raise other KeyErrors

# --- Initialization Logic (Updated with Health Check) ---

def initialize_rag_system(documents_folder, force_rebuild=False):
    """Initialize the RAG system with smart caching"""
    global qa_chain, is_initialized, initialization_progress
    
    try:
        initialization_progress = {
            "status": "ollama_check", 
            "message": "Checking Ollama server status...", 
            "progress": 5
        }
        
        # NEW: Check Ollama status first
        if not check_ollama_status():
            raise Exception("Ollama server is not running or is unreachable.")

        initialization_progress = {
            "status": "checking", 
            "message": "Checking if database needs update...", 
            "progress": 10
        }

        # Determine if we need to rebuild
        needs_rebuild, reason = should_rebuild_database(documents_folder)

        if force_rebuild:
            print("Force rebuild requested", flush=True)
            needs_rebuild = True
            reason = "force_rebuild"

        # Build or load vector store
        if needs_rebuild:
            print(f"Rebuilding database (reason: {reason})", flush=True)
            vectorstore, embeddings = build_vectorstore_from_scratch(documents_folder)
        else:
            print("Using existing database", flush=True)
            initialization_progress = {
                "status": "loading", 
                "message": "Loading existing vector database...", 
                "progress": 50
            }
            vectorstore, embeddings = load_existing_vectorstore()

        initialization_progress = {
            "status": "model", 
            "message": "Setting up language model...", 
            "progress": 90
        }
        
        print("Loading LLM...", flush=True)
        llm = Ollama(model="llama3.2", temperature=0.3)
        
        prompt_template = """Use the following context to answer the question. 
If you don't know the answer, say "I cannot find this information in the documents."

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RunnableParallel(
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                # NEW: Added 'source_documents' into the main chain structure
                "source_documents": retriever 
            }
        ).assign(
            result=lambda x: (PROMPT | llm | StrOutputParser()).invoke({
                "context": x["context"],
                "question": x["question"]
            })
        )

        is_initialized = True

        initialization_progress = {
            "status": "complete", 
            "message": f"RAG system ready! (database: {reason})", 
            "progress": 100
        }

        print(f"RAG system initialized successfully! (database: {reason})", flush=True)         
        
    except Exception as e:
        print(f"ERROR during initialization: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        initialization_progress = {
            "status": "error", 
            "message": f"Initialization failed: {str(e)}", 
            "progress": 0
        }

# --- Flask Routes ---

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG system with documents"""
    global initialization_progress

    try:
        data = request.json
        documents_folder = data.get('documents_folder', 'documents')
        
        # Start initialization in background thread
        initialization_progress = {
            "status": "starting", 
            "message": "Starting initialization...", 
            "progress": 5
        }

        thread = threading.Thread(target=initialize_rag_system, args=(documents_folder,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'RAG system initialization started successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question to the RAG system"""
    global qa_chain, is_initialized
    
    if not is_initialized:
        return jsonify({
            'status': 'error',
            'message': 'RAG system not initialized. Call /api/initialize first.'
        }), 400
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({
                'status': 'error',
                'message': 'Question is required'
            }), 400
        
        result = qa_chain.invoke(question)
        
        return jsonify({
            'status': 'success',
            'answer': result['result'],
            'sources_count': len(result['source_documents']),
            'sources': [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } 
                for doc in result['source_documents']
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check if RAG system is ready and get progress"""
    global initialization_progress, is_initialized
    return jsonify({
        'status': 'success',
        'initialized': is_initialized,
        'progress': initialization_progress
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/rebuild', methods=['POST'])
def rebuild_database():
    """Force rebuild of the vector database"""
    global is_initialized, initialization_progress
    
    if is_initialized:
        # Prevent starting a rebuild while the system is actively running/serving
        # You may relax this if you implement proper lock/shutdown logic
        return jsonify({"error": "System is already running. Stop it first."}), 400
    
    try:
        # Start rebuild in background thread
        threading.Thread(
            target=initialize_rag_system, 
            args=(DOCUMENTS_FOLDER, True),  # force_rebuild=True
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
    """Delete the vector database to force full rebuild on next start"""
    global is_initialized
    
    if is_initialized:
        return jsonify({"error": "Stop the system first before clearing cache"}), 400
    
    try:
        import shutil
        chroma_db_path = "./chroma_db"
        if os.path.exists(chroma_db_path):
            shutil.rmtree(chroma_db_path)
            return jsonify({"message": "Cache cleared successfully"})
        else:
            return jsonify({"message": "No cache to clear"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("RAG API Server Starting (Centralized Server Mode)...")
    print(f"Server will be accessible at: http://0.0.0.0:5000")
    print("=" * 60)
    
    # Auto-initialize on startup in background thread
    print("\nAuto-initializing RAG system in background...")
    print("You can check progress at: http://localhost:5000/api/status\n")
    
    init_thread = threading.Thread(target=initialize_rag_system, args=(DOCUMENTS_FOLDER,))
    init_thread.daemon = True
    init_thread.start()
    
    # Start Flask server - CENTRALIZED SERVER
    app.run(host='0.0.0.0', port=5000, debug=False)