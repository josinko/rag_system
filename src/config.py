import os
from pathlib import Path

class Config:
    """Application configuration"""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DOCUMENTS_FOLDER = os.getenv('DOCUMENTS_FOLDER', str(BASE_DIR / 'documents'))
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', str(BASE_DIR / 'chroma_db'))
    METADATA_FILE = os.path.join(CHROMA_DB_PATH, 'metadata.json')
    
    # Server settings
    HOST = os.getenv('RAG_HOST', '0.0.0.0')  # 0.0.0.0 allows network access
    PORT = int(os.getenv('RAG_PORT', 5000))
    DEBUG = os.getenv('RAG_DEBUG', 'False').lower() == 'true'
    
    # RAG settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVER_K = 3  # Number of documents to retrieve
    
    # Model settings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
    LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2')
    LLM_TEMPERATURE = 0.3
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DOCUMENTS_FOLDER, exist_ok=True)
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)