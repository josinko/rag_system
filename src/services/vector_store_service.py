import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStoreService:
    """Manages vector store operations"""
    
    def __init__(self, config, hash_manager):
        self.config = config
        self.hash_manager = hash_manager
        self.embeddings = None
        self.vectorstore = None
    
    def initialize_embeddings(self):
        """Initialize the embeddings model"""
        print("Initializing embeddings model...", flush=True)
        self.embeddings = OllamaEmbeddings(model=self.config.EMBEDDING_MODEL)
        return self.embeddings
    
    def should_rebuild(self):
        """Determine if vector store needs to be rebuilt"""
        # Check if database exists
        if not os.path.exists(self.config.CHROMA_DB_PATH) or not os.listdir(self.config.CHROMA_DB_PATH):
            return True, "no_database"
        
        # Check if documents changed
        has_changed, reason = self.hash_manager.has_documents_changed()
        return has_changed, reason
    
    def build_from_documents(self, progress_callback=None):
        """Build vector store from PDF documents"""
        print(f"Loading PDFs from '{self.config.DOCUMENTS_FOLDER}'...", flush=True)
        
        if progress_callback:
            progress_callback("loading", "Loading PDF documents...", 10)
        
        # Load documents
        loader = PyPDFDirectoryLoader(self.config.DOCUMENTS_FOLDER)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages", flush=True)
        
        if progress_callback:
            progress_callback("splitting", f"Loaded {len(documents)} pages, splitting into chunks...", 25)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks", flush=True)
        
        # Initialize embeddings if not already done
        if self.embeddings is None:
            self.initialize_embeddings()
        
        if progress_callback:
            progress_callback("vectorstore", f"Building vector database from {len(chunks)} chunks...", 70)
        
        # Create vector store
        print(f"Creating vector store from {len(chunks)} chunks...", flush=True)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.config.CHROMA_DB_PATH
        )
        print("Vector store created successfully!", flush=True)
        
        # Save document hash
        current_hash = self.hash_manager.get_documents_hash()
        self.hash_manager.save_hash(current_hash)
        
        return self.vectorstore
    
    def load_existing(self):
        """Load existing vector store from disk"""
        print("Loading existing vector database...", flush=True)
        
        # Initialize embeddings if not already done
        if self.embeddings is None:
            self.initialize_embeddings()
        
        self.vectorstore = Chroma(
            persist_directory=self.config.CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )
        print("Vector store loaded from disk!", flush=True)
        return self.vectorstore
    
    def get_retriever(self):
        """Get retriever from vector store"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        return self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.RETRIEVER_K}
        )
    
    def clear_database(self):
        """Delete the vector database"""
        import shutil
        if os.path.exists(self.config.CHROMA_DB_PATH):
            shutil.rmtree(self.config.CHROMA_DB_PATH)
            print("Vector database cleared", flush=True)