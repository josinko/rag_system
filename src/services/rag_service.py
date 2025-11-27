from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class RAGService:
    """Main RAG system orchestrator"""
    
    def __init__(self, config, vector_store_service):
        self.config = config
        self.vector_store_service = vector_store_service
        self.qa_chain = None
        self.is_initialized = False
        self.initialization_progress = {
            "status": "not_started",
            "message": "Not initialized",
            "progress": 0
        }
    
    def update_progress(self, status, message, progress):
        """Update initialization progress"""
        self.initialization_progress = {
            "status": status,
            "message": message,
            "progress": progress
        }
    
    def get_progress(self):
        """Get current initialization progress"""
        return self.initialization_progress
    
    def initialize(self, force_rebuild=False):
        """Initialize the RAG system"""
        try:
            self.update_progress("checking", "Checking if database needs update...", 10)
            
            # Determine if rebuild is needed
            needs_rebuild, reason = self.vector_store_service.should_rebuild()
            
            if force_rebuild:
                print("Force rebuild requested", flush=True)
                needs_rebuild = True
                reason = "force_rebuild"
            
            # Build or load vector store
            if needs_rebuild:
                print(f"Rebuilding database (reason: {reason})", flush=True)
                self.vector_store_service.build_from_documents(
                    progress_callback=self.update_progress
                )
            else:
                print("Using existing database", flush=True)
                self.update_progress("loading", "Loading existing vector database...", 50)
                self.vector_store_service.load_existing()
            
            # Initialize LLM
            self.update_progress("model", "Setting up language model...", 90)
            self._setup_qa_chain()
            
            self.is_initialized = True
            self.update_progress("complete", f"RAG system ready! (database: {reason})", 100)
            print(f"RAG system initialized successfully! (database: {reason})", flush=True)
            
        except Exception as e:
            print(f"ERROR during initialization: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            self.update_progress("error", f"Initialization failed: {str(e)}", 0)
            raise
    
    def _setup_qa_chain(self):
        """Setup the QA chain with LLM and retriever"""
        print("Loading LLM...", flush=True)
        llm = Ollama(
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMPERATURE
        )
        
        prompt_template = """Use the following context to answer the question. 
If you don't know the answer, say "I cannot find this information in the documents."

Context: {context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store_service.get_retriever()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def ask(self, question):
        """Ask a question to the RAG system"""
        if not self.is_initialized:
            raise ValueError("RAG system not initialized")
        
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        print(f"Processing question: {question}", flush=True)
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result.get("source_documents", [])
            ]
        }
    
    def shutdown(self):
        """Shutdown the RAG system"""
        self.qa_chain = None
        self.is_initialized = False
        self.update_progress("not_started", "System stopped", 0)
        print("RAG system shut down", flush=True)