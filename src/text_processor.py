from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DOCUMENT_PATH = "documents"
def chunks_pdfs() -> list[Document]:    
    document_loader = PyPDFDirectoryLoader(DOCUMENT_PATH)
    documents = document_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks