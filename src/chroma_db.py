import os
import shutil
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
def save_to_chroma_db(chunks: list[Document], embedding_model) -> Chroma:
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except Exception as e:
            print(f"Error deleting existing Chroma directory: {e}")

    db = Chroma.from_documents(
        chunks,
        persist_directory=CHROMA_PATH,
        embedding=embedding_model,
    )
    
    print(f"Saved {len(chunks)} chunks to ChromaDB at {CHROMA_PATH}")

    return db