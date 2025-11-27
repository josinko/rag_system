import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

class DocumentHashManager:
    """Manages document hashing for change detection"""
    
    def __init__(self, documents_folder, metadata_file):
        self.documents_folder = Path(documents_folder)
        self.metadata_file = metadata_file
    
    def get_documents_hash(self):
        """Calculate hash of all PDF files to detect changes"""
        pdf_files = sorted(self.documents_folder.glob("*.pdf"))
        
        if not pdf_files:
            return None
        
        hash_data = []
        for pdf_file in pdf_files:
            stat = pdf_file.stat()
            hash_data.append(f"{pdf_file.name}:{stat.st_size}:{stat.st_mtime}")
        
        combined = "|".join(hash_data)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def save_hash(self, hash_value):
        """Save the current documents hash"""
        metadata = {
            "documents_hash": hash_value,
            "last_updated": datetime.now().isoformat(),
            "documents_folder": str(self.documents_folder)
        }
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_hash(self):
        """Load the saved documents hash"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get("documents_hash")
        except Exception as e:
            print(f"Could not load metadata: {e}", flush=True)
        return None
    
    def has_documents_changed(self):
        """Check if documents have changed since last build"""
        current_hash = self.get_documents_hash()
        if current_hash is None:
            return True, "no_documents"
        
        saved_hash = self.load_hash()
        if saved_hash is None:
            return True, "no_metadata"
        
        if current_hash != saved_hash:
            return True, "documents_changed"
        
        return False, "unchanged"