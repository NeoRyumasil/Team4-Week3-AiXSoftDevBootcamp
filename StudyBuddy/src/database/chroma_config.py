import os
import chromadb
from chromadb.config import Settings as ChromaSettings

def get_chroma_client(persist_directory: str):
    """Get a ChromaDB client with proper configuration to avoid telemetry issues"""
    
    # Disable telemetry to avoid the capture() error
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    
    try:
        # Try with minimal settings first
        client = chromadb.PersistentClient(
            path=persist_directory
        )
        return client
    except Exception as e:
        print(f"Warning: ChromaDB telemetry issue (non-critical): {e}")
        
        # Fallback: try with explicit settings
        try:
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            return client
        except Exception as e2:
            print(f"ChromaDB initialization error: {e2}")
            raise e2