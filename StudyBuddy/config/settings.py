import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Settings:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Database settings
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/embeddings")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_base")
    
    # Model settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Retrieval settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Document settings
    DOCUMENTS_DIRECTORY: str = os.getenv("DOCUMENTS_DIRECTORY", "./data/documents")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # UI settings
    APP_TITLE: str = os.getenv("APP_TITLE", "Personal Knowledge Assistant")
    APP_DESCRIPTION: str = os.getenv("APP_DESCRIPTION", "Chat with your personal knowledge base")
    
    def validate_required_keys(self):
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required. Please set it in your .env file")

settings = Settings()