import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
import os

import src.utils.text_processing as text_utils
from config.settings import settings
from src.database.chroma_config import get_chroma_client

class VectorStore:
    def __init__(self):
        # Set environment variable to disable telemetry
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        try:
            self.client = get_chroma_client(settings.CHROMA_PERSIST_DIRECTORY)
        except Exception as e:
            print(f"ChromaDB initialization warning (continuing anyway): {e}")
            # Fallback to basic client
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        self.collection_name = settings.COLLECTION_NAME
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create a collection, ensuring it exists"""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def _refresh_collection(self):
        """Refresh collection reference to avoid stale references"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self._get_or_create_collection()
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self._refresh_collection()  # Ensure we have a valid collection reference
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            
            chunks = text_utils.chunk_text(
                content, 
                chunk_size=settings.CHUNK_SIZE, 
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_text = chunk['text']
                
                chunk_metadata = {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'token_count': chunk['token_count']
                }
                
                # Create embedding
                embedding = self.embedding_model.encode(chunk_text).tolist()
                
                all_chunks.append(chunk_text)
                all_embeddings.append(embedding)
                all_metadatas.append(chunk_metadata)
                all_ids.append(chunk_id)
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_end = min(i + batch_size, len(all_chunks))
            
            self.collection.add(
                documents=all_chunks[i:batch_end],
                embeddings=all_embeddings[i:batch_end],
                metadatas=all_metadatas[i:batch_end],
                ids=all_ids[i:batch_end]
            )
        
        print(f"Added {len(all_chunks)} chunks from {len(documents)} documents")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        self._refresh_collection()  # Ensure we have a valid collection reference
        
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity_score = 1 - distance  # Convert distance to similarity
            
            if similarity_score >= settings.SIMILARITY_THRESHOLD:
                search_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'rank': i + 1
                })
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        self._refresh_collection()  # Ensure we have a valid collection reference
        
        try:
            count = self.collection.count()
        except Exception as e:
            # If collection doesn't exist, return zero stats
            print(f"Warning: Could not get collection stats: {e}")
            count = 0
        return {
            'total_chunks': count,
            'collection_name': settings.COLLECTION_NAME,
            'embedding_model': settings.EMBEDDING_MODEL
        }
    
    def clear_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection might not exist
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared successfully")
    
    def delete_by_filename(self, filename: str):
        self._refresh_collection()  # Ensure we have a valid collection reference
        
        results = self.collection.get(
            where={"filename": filename},
            include=['ids']
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks from {filename}")
    
    def list_files(self) -> List[str]:
        self._refresh_collection()  # Ensure we have a valid collection reference
        
        try:
            results = self.collection.get(include=['metadatas'])
        except Exception as e:
            print(f"Warning: Could not list files: {e}")
            return []
        filenames = set()
        
        for metadata in results['metadatas']:
            if 'filename' in metadata:
                filenames.add(metadata['filename'])
        
        return sorted(list(filenames))