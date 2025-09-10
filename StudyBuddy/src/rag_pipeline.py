from typing import Dict, Any, List, Optional
from pathlib import Path

from src.ingestion.document_processor import DocumentProcessor
from src.database.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.llm_client import GeminiClient
from config.settings import settings

class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retriever = Retriever()
        self.llm_client = GeminiClient()
    
    def ingest_documents(self, documents_path: str) -> Dict[str, Any]:
        documents_path = Path(documents_path)
        
        if not documents_path.exists():
            return {
                'success': False,
                'error': f"Directory {documents_path} does not exist",
                'documents_processed': 0
            }
        
        try:
            # Process documents
            documents = self.document_processor.process_directory(documents_path)
            
            if not documents:
                return {
                    'success': False,
                    'error': "No supported documents found",
                    'documents_processed': 0
                }
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            return {
                'success': True,
                'error': None,
                'documents_processed': len(documents),
                'stats': self.vector_store.get_collection_stats()
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'documents_processed': 0
            }
    
    def query(
        self, 
        question: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        try:
            # Step 1: Retrieve relevant context
            context_data = self.retriever.get_context_for_query(question)
            
            if not context_data['context'].strip():
                return {
                    'answer': "I couldn't find any relevant information in your knowledge base to answer this question. Please make sure you have uploaded and processed relevant documents.",
                    'sources': [],
                    'context_used': '',
                    'success': True,
                    'error': None
                }
            
            # Step 2: Generate response using LLM
            response_data = self.llm_client.generate_response(
                question, 
                context_data['context'], 
                conversation_history
            )
            
            result = {
                'answer': response_data['response'],
                'success': response_data['success'],
                'error': response_data['error'],
                'usage': response_data['usage']
            }
            
            if include_sources:
                result.update({
                    'sources': context_data['sources'],
                    'context_used': context_data['context'],
                    'num_chunks_used': context_data['num_chunks'],
                    'context_length': context_data['context_length']
                })
            
            return result
            
        except Exception as e:
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'context_used': '',
                'success': False,
                'error': str(e),
                'usage': None
            }
    
    def add_single_document(self, file_path: str) -> Dict[str, Any]:
        file_path = Path(file_path)
        
        try:
            processed_doc = self.document_processor.process_file(file_path)
            
            if not processed_doc:
                return {
                    'success': False,
                    'error': f"Failed to process {file_path.name}",
                    'filename': file_path.name
                }
            
            self.vector_store.add_documents([processed_doc])
            
            return {
                'success': True,
                'error': None,
                'filename': file_path.name,
                'stats': self.vector_store.get_collection_stats()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': file_path.name
            }
    
    def remove_document(self, filename: str) -> Dict[str, Any]:
        try:
            self.vector_store.delete_by_filename(filename)
            return {
                'success': True,
                'error': None,
                'filename': filename,
                'stats': self.vector_store.get_collection_stats()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        try:
            stats = self.vector_store.get_collection_stats()
            files = self.vector_store.list_files()
            
            return {
                'success': True,
                'stats': stats,
                'files': files,
                'total_files': len(files)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stats': {},
                'files': [],
                'total_files': 0
            }
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        try:
            self.vector_store.clear_collection()
            return {
                'success': True,
                'error': None,
                'message': "Knowledge base cleared successfully"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to clear knowledge base"
            }
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        try:
            results = self.retriever.retrieve_with_reranking(query, top_k)
            
            return {
                'success': True,
                'results': results,
                'query': query,
                'total_results': len(results)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'query': query,
                'total_results': 0
            }