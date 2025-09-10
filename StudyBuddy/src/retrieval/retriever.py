from typing import List, Dict, Any, Optional
from src.database.vector_store import VectorStore
from config.settings import settings

class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        results = self.vector_store.search(query, top_k=top_k)
        
        # Post-process results
        processed_results = []
        for result in results:
            processed_results.append({
                'content': result['content'],
                'source': result['metadata'].get('filename', 'Unknown'),
                'title': result['metadata'].get('title', 'Untitled'),
                'similarity_score': result['similarity_score'],
                'metadata': result['metadata']
            })
        
        return processed_results
    
    def retrieve_with_reranking(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        initial_results = self.retrieve(query, top_k=(top_k or settings.TOP_K_RESULTS) * 2)
        
        # Simple reranking based on query term overlap
        query_terms = set(query.lower().split())
        
        for result in initial_results:
            content_terms = set(result['content'].lower().split())
            term_overlap = len(query_terms.intersection(content_terms)) / len(query_terms) if query_terms else 0
            
            # Combine similarity score with term overlap
            result['combined_score'] = (result['similarity_score'] * 0.7) + (term_overlap * 0.3)
        
        # Sort by combined score and return top_k
        reranked_results = sorted(initial_results, key=lambda x: x['combined_score'], reverse=True)
        return reranked_results[:top_k or settings.TOP_K_RESULTS]
    
    def get_context_for_query(self, query: str, max_context_length: int = 4000) -> Dict[str, Any]:
        results = self.retrieve_with_reranking(query)
        
        context_parts = []
        total_length = 0
        sources = set()
        
        for i, result in enumerate(results):
            content = result['content']
            source = result['source']
            
            # Add source information
            formatted_content = f"[Source: {source}]\n{content}\n"
            
            if total_length + len(formatted_content) <= max_context_length:
                context_parts.append(formatted_content)
                total_length += len(formatted_content)
                sources.add(source)
            else:
                # Try to fit partial content
                remaining_space = max_context_length - total_length - len(f"[Source: {source}]\n") - 50
                if remaining_space > 100:
                    truncated_content = content[:remaining_space] + "..."
                    formatted_content = f"[Source: {source}]\n{truncated_content}\n"
                    context_parts.append(formatted_content)
                    total_length += len(formatted_content)  # Fix: Update total_length
                    sources.add(source)
                break
        
        return {
            'context': '\n---\n'.join(context_parts),
            'sources': list(sources),
            'num_chunks': len(context_parts),
            'context_length': total_length
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_collection_stats()