import re
from typing import List, Dict, Any
import tiktoken

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        
        chunks.append({
            'text': chunk_text,
            'start_token': start,
            'end_token': end,
            'token_count': len(chunk_tokens)
        })
        
        start = end - chunk_overlap
        
        if start >= len(tokens) - chunk_overlap:
            break
    
    return chunks

def extract_metadata_from_text(text: str, filename: str) -> Dict[str, Any]:
    lines = text.split('\n')
    first_line = lines[0].strip() if lines else ""
    
    # Try to extract title
    title = first_line if first_line and len(first_line) < 100 else filename
    
    # Word count
    word_count = len(text.split())
    
    # Character count
    char_count = len(text)
    
    return {
        'title': title,
        'filename': filename,
        'word_count': word_count,
        'char_count': char_count,
        'first_line': first_line[:200]
    }