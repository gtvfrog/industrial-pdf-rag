from app.services.models import Chunk
import uuid
from typing import List

def chunk_text_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text or not text.strip():
        return []
    
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start = end - overlap
        
        if start >= end:
            start = end
    
    return chunks

def chunk_pages(pages: list[dict], chunk_size: int, overlap: int, doc_id: str) -> list[Chunk]:
    all_chunks = []
    for p in pages:
        page_num = p.get("page")
        text = p.get("text", "")
        
        text_chunks = chunk_text_by_chars(text, chunk_size, overlap)
        
        for tc in text_chunks:
            chunk_id = str(uuid.uuid4())
            chunk_obj = Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=tc,
                page=page_num,
                metadata={}
            )
            all_chunks.append(chunk_obj)
            
    return all_chunks
