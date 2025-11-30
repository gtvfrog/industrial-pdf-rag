import logging
import fitz
from app.core.config import Settings
from app.services.models import IndexedDocument
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.chunking import chunk_pages

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> list[dict]:
    pages = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append({"page": i + 1, "text": text})
        doc.close()
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        raise e
        
    return pages

def ingest_pdf(
    file_path: str, 
    doc_id: str, 
    filename: str, 
    settings: Settings, 
    vector_store: VectorStore, 
    embedding_service: EmbeddingService
) -> IndexedDocument:
    
    logger.info(f"ingestion_start doc_id={doc_id} filename={filename}")
    
    pages = extract_text_from_pdf(file_path)
    
    chunks = chunk_pages(
        pages, 
        chunk_size=settings.RAG_CHUNK_SIZE_CHARS, 
        overlap=settings.RAG_CHUNK_OVERLAP_CHARS, 
        doc_id=doc_id
    )
    logger.info(f"Generated {len(chunks)} chunks for doc_id={doc_id}")
    
    for chunk in chunks:
        chunk.metadata['filename'] = filename
    
    if not chunks:
        logger.warning(f"No text extracted from {filename}")
        return IndexedDocument(doc_id=doc_id, filename=filename, num_chunks=0)

    texts = [c.text for c in chunks]
    embeddings = embedding_service.embed(texts)
    
    vector_store.add(embeddings, chunks)
    
    logger.info(f"ingestion_end doc_id={doc_id}")
    
    return IndexedDocument(doc_id=doc_id, filename=filename, num_chunks=len(chunks))
