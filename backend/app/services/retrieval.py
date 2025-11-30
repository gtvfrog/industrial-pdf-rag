import logging
from typing import Optional, List
from app.core.config import Settings
from app.services.models import Chunk, QuestionResponse, Reference
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_orchestrator import answer_with_fallback

logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(
    question: str, 
    settings: Settings, 
    embedding_service: EmbeddingService, 
    vector_store: VectorStore, 
    k: int = 5, 
    doc_ids: Optional[List[str]] = None
) -> List[Chunk]:
    
    q_clean = question.strip()
    
    q_emb = embedding_service.embed([q_clean])[0]
    
    import time
    from app.services.metrics import get_metrics_collector
    
    start_time = time.time()
    results_with_scores = vector_store.search(q_emb, k=k, doc_ids=doc_ids)
    duration = time.time() - start_time
    
    scores = [s for s, _ in results_with_scores]
    chunks = [c for _, c in results_with_scores]
    
    get_metrics_collector().record_retrieval(
        query=q_clean,
        duration=duration,
        top_k_scores=scores
    )
    
    return chunks

def answer_question(
    question: str, 
    doc_ids: Optional[List[str]], 
    settings: Settings, 
    embedding_service: EmbeddingService, 
    vector_store: VectorStore,
    llm_provider: Optional[str] = None
) -> QuestionResponse:
    
    logger.info("question_pipeline_start")
    
    chunks = retrieve_relevant_chunks(question, settings, embedding_service, vector_store, k=5, doc_ids=doc_ids)
    
    answer, provider_used, fallback_from = answer_with_fallback(
        question=question,
        chunks=chunks,
        settings=settings,
        requested_provider=llm_provider
    )
    
    references = []
    for c in chunks:
        filename = c.metadata.get('filename', f'Doc {c.doc_id[:8]}')
        page = c.page
        snippet = c.text[:200].strip()
        
        references.append(Reference(
            doc_id=c.doc_id,
            page=page,
            snippet=snippet,
            filename=filename
        ))
        
    logger.info("question_pipeline_end")
    
    return QuestionResponse(
        answer=answer,
        references=references,
        provider_used=provider_used,
        fallback_from=fallback_from
    )
