import logging
import time
from typing import Optional, List, Dict, Tuple
from app.core.config import Settings
from app.services.models import Chunk, QuestionResponse, Reference
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_orchestrator import answer_with_fallback
from app.services.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


def normalize_scores(results_with_scores: List[Tuple[float, Chunk]]) -> List[Tuple[float, Chunk]]:
    """
    Normaliza scores para range 0-1.
    Protege contra divisão por zero.
    """
    if not results_with_scores:
        return []
    
    scores = [score for score, _ in results_with_scores]
    
    if not scores:
        return results_with_scores
    
    max_score = max(scores)
    min_score = min(scores)
    
    if max_score == min_score:
        return [(1.0, chunk) for _, chunk in results_with_scores]
    
    normalized = []
    for score, chunk in results_with_scores:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((norm_score, chunk))
    
    return normalized


def multi_query_retrieval(
    queries: List[str],
    settings: Settings,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    k: int = 5,
    doc_ids: Optional[List[str]] = None
) -> List[Tuple[float, Chunk]]:
    """
    Executa retrieval para múltiplas queries e combina resultados.
    
    1. Para cada query, faz busca no vector store
    2. Normaliza scores entre 0-1
    3. Combina resultados por chunk_id (mantém maior score)
    4. Ordena por score final
    5. Retorna top-K chunks únicos com scores
    """
    merged_results: Dict[str, Tuple[float, Chunk]] = {}
    
    top_k_per_query = settings.MULTI_QUERY_TOP_K_PER_QUERY
    
    for query in queries:
        q_emb = embedding_service.embed([query])[0]
        
        results_with_scores = vector_store.search(q_emb, k=top_k_per_query, doc_ids=doc_ids)
        
        normalized = normalize_scores(results_with_scores)
        
        for score, chunk in normalized:
            chunk_id = f"{chunk.doc_id}_{chunk.id}"
            
            if chunk_id not in merged_results:
                merged_results[chunk_id] = (score, chunk)
            else:
                old_score = merged_results[chunk_id][0]
                merged_results[chunk_id] = (max(score, old_score), chunk)
    
    sorted_results = sorted(merged_results.values(), key=lambda x: x[0], reverse=True)
    
    return sorted_results[:k]


def retrieve_relevant_chunks(
    question: str, 
    settings: Settings, 
    embedding_service: EmbeddingService, 
    vector_store: VectorStore, 
    k: int = 8, 
    doc_ids: Optional[List[str]] = None
) -> List[Chunk]:
    
    q_clean = question.strip()
    
    start_time = time.time()
    
    if settings.ENABLE_QUERY_EXPANSION:
        from app.services.query_expansion import expand_query
        
        logger.info(f"[QUERY] original: {q_clean}")
        expanded_queries = expand_query(q_clean, settings)
        
        for i, q in enumerate(expanded_queries):
            logger.info(f"[QUERY] expanded[{i}]: {q}")
        
        results_with_scores = multi_query_retrieval(
            queries=expanded_queries,
            settings=settings,
            embedding_service=embedding_service,
            vector_store=vector_store,
            k=k,
            doc_ids=doc_ids
        )
        
        logger.info(f"[RETRIEVAL] total_candidates: {len(results_with_scores)}")
        logger.info(f"[RETRIEVAL] final_selected: {min(k, len(results_with_scores))}")
        
        scores = [s for s, _ in results_with_scores]
        chunks = [c for _, c in results_with_scores]
    else:
        q_emb = embedding_service.embed([q_clean])[0]
        
        results_with_scores = vector_store.search(q_emb, k=k, doc_ids=doc_ids)
        
        scores = [s for s, _ in results_with_scores]
        chunks = [c for _, c in results_with_scores]
    
    duration = time.time() - start_time
    
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
    
    chunks = retrieve_relevant_chunks(question, settings, embedding_service, vector_store, k=8, doc_ids=doc_ids)
    
    answer, provider_used, fallback_from = answer_with_fallback(
        question=question,
        chunks=chunks,
        settings=settings,
        requested_provider=llm_provider
    )
    
    references = []
    for c in chunks:
        filename = c.metadata.get('filename', f'Doc {c.doc_id[:8]}')
        page = c.page if c.page is not None else c.metadata.get('page_number', '?')
        snippet = c.text[:200].strip()
        
        references.append(Reference(
            doc_id=c.doc_id,
            page=page if isinstance(page, int) else None,
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
