from fastapi import APIRouter, Depends, HTTPException
from app.core.config import Settings, get_settings
from app.services.models import QuestionRequest, QuestionResponse
from app.services.retrieval import answer_question
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.llm_orchestrator import LLMProviderError

router = APIRouter(prefix="/question", tags=["question"])

@router.post("", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    settings: Settings = Depends(get_settings),
):
    try:
        embedding_service = get_embedding_service(settings)
        vector_store = get_vector_store()
        
        response = answer_question(
            question=request.question,
            doc_ids=request.doc_ids,
            settings=settings,
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_provider=request.llm_provider
        )
        
        return response
        
    except LLMProviderError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "All LLM providers failed",
                "error": str(e),
                "provider": e.provider
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
