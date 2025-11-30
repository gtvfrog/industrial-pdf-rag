import asyncio
from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.api import routes_documents, routes_questions, routes_metrics
from app.services.llm_orchestrator import get_llm_orchestrator
from app.services.models import Chunk

llm_ready = False
llm_loading = False

app = FastAPI()
setup_logging()
settings = get_settings()

@app.on_event("startup")
async def startup_event():
    global llm_ready, llm_loading
    
    if settings.LLM_PROVIDER == "hf_local":
        llm_loading = True
        
        async def load_llm_background():
            global llm_ready, llm_loading
            try:
                await asyncio.sleep(1)
                
                print("\nLoading LLM model in background...")
                print("Server is ready - you can upload PDFs now")
                print("Q&A will be enabled once LLM loads\n")
                
                llm = get_llm_orchestrator(settings)
                dummy = Chunk(id="warmup", doc_id="warmup", text="test", page=1, metadata={})
                _ = llm.answer("warmup", [dummy])
                
                llm_ready = True
                llm_loading = False
                print("\nLLM model loaded successfully")
                print("Q&A is now enabled\n")
            except Exception as e:
                llm_loading = False
                print(f"\nLLM loading failed: {e}")
                print("Q&A will use fallback provider (if configured)\n")
        
        asyncio.create_task(load_llm_background())
    else:
        llm_ready = True

app.include_router(routes_documents.router)
app.include_router(routes_questions.router)
app.include_router(routes_metrics.router)

@app.get("/")
async def root():
    return {"message": "Welcome to PDF RAG API. Use /docs for documentation."}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_ready": llm_ready,
        "llm_loading": llm_loading,
    }
