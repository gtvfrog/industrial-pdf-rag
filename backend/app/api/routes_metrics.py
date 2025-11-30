from fastapi import APIRouter
from app.services.metrics import get_metrics_collector

router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("")
async def get_metrics():
    collector = get_metrics_collector()
    return collector.get_all_metrics()

@router.get("/summary")
async def get_metrics_summary():
    collector = get_metrics_collector()
    data = collector.data
    
    return {
        "counts": {
            "ingestion": len(data.get("ingestion", [])),
            "embeddings": len(data.get("embeddings", [])),
            "retrieval": len(data.get("retrieval", [])),
            "llm": len(data.get("llm", []))
        },
        "current_system": collector.get_system_metrics()
    }

@router.post("/reset")
async def reset_metrics():
    collector = get_metrics_collector()
    collector.reset_metrics()
    return {"status": "metrics reset"}
