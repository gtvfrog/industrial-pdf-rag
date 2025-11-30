import shutil
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.core.config import Settings, get_settings
from app.services.models import DocumentsResponse
from app.services.ingestion import ingest_pdf
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("", response_model=list[str])
async def list_documents():
    vector_store = get_vector_store()
    
    if not hasattr(vector_store, 'chunks') or not vector_store.chunks:
        return []
    
    filenames = set()
    for chunk in vector_store.chunks:
        if 'filename' in chunk.metadata:
            filenames.add(chunk.metadata['filename'])
    
    return sorted(list(filenames))

@router.post("", response_model=DocumentsResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
):
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    save_dir = project_root / "documents"
    save_dir.mkdir(exist_ok=True)
    
    total_chunks = 0
    documents_indexed = 0
    
    embedding_service = get_embedding_service(settings)
    vector_store = get_vector_store()
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        
        file_path = save_dir / f"{uuid.uuid4()}_{file.filename}"
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            doc_id = str(uuid.uuid4())
            
            indexed_doc = ingest_pdf(
                file_path=str(file_path),
                doc_id=doc_id,
                filename=file.filename,
                settings=settings,
                vector_store=vector_store,
                embedding_service=embedding_service
            )
            
            total_chunks += indexed_doc.num_chunks
            documents_indexed += 1
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return DocumentsResponse(
        message="Documents processed successfully",
        documents_indexed=documents_indexed,
        total_chunks=total_chunks
    )
