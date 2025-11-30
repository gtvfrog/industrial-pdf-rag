from typing import Any, Optional, List
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    page: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class IndexedDocument(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int

class DocumentsResponse(BaseModel):
    message: str
    documents_indexed: int
    total_chunks: int

class Reference(BaseModel):
    doc_id: str
    page: Optional[int]
    snippet: str
    filename: Optional[str] = None

class QuestionRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    llm_provider: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    references: List[Reference]
    provider_used: str
    fallback_from: Optional[str] = None
