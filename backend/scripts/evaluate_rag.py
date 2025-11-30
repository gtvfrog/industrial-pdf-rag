import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.retrieval import retrieve_relevant_chunks
from app.services.llm_orchestrator import get_llm_orchestrator
from app.services.embeddings import get_embedding_service
from app.services.vector_store import InMemoryVectorStore
from app.services.models import Chunk

async def main():
    print("--- Starting RAG Evaluation ---")
    
    settings = get_settings()
    
    print(f"Embedding Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"Embedding Model: {getattr(settings, 'EMBEDDING_MODEL_NAME', 'N/A')}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    
    if settings.EMBEDDING_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found for gemini provider.")
        return

    embedding_service = get_embedding_service(settings)
    vector_store = InMemoryVectorStore()
    llm_service = get_llm_orchestrator(settings)

    sample_file_path = "tmp/sample_rag_test.txt"
    os.makedirs("tmp", exist_ok=True)
    
    import uuid
    doc_id = str(uuid.uuid4())
    
    chunks = [
        Chunk(id=str(uuid.uuid4()), doc_id=doc_id, text="The Gemini RAG System is a powerful tool for answering questions from PDF documents.", source=sample_file_path, page_number=1),
        Chunk(id=str(uuid.uuid4()), doc_id=doc_id, text="It uses Google's Gemini models for both embeddings and text generation.", source=sample_file_path, page_number=1),
        Chunk(id=str(uuid.uuid4()), doc_id=doc_id, text="The system splits documents into chunks, embeds them, and stores them in a vector store.", source=sample_file_path, page_number=1),
        Chunk(id=str(uuid.uuid4()), doc_id=doc_id, text="When a user asks a question, relevant chunks are retrieved and passed to the LLM.", source=sample_file_path, page_number=1),
        Chunk(id=str(uuid.uuid4()), doc_id=doc_id, text="Performance depends heavily on the quality of embeddings.", source=sample_file_path, page_number=1),
        Chunk(id=str(uuid.uuid4()), doc_id="other_doc", text="Bananas are rich in potassium and are a popular fruit worldwide.", source="fruit_facts.txt", page_number=1)
    ]
    
    print("Embedding and storing chunks...")
    texts = [c.text for c in chunks]
    embeddings = embedding_service.embed(texts)
    vector_store.add(embeddings, chunks)
    
    questions = [
        "What models does the system use?",
        "How does the system handle documents?",
        "What affects the performance?",
        "Tell me about bananas."
    ]
    
    for q in questions:
        print(f"\n--- Question: {q} ---")
        
        retrieved_chunks = retrieve_relevant_chunks(
            question=q,
            settings=settings,
            embedding_service=embedding_service,
            vector_store=vector_store,
            k=2
        )
        print("Retrieved Contexts:")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  {i+1}. {chunk.text}")
            
        answer = llm_service.answer(q, retrieved_chunks)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
