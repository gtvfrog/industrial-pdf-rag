import pytest
import numpy as np
from app.services.models import Chunk
from app.services.embeddings import FakeEmbeddingService
from app.services.vector_store import InMemoryVectorStore

def test_vector_store_returns_most_similar_chunk():
    emb = FakeEmbeddingService()
    vs = InMemoryVectorStore()

    chunks = [
        Chunk(id="1", doc_id="doc1", text="motor elétrico trifásico", page=1, metadata={}),
        Chunk(id="2", doc_id="doc1", text="receita de bolo de chocolate", page=2, metadata={}),
    ]
    
    
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    
    vs.add([vec1, vec2], chunks)

    q_emb = [0.9, 0.1, 0.0]
    results = vs.search(q_emb, k=1)

    assert len(results) == 1
    assert results[0].id == "1"
    assert "motor" in results[0].text

def test_vector_store_filtering():
    vs = InMemoryVectorStore()
    chunks = [
        Chunk(id="1", doc_id="doc1", text="A", page=1),
        Chunk(id="2", doc_id="doc2", text="B", page=1),
    ]
    vs.add([[1.0], [1.0]], chunks)
    
    results = vs.search([1.0], k=2, doc_ids=["doc1"])
    assert len(results) == 1
    assert results[0].doc_id == "doc1"
