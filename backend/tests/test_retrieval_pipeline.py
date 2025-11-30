from app.core.config import Settings
from app.services.models import Chunk
from app.services.embeddings import FakeEmbeddingService
from app.services.vector_store import InMemoryVectorStore
from app.services.llm_orchestrator import FakeLLMOrchestrator
from app.services.retrieval import answer_question

def test_answer_question_pipeline_runs_with_fake_llm():
    settings = Settings(EMBEDDING_PROVIDER="fake", LLM_PROVIDER="fake")
    emb = FakeEmbeddingService()
    vs = InMemoryVectorStore()
    llm = FakeLLMOrchestrator()

    # Index 1 chunk
    chunk = Chunk(id="1", doc_id="doc1", text="para desligar o equipamento, remova-o da tomada.", page=3, metadata={})
    # We need to embed it
    embeddings = emb.embed([chunk.text])
    vs.add(embeddings, [chunk])

    resp = answer_question("como desligar o equipamento?", ["doc1"], settings, emb, vs, llm)

    assert resp.answer
    assert "Fake Answer" in resp.answer
    assert resp.references
    assert resp.references[0]["doc_id"] == "doc1"
