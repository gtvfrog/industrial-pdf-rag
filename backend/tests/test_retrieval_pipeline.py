from unittest.mock import patch

from app.core.config import Settings
from app.services.embeddings import FakeEmbeddingService
from app.services.models import Chunk
from app.services.retrieval import answer_question
from app.services.vector_store import InMemoryVectorStore


def test_answer_question_pipeline_runs_with_mock_llm():
    settings = Settings(EMBEDDING_PROVIDER="fake", LLM_PROVIDER="local")
    embedding = FakeEmbeddingService()
    store = InMemoryVectorStore()

    chunk = Chunk(
        id="1",
        doc_id="doc1",
        text="para desligar o equipamento, remova-o da tomada.",
        page=3,
        metadata={},
    )
    embeddings = embedding.embed([chunk.text])
    store.add(embeddings, [chunk])

    with patch("app.services.retrieval.answer_with_fallback") as mock_answer:
        mock_answer.return_value = ("Fake Answer", "mock", None)

        resp = answer_question(
            "como desligar o equipamento?",
            ["doc1"],
            settings,
            embedding,
            store,
            llm_provider="mock",
        )

    assert resp.answer == "Fake Answer"
    assert resp.references
    assert resp.references[0].doc_id == "doc1"
    assert resp.provider_used == "mock"
