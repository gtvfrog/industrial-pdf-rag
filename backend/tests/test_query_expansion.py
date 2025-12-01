import pytest
from unittest.mock import Mock

from app.core.config import Settings
from app.services.models import Chunk
from app.services.query_expansion import expand_query
from app.services.retrieval import multi_query_retrieval, normalize_scores


@pytest.fixture
def settings_no_llm():
    settings = Settings()
    settings.ENABLE_QUERY_EXPANSION = True
    settings.QUERY_EXPANSION_USE_LLM = False
    return settings


def test_expand_query_returns_original_only_when_llm_disabled(settings_no_llm):
    question = "qual é o prazo de entrega?"

    expanded = expand_query(question, settings_no_llm)

    assert expanded == [question]


def test_normalize_scores_single_value():
    chunk = Chunk(doc_id="1", id="1", text="test", page=1, metadata={})
    normalized = normalize_scores([(0.5, chunk)])

    assert len(normalized) == 1
    assert normalized[0][0] == 1.0


def test_normalize_scores_multiple_values():
    c1 = Chunk(doc_id="1", id="1", text="a", page=1, metadata={})
    c2 = Chunk(doc_id="2", id="2", text="b", page=1, metadata={})
    c3 = Chunk(doc_id="3", id="3", text="c", page=1, metadata={})

    normalized = normalize_scores([(0.1, c1), (0.5, c2), (0.9, c3)])
    scores = [s for s, _ in normalized]

    assert scores == [0.0, 0.5, 1.0]


def test_multi_query_retrieval_deduplicates_chunks():
    chunk = Chunk(doc_id="doc1", id="1", text="teste", page=1, metadata={})

    mock_settings = Mock()
    mock_settings.MULTI_QUERY_TOP_K_PER_QUERY = 5

    mock_embedding = Mock()
    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_store = Mock()
    mock_store.search.side_effect = [
        [(0.3, chunk)],
        [(0.9, chunk)],
    ]

    results = multi_query_retrieval(
        queries=["q1", "q2"],
        settings=mock_settings,
        embedding_service=mock_embedding,
        vector_store=mock_store,
        k=5,
        doc_ids=None,
    )

    assert len(results) == 1
    assert results[0][0] == 1.0


def test_multi_query_retrieval_respects_top_k():
    chunks = [
        Chunk(doc_id="d1", id="1", text="a", page=1, metadata={}),
        Chunk(doc_id="d2", id="2", text="b", page=1, metadata={}),
        Chunk(doc_id="d3", id="3", text="c", page=1, metadata={}),
    ]

    mock_settings = Mock()
    mock_settings.MULTI_QUERY_TOP_K_PER_QUERY = 5

    mock_embedding = Mock()
    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]

    mock_store = Mock()
    mock_store.search.return_value = [(0.9, chunks[0]), (0.8, chunks[1]), (0.7, chunks[2])]

    results = multi_query_retrieval(
        queries=["q"],
        settings=mock_settings,
        embedding_service=mock_embedding,
        vector_store=mock_store,
        k=2,
        doc_ids=None,
    )

    assert len(results) == 2
