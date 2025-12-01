import pytest
from unittest.mock import Mock, patch, MagicMock
from app.services.query_expansion import expand_query, _expand_with_llm
from app.services.retrieval import normalize_scores, multi_query_retrieval
from app.services.models import Chunk
from app.core.config import Settings


@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.ENABLE_QUERY_EXPANSION = True
    settings.QUERY_EXPANSION_USE_LLM = False
    settings.MULTI_QUERY_TOP_K_PER_QUERY = 10
    return settings


@pytest.fixture
def mock_settings_no_key():
    question = "qual é o prazo de entrega?"
    
    expanded = expand_query(question, mock_settings_no_llm)
    
    assert len(expanded) >= 1
    assert expanded[0] == question


def test_query_expansion_first_is_original(mock_settings_no_llm):
    question = "qual é o prazo de entrega?"
    
    expanded = expand_query(question, mock_settings_no_llm)
    
    assert len(expanded) == 1
    assert expanded[0] == question


def test_query_expansion_fallback_when_no_api_key(mock_settings_no_key):
    class MockGenerativeModel:
        def generate_content(self, *args, **kwargs):
            pass
            
    mock_genai.GenerativeModel = MockGenerativeModel
    mock_model_instance = MockGenerativeModel()
    
    mock_response = MagicMock()
    mock_response.text = "como transportar redutores?\ncomo manusear redutores?\ncomo armazenar redutores?"
    
    with patch.object(MockGenerativeModel, 'generate_content', return_value=mock_response):
        mock_client = MagicMock()
        mock_client.model = mock_model_instance
        mock_get_client.return_value = mock_client
        
        result = _expand_with_llm("como é o transporte de redutores?", mock_settings)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert len(result) <= 3


def test_normalize_scores_empty_list():
    chunk = Chunk(doc_id="1", id="1", text="test", page=1, metadata={})
    results = [(0.5, chunk)]
    
    normalized = normalize_scores(results)
    
    assert len(normalized) == 1
    assert normalized[0][0] == 1.0


def test_normalize_scores_same_values():
    chunk1 = Chunk(doc_id="1", id="1", text="test1", page=1, metadata={})
    chunk2 = Chunk(doc_id="2", id="2", text="test2", page=2, metadata={})
    chunk3 = Chunk(doc_id="3", id="3", text="test3", page=3, metadata={})
    
    results = [(0.1, chunk1), (0.5, chunk2), (0.9, chunk3)]
    
    normalized = normalize_scores(results)
    
    assert normalized[0][0] == 0.0
    assert normalized[1][0] == 0.5
    assert normalized[2][0] == 1.0


def test_multi_query_retrieval_deduplication():
    chunk1 = Chunk(doc_id="doc1", id="1", text="test1", page=1, metadata={})
    
    mock_settings = Mock()
    mock_settings.MULTI_QUERY_TOP_K_PER_QUERY = 5
    
    mock_embedding_service = Mock()
    mock_embedding_service.embed.return_value = [[0.1, 0.2, 0.3]]
    
    mock_vector_store = Mock()
    mock_vector_store.search.side_effect = [
        [(0.5, chunk1)],
        [(0.9, chunk1)],
    ]
    
    queries = ["query1", "query2"]
    
    results = multi_query_retrieval(
        queries=queries,
        settings=mock_settings,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        k=5,
        doc_ids=None
    )
    
    assert len(results) == 1
    assert results[0][0] == 1.0


def test_multi_query_retrieval_respects_top_k():
