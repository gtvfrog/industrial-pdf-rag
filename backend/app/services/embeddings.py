from typing import Protocol
import numpy as np
import google.generativeai as genai
from app.core.config import Settings

class EmbeddingService(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

class FakeEmbeddingService:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.rng = np.random.default_rng(42)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for _ in texts:
            vec = self.rng.random(self.dim).tolist()
            embeddings.append(vec)
        return embeddings

class GeminiEmbeddingService:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
             raise ValueError("Gemini API Key not provided")
        
        model = "models/embedding-001"
        
        try:
            embeddings: list[list[float]] = []
            for text in texts:
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                if isinstance(result, dict) and "embedding" in result:
                    embeddings.append(result["embedding"])
                else:
                    raise ValueError(f"Unexpected response from Gemini Embedding: {result}")
            
            return embeddings

        except Exception as e:
            print(f"Error calling Gemini Embedding: {e}")
            raise e

class OpenAIEmbeddingService:
    def __init__(self, api_key: str | None):
        self.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
             raise ValueError("OpenAI API Key not provided")

        raise NotImplementedError("OpenAI embedding not implemented yet.")

class HFEmbeddingService:
    _model = None
    _model_name = None
    _lock = None
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None, cache_dir: str | None = None):
        if HFEmbeddingService._lock is None:
            import threading
            HFEmbeddingService._lock = threading.Lock()
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Please run `pip install sentence-transformers`.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        
        with HFEmbeddingService._lock:
            if (HFEmbeddingService._model is None or 
                HFEmbeddingService._model_name != model_name):
                print(f"Loading HF Model: {model_name} on {device}...")
                HFEmbeddingService._model = SentenceTransformer(
                    model_name, 
                    device=device,
                    cache_folder=cache_dir
                )
                HFEmbeddingService._model_name = model_name
                print(f"Embedding model loaded successfully")
        
        self.model = HFEmbeddingService._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        import time
        from app.services.metrics import get_metrics_collector
        
        start_time = time.time()
        
        processed = [f"passage: {t}" for t in texts]
        vectors = self.model.encode(
            processed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        duration = time.time() - start_time
        get_metrics_collector().record_embedding(
            batch_size=len(texts),
            duration=duration
        )
        
        return vectors.tolist()

def get_embedding_service(settings: Settings) -> EmbeddingService:
    provider = settings.EMBEDDING_PROVIDER.lower()
    
    if provider == "fake":
        return FakeEmbeddingService()
    elif provider == "huggingface":
        return HFEmbeddingService(
            model_name=settings.EMBEDDING_MODEL_NAME,
            cache_dir=settings.HF_CACHE_DIR
        )
    elif provider == "gemini":
        return GeminiEmbeddingService(api_key=settings.GEMINI_API_KEY)
    elif provider == "openai":
        return OpenAIEmbeddingService(api_key=settings.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")
