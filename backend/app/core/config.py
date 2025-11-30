from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LLM_PROVIDER: str = "local"
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    
    LLM_LOCAL_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    HF_LLM_MAX_NEW_TOKENS: int = 512
    HF_LLM_TEMPERATURE: float = 0.1
    HF_CACHE_DIR: str = "../models_cache"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_LLM_MODEL: str = "gemini-2.0-flash-exp"
    
    OPENAI_API_KEY: Optional[str] = None
    VECTOR_STORE_PATH: str = "../data/vector_store"
    METRICS_HISTORY_DIR: str = "../metrics_history"
    
    RAG_CHUNK_SIZE_CHARS: int = 1000
    RAG_CHUNK_OVERLAP_CHARS: int = 150

    model_config = SettingsConfigDict(env_file=["../config/.env", ".env"], env_file_encoding="utf-8", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    return Settings()
