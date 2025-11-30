import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import get_settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

def download_models():
    settings = get_settings()
    
    print("Starting model download...")
    print(f"Cache directory: {settings.HF_CACHE_DIR}")
    
    os.makedirs(settings.HF_CACHE_DIR, exist_ok=True)
    
    print(f"\nDownloading Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    try:
        SentenceTransformer(settings.EMBEDDING_MODEL_NAME, cache_folder=settings.HF_CACHE_DIR)
        print("Embedding model downloaded successfully")
    except Exception as e:
        print(f"Error downloading embedding model: {e}")
        
    if settings.LLM_PROVIDER.lower() == "hf_local":
        print(f"\nDownloading LLM: {settings.HF_LLM_MODEL_NAME}")
        try:
            print("   - Downloading Tokenizer...")
            AutoTokenizer.from_pretrained(
                settings.HF_LLM_MODEL_NAME, 
                cache_dir=settings.HF_CACHE_DIR,
                trust_remote_code=True
            )
            
            print("   - Downloading Model (this may take a while)...")
            AutoModelForCausalLM.from_pretrained(
                settings.HF_LLM_MODEL_NAME, 
                cache_dir=settings.HF_CACHE_DIR,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("LLM downloaded successfully")
        except Exception as e:
            print(f"Error downloading LLM: {e}")
    else:
        print(f"\nLLM Provider is '{settings.LLM_PROVIDER}', skipping LLM download.")

    print("\nAll downloads completed")

if __name__ == "__main__":
    download_models()
