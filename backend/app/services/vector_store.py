from typing import Protocol
import logging
import numpy as np
import os
import pickle
from pathlib import Path

from app.services.models import Chunk

logger = logging.getLogger(__name__)

class VectorStore(Protocol):
    def add(self, embeddings: list[list[float]], chunks: list[Chunk]) -> None:
        ...

    def search(self, query_embedding: list[float], k: int = 5, doc_ids: list[str] | None = None) -> list[tuple[float, Chunk]]:
        ...

    def save(self, path: str | Path) -> None:
        ...

    def load(self, path: str | Path) -> None:
        ...

class InMemoryVectorStore:
    def __init__(self):
        self._vectors: list[np.ndarray] = []
        self._chunks: list[Chunk] = []
    
    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def add(self, embeddings: list[list[float]], chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")
        
        for emb, chunk in zip(embeddings, chunks):
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self._vectors.append(vec)
            self._chunks.append(chunk)

    def search(self, query_embedding: list[float], k: int = 5, doc_ids: list[str] | None = None) -> list[tuple[float, Chunk]]:
        if not self._vectors:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        scores = []
        for i, vec in enumerate(self._vectors):
            if doc_ids and self._chunks[i].doc_id not in doc_ids:
                continue
            
            score = float(np.dot(query_vec, vec))
            scores.append((score, self._chunks[i]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return scores[:k]
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        vectors_array = np.array(self._vectors) if self._vectors else np.array([])
        chunks_dicts = [chunk.model_dump() for chunk in self._chunks]
        
        data = {
            'vectors': vectors_array,
            'chunks': chunks_dicts
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Store file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vectors_array = data['vectors']
        chunks_dicts = data['chunks']
        
        self._vectors = [vec for vec in vectors_array]
        self._chunks = [Chunk(**chunk_dict) for chunk_dict in chunks_dicts]

class FaissVectorStore:
    def __init__(self, dim: int = 768):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []
    
    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def add(self, embeddings: list[list[float]], chunks: list[Chunk]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length")
        
        if not embeddings:
            return
        
        vecs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs = vecs / norms
        
        self.index.add(vecs)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], k: int = 5, doc_ids: list[str] | None = None) -> list[tuple[float, Chunk]]:
        if self.index.ntotal == 0:
            return []
        
        query_vec = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        search_k = k * 3 if doc_ids else k
        search_k = min(search_k, self.index.ntotal)
        
        scores, indices = self.index.search(query_vec, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            
            chunk = self._chunks[idx]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            
            results.append((float(score), chunk))
            if len(results) >= k:
                break
        
        return results
    
    def save(self, path: str | Path) -> None:
        import faiss
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        index_path = path.with_suffix('.index')
        meta_path = path.with_suffix('.meta.pkl')
        
        faiss.write_index(self.index, str(index_path))
        
        chunks_dicts = [chunk.model_dump() for chunk in self._chunks]
        with open(meta_path, 'wb') as f:
            pickle.dump({'chunks': chunks_dicts, 'dim': self.dim}, f)
    
    def load(self, path: str | Path) -> None:
        import faiss
        
        path = Path(path)
        index_path = path.with_suffix('.index')
        meta_path = path.with_suffix('.meta.pkl')
        
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Store files not found: {index_path} or {meta_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dim = data['dim']
        self._chunks = [Chunk(**chunk_dict) for chunk_dict in data['chunks']]

_backend_cache: VectorStore | None = None

def _resolve_store_path() -> Path:
    raw_path = os.getenv("VECTOR_STORE_PATH", "../data/vector_store")
    path = Path(raw_path)
    if not path.is_absolute():
        backend_dir = Path(__file__).resolve().parents[2]  # points to backend/
        path = (backend_dir / path).resolve()
    return path


def _load_store_if_exists(store: VectorStore, path: Path) -> None:
    try:
        if isinstance(store, FaissVectorStore):
            index_path = path.with_suffix(".index")
            meta_path = path.with_suffix(".meta.pkl")
            if index_path.exists() and meta_path.exists():
                store.load(path)
                logger.info(f"[VECTOR_STORE] Loaded FAISS index from {index_path}")
        elif isinstance(store, InMemoryVectorStore):
            if path.exists():
                store.load(path)
                logger.info(f"[VECTOR_STORE] Loaded in-memory store from {path}")
    except Exception as exc:
        logger.warning(f"[VECTOR_STORE] Failed to load persisted store: {exc}")


def persist_vector_store(store: VectorStore) -> None:
    path = _resolve_store_path()
    try:
        store.save(path)
        logger.info(f"[VECTOR_STORE] Persisted store to {path}")
    except Exception as exc:
        logger.warning(f"[VECTOR_STORE] Failed to persist store to {path}: {exc}")


def get_vector_store() -> VectorStore:
    global _backend_cache
    if _backend_cache is not None:
        return _backend_cache

    backend = os.getenv("VECTOR_STORE_BACKEND", "faiss").lower()

    if backend == "inmemory":
        store = InMemoryVectorStore()
    else:
        store = FaissVectorStore()

    _load_store_if_exists(store, _resolve_store_path())
    _backend_cache = store
    return store
