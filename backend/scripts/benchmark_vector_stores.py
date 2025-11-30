import sys
import os
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import time
import numpy as np
from app.services.vector_store import InMemoryVectorStore, FaissVectorStore
from app.services.models import Chunk

def generate_fake_chunks(n: int) -> list[Chunk]:
    chunks = []
    for i in range(n):
        chunk = Chunk(
            id=f"chunk-{i}",
            doc_id=f"doc-{i % 100}",
            text=f"This is fake text for chunk {i}",
            page=i % 50,
            metadata={}
        )
        chunks.append(chunk)
    return chunks

def benchmark_backend(backend_name: str, store, dim: int, n: int, num_queries: int, k: int):
    embeddings = np.random.randn(n, dim).tolist()
    chunks = generate_fake_chunks(n)
    
    start = time.perf_counter()
    store.add(embeddings, chunks)
    index_time = time.perf_counter() - start
    
    queries = np.random.randn(num_queries, dim).tolist()
    
    start = time.perf_counter()
    for query in queries:
        _ = store.search(query, k=k)
    search_time = time.perf_counter() - start
    
    avg_search_ms = (search_time / num_queries) * 1000
    
    return {
        'backend': backend_name,
        'n': n,
        'index_time_s': round(index_time, 4),
        'search_time_s': round(search_time, 4),
        'avg_search_ms': round(avg_search_ms, 2)
    }

def main():
    dim = 768
    num_queries = 100
    k = 5
    dataset_sizes = [1_000, 5_000, 10_000]
    
    print("=" * 70)
    print("=== Benchmark Vector Stores ===")
    print(f"Dim: {dim} | Queries: {num_queries} | k: {k}")
    print("=" * 70)
    print(f"{'Backend':<15} {'N':<10} {'index_time_s':<15} {'search_time_s':<15} {'avg_search_ms':<15}")
    print("-" * 70)
    
    results = []
    
    for n in dataset_sizes:
        inmem_store = InMemoryVectorStore()
        result = benchmark_backend("InMemory", inmem_store, dim, n, num_queries, k)
        results.append(result)
        print(f"{result['backend']:<15} {result['n']:<10} {result['index_time_s']:<15} {result['search_time_s']:<15} {result['avg_search_ms']:<15}")
        
        faiss_store = FaissVectorStore(dim=dim)
        result = benchmark_backend("Faiss", faiss_store, dim, n, num_queries, k)
        results.append(result)
        print(f"{result['backend']:<15} {result['n']:<10} {result['index_time_s']:<15} {result['search_time_s']:<15} {result['avg_search_ms']:<15}")
        print()
    
    print("=" * 70)
    print("Benchmark completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
