import requests
import json

API_URL = "http://localhost:8000"

print("=" * 60)
print("Testing Metrics API")
print("=" * 60)

print("\n1. Testing health endpoint...")
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"Backend is running")
        print(f"   LLM Provider: {data.get('llm_provider')}")
        print(f"   LLM Ready: {data.get('llm_ready')}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Connection error: {e}")
    print("\nMake sure the backend is running:")
    print("   python run_backend.py")
    exit(1)

print("\n2. Testing metrics endpoint...")
try:
    response = requests.get(f"{API_URL}/metrics")
    if response.status_code == 200:
        metrics = response.json()
        print("Metrics endpoint is working")
        
        print("\nMetrics Summary:")
        print(f"   Documents processed: {len(metrics.get('ingestion', []))}")
        print(f"   Searches performed: {len(metrics.get('retrieval', []))}")
        print(f"   LLM requests: {len(metrics.get('llm', []))}")
        print(f"   Embedding batches: {len(metrics.get('embeddings', []))}")
        
        sys = metrics.get('current_system', {})
        if sys:
            print("\nSystem:")
            print(f"   CPU: {sys.get('cpu_percent', 0):.1f}%")
            print(f"   RAM: {sys.get('ram_percent', 0):.1f}%")
            gpu = sys.get('gpu', {})
            if gpu.get('available'):
                print(f"   GPU: {gpu.get('name')} - {gpu.get('memory_allocated_mb', 0):.0f} MB")
            else:
                print(f"   GPU: Not available")
        
        if metrics.get('ingestion'):
            print("\nLast documents:")
            for doc in metrics['ingestion'][-3:]:
                print(f"   - {doc['filename']} ({doc['chunks_count']} chunks)")
        
        if metrics.get('retrieval'):
            print(f"\nLast {min(3, len(metrics['retrieval']))} searches:")
            for ret in metrics['retrieval'][-3:]:
                print(f"   - Score: {ret['top_score']:.4f} | Duration: {ret['duration_seconds']:.3f}s")
                
        if metrics.get('llm'):
            print(f"\nLast {min(3, len(metrics['llm']))} responses:")
            for llm in metrics['llm'][-3:]:
                print(f"   - Tokens: {llm.get('input_tokens', 0)} in / {llm.get('output_tokens', 0)} out | {llm['duration_seconds']:.2f}s")
        
        with open("metrics_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print("\nFull metrics saved to: metrics_snapshot.json")
        
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)
print("\nTo view charts, access Performance page in Streamlit")
print("   URL: http://localhost:8501/Performance")
