import requests
import time
import random
import sys
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000"

QUESTIONS = [
    "Qual a temperatura máxima?",
    "Como fazer a manutenção preventiva?",
    "Quais os riscos de operação?",
    "Onde fica o botão de emergência?",
    "Qual a voltagem nominal?",
    "Como trocar o óleo?",
    "Qual a garantia do equipamento?",
    "Procedimento de ligar e desligar",
    "Códigos de erro comuns",
    "Especificações técnicas do motor"
]

def check_api():
    try:
        r = requests.get(f"{API_URL}/")
        return r.status_code == 200
    except:
        return False

def ask_question(i):
    q = random.choice(QUESTIONS)
    start = time.time()
    try:
        r = requests.post(f"{API_URL}/question", json={"question": q})
        duration = time.time() - start
        status = r.status_code
        return {"id": i, "status": status, "duration": duration, "question": q}
    except Exception as e:
        return {"id": i, "status": "error", "duration": time.time() - start, "error": str(e)}

def run_benchmark(n_requests=10, concurrency=1):
    print(f"Starting Benchmark: {n_requests} requests (concurrency={concurrency})")
    
    if not check_api():
        print("API is not running at http://localhost:8000")
        sys.exit(1)
        
    start_total = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(ask_question, i) for i in range(n_requests)]
        for f in futures:
            results.append(f.result())
            print(".", end="", flush=True)
            
    total_time = time.time() - start_total
    print(f"\n\nBenchmark Complete in {total_time:.2f}s")
    
    success = [r for r in results if r["status"] == 200]
    errors = [r for r in results if r["status"] != 200]
    
    avg_lat = sum(r["duration"] for r in success) / len(success) if success else 0
    min_lat = min(r["duration"] for r in success) if success else 0
    max_lat = max(r["duration"] for r in success) if success else 0
    
    print("\nResults:")
    print(f"  - Total Requests: {n_requests}")
    print(f"  - Success: {len(success)}")
    print(f"  - Errors: {len(errors)}")
    print(f"  - Avg Latency: {avg_lat:.2f}s")
    print(f"  - Min Latency: {min_lat:.2f}s")
    print(f"  - Max Latency: {max_lat:.2f}s")
    print(f"  - Throughput: {len(success)/total_time:.2f} req/s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of requests")
    parser.add_argument("--c", type=int, default=1, help="Concurrency")
    args = parser.parse_args()
    
    run_benchmark(args.n, args.c)
