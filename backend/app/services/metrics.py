import json
import time
import threading
import psutil
import torch
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

class MetricsCollector:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsCollector, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        
        self.metrics_file = project_root / "metrics_history" / "metrics.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.data = {
            "ingestion": [],
            "embeddings": [],
            "retrieval": [],
            "llm": [],
            "system_snapshots": []
        }
        
        self._load_metrics()
        self._initialized = True
    
    def _load_metrics(self):
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r") as f:
                    saved_data = json.load(f)
                    self.data.update(saved_data)
            except Exception as e:
                print(f"Error loading metrics: {e}")

    def _save_metrics(self):
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def record_ingestion(self, filename: str, duration: float, chunks_count: int):
        with self._lock:
            self.data["ingestion"].append({
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "duration_seconds": round(duration, 3),
                "chunks_count": chunks_count
            })
            self.data["ingestion"] = self.data["ingestion"][-100:]
            self._save_metrics()

    def record_embedding(self, batch_size: int, duration: float):
        with self._lock:
            self.data["embeddings"].append({
                "timestamp": datetime.now().isoformat(),
                "batch_size": batch_size,
                "duration_seconds": round(duration, 4),
                "throughput_items_per_sec": round(batch_size / duration, 2) if duration > 0 else 0
            })
            self.data["embeddings"] = self.data["embeddings"][-200:]
            self._save_metrics()

    def record_retrieval(self, query: str, duration: float, top_k_scores: List[float]):
        with self._lock:
            avg_score = sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0
            self.data["retrieval"].append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "duration_seconds": round(duration, 3),
                "avg_score": round(avg_score, 4),
                "top_score": round(max(top_k_scores), 4) if top_k_scores else 0
            })
            self.data["retrieval"] = self.data["retrieval"][-100:]
            self._save_metrics()

    def record_llm(self, question: str, duration: float, input_tokens: int, output_tokens: int):
        with self._lock:
            self.data["llm"].append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "duration_seconds": round(duration, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": round(output_tokens / duration, 2) if duration > 0 else 0
            })
            self.data["llm"] = self.data["llm"][-100:]
            self._save_metrics()

    def get_system_metrics(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        
        gpu_stats = {"available": False}
        if torch.cuda.is_available():
            gpu_stats = {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024**2, 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 2)
            }
            
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "ram_percent": ram.percent,
            "ram_used_gb": round(ram.used / 1024**3, 2),
            "gpu": gpu_stats
        }

    def get_all_metrics(self):
        with self._lock:
            current_sys = self.get_system_metrics()
            return {
                **self.data,
                "current_system": current_sys
            }

    def reset_metrics(self):
        with self._lock:
            self.data = {
                "ingestion": [],
                "embeddings": [],
                "retrieval": [],
                "llm": [],
                "system_snapshots": []
            }
            self._save_metrics()

def get_metrics_collector() -> MetricsCollector:
    return MetricsCollector()
