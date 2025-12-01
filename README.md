# Industrial RAG Assistant

Sistema de perguntas e respostas sobre PDFs de manuais industriais (WEG, WEG-CESTARI, Baldor).

---

## Arquitetura

```
industrial-rag-assistant/
├─ backend/
│  ├─ app/
│  │  ├─ api/                # Rotas FastAPI
│  │  ├─ core/               # Configurações
│  │  ├─ services/           # Serviços (embeddings, retrieval, LLM, métricas)
│  │  └─ main.py
│  ├─ scripts/               # Utilitários
│  └─ tests/
├─ frontend/                 # Interface Streamlit
├─ config/                   # .env, logging.json
├─ k8s/                      # Manifests Kubernetes
├─ docker-compose.yml
└─ README.md
```

---

## Tecnologias

- **Backend**: FastAPI + Python 3.10+  
- **Embeddings**: HuggingFace (`intfloat/multilingual-e5-base`)  
- **Vector Store**: FAISS (persistido em disco) ou InMemory  
- **LLM**: Gemini 2.0 Flash (API) ou Mistral 7B local (quantizado)  
- **Frontend**: Streamlit  
- **Deploy**: Docker + Kubernetes  

---

## Como rodar

### Local
```bash
# Backend
cd backend
pip install -r ../requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (em outro terminal)
streamlit run frontend/Chat.py --server.port 8501
```

### Docker
```bash
docker-compose up --build
```
- Backend: http://localhost:8000/docs  
- Frontend: http://localhost:8501  

### Kubernetes
```bash
kubectl apply -f k8s/secret-example.yaml   # configure secrets
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## Endpoints principais

- **POST /documents** — upload e indexação de PDFs  
  ```bash
  curl -X POST "http://localhost:8000/documents" \
    -F "files=@manual.pdf"
  ```

- **POST /question** — fazer perguntas  
  ```bash
  curl -X POST http://localhost:8000/question \
    -H "Content-Type: application/json" \
    -d '{"question": "Como é o transporte de redutores?"}'
  ```

- **GET /documents** — lista documentos indexados  
- **GET /metrics** — métricas do sistema  

---

## Configuração

Variáveis principais (`config/.env`):
```bash
# LLM
LLM_PROVIDER=gemini                     # ou "local"
GEMINI_API_KEY=your_key_here
GEMINI_LLM_MODEL=gemini-2.0-flash-exp

# Embeddings
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base
HF_CACHE_DIR=../models_cache

# Vector Store
VECTOR_STORE_BACKEND=faiss              # ou "inmemory"
VECTOR_STORE_PATH=../data/vector_store

# RAG
RAG_CHUNK_SIZE_CHARS=1000
RAG_CHUNK_OVERLAP_CHARS=150
ENABLE_QUERY_EXPANSION=true
QUERY_EXPANSION_USE_LLM=true
MULTI_QUERY_TOP_K_PER_QUERY=10
```

---

## Testes
```bash
pytest backend/tests/ -v
```

---

## Funcionalidades

- Upload múltiplo de PDFs
- Chunking com overlap configurável
- Embeddings multilíngues
- Query expansion via LLM (opcional)
- Multi-query retrieval
- Suporte a Gemini e LLM local (Mistral 7B quantizado)
- Métricas de ingestão, embeddings, retrieval e LLM
- Interface web interativa (Streamlit)
- Deploy com Docker e Kubernetes
