# 📄 Industrial RAG Assistant

Sistema de perguntas e respostas baseado em PDFs de manuais industriais (WEG, WEG-CESTARI, Baldor).

---

## 🏗️ Arquitetura

```
industrial-rag-assistant/
├── backend/
│   ├── app/
│   │   ├── api/                # Rotas FastAPI
│   │   ├── core/               # Configurações
│   │   ├── services/           # Serviços (embeddings, retrieval, LLM, metrics)
│   │   └── main.py
│   ├── scripts/                # Utilitários
│   └── tests/
├── frontend/
│   └── Chat.py                 # Interface Streamlit
├── config/
│   ├── .env
│   └── logging.json
├── k8s/                        # Manifests Kubernetes
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Tecnologias

- **Backend**: FastAPI + Python 3.10+
- **Embeddings**: HuggingFace (multilingual-e5-base)
- **Vector Store**: FAISS ou InMemory
- **LLM**: Gemini 2.0 Flash ou Mistral 7B local
- **Frontend**: Streamlit
- **Deploy**: Docker + Kubernetes

---

## 🚀 Como rodar

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

- Backend: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

### Kubernetes

```bash
# Configurar secrets
kubectl apply -f k8s/secret-example.yaml

# Deploy
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

---

## 📌 Endpoints Principais

### **POST /documents**
Upload e indexação de PDFs

```bash
curl -X POST "http://localhost:8000/documents" \
  -F "files=@manual.pdf"
```

### **POST /question**
Realizar pergunta

```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "Como é o transporte de redutores?"}'
```

### **GET /documents**
Listar documentos indexados

### **GET /metrics**
Métricas do sistema (retrieval, LLM, embeddings)

---

## 🔧 Configuração

Principais variáveis de ambiente (`.env`):

```bash
# LLM
LLM_PROVIDER=gemini                               # ou "local"
GEMINI_API_KEY=your_key_here
GEMINI_LLM_MODEL=gemini-2.0-flash-exp

# Embeddings
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-base

# Vector Store
VECTOR_STORE_BACKEND=faiss                        # ou "inmemory"

# RAG
RAG_CHUNK_SIZE_CHARS=1000
RAG_CHUNK_OVERLAP_CHARS=150
ENABLE_QUERY_EXPANSION=true
QUERY_EXPANSION_USE_LLM=true
```

---

## 🧪 Testes

```bash
pytest backend/tests/ -v
```

---

## 📝 Funcionalidades

- ✅ Upload múltiplo de PDFs
- ✅ Chunking inteligente com overlap
- ✅ Embeddings multilíngues
- ✅ Query expansion com LLM
- ✅ Multi-query retrieval
- ✅ Suporte a Gemini e LLMs locais
- ✅ Métricas detalhadas
- ✅ Interface web interativa
- ✅ Deploy com Docker e Kubernetes
