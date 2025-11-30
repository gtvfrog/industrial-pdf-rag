# 📄 RAG PDF System – Document QA

Este projeto implementa um sistema simples e funcional para perguntas e respostas baseadas em PDFs.  
A ideia é permitir que o usuário envie documentos, o backend processe o conteúdo e, depois, um LLM responda perguntas usando somente informações retiradas desses arquivos.

O foco é clareza, modularidade e facilidade de execução — alinhado ao que o desafio pede.

---

## 🧠 Visão geral

O sistema funciona em três etapas principais:

1. **Envio de PDFs**  
   O backend recebe um ou mais arquivos, extrai texto, divide em chunks e gera embeddings.

2. **Indexação**  
   Os vetores são armazenados em memória (FAISS).  
   Simples, rápido e adequado ao escopo do desafio.

3. **Perguntas**  
   O usuário envia uma pergunta → o sistema busca os chunks mais relevantes → monta o contexto → passa para o LLM gerar a resposta.

O modelo pode ser local ou remoto. A implementação deixa isso flexível.

---

## 🏗️ Arquitetura do Projeto

```
rag-pdf-system/
├── backend/
│   ├── app/
│   │   ├── api/           # Rotas e validações
│   │   ├── core/          # Configs e setup
│   │   ├── rag/           # Chunking, embeddings, FAISS, retrieval
│   │   └── llm/           # Integração com LLM
│   └── main.py            # FastAPI entrypoint
├── frontend/
│   └── streamlit_app.py   # Interface (opcional)
├── config/
│   ├── .env.example
│   └── settings.yaml
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## ⚙️ Tecnologias

- **FastAPI** – API rápida e tipada  
- **FAISS** – Busca vetorial  
- **HuggingFace Embeddings** – sentence-transformers  
- **LLM local ou API externa** – Mistral, Ollama, OpenAI etc.  
- **Streamlit** (opcional) – Interface visual simples  

---

## 📌 Endpoints

### **POST /documents**
Recebe e indexa PDFs.

Exemplo:
```bash
curl -X POST "http://localhost:8000/documents" \
  -F "files=@manual1.pdf" \
  -F "files=@manual2.pdf"
```

Resposta:
```json
{
  "message": "Documents processed successfully",
  "documents_indexed": 2,
  "total_chunks": 128
}
```

---

### **POST /question**
Recebe uma pergunta e retorna resposta + referências.

```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the power consumption?"}'
```

Resposta:
```json
{
  "answer": "The motor's power consumption is 2.3 kW.",
  "references": [
    "the motor xxx requires 2.3kw to operate at 60hz"
  ]
}
```

---

## 🚀 Como rodar

### **1. Instalar dependências**
```bash
pip install -r requirements.txt
```

### **2. Subir o backend**
```bash
uvicorn backend.main:app --reload
```

### **3. Opcional: rodar o Streamlit**
```bash
streamlit run frontend/streamlit_app.py
```

---

## 🐳 Docker

```bash
docker-compose up --build
```

Backend:  
`http://localhost:8000/docs`

Frontend opcional:  
`http://localhost:8501`

---

## 🗂️ Vector Store Backends

O sistema suporta dois backends de busca vetorial:

### **FAISS** (padrão)
- Usa FAISS IndexFlatIP
- Mais rápido para datasets grandes (>5K vetores)
- Suporta save/load com persistência de índice
- Otimizado para similaridade de cosseno

### **InMemory**
- Usa NumPy puro
- Ideal para desenvolvimento e datasets pequenos
- Suporta save/load em disco (pickle)
- Normalização automática de vetores para cosine similarity

### Como trocar entre backends

Configure a variável de ambiente `VECTOR_STORE_BACKEND`:

```bash
# Usar FAISS (padrão)
export VECTOR_STORE_BACKEND=faiss

# Usar InMemory
export VECTOR_STORE_BACKEND=inmemory
```

Ou no `.env`:
```
VECTOR_STORE_BACKEND=faiss
```

### Benchmark de Performance

Para comparar a performance entre os backends:

```bash
python -m backend.scripts.benchmark_vector_stores
```

Exemplo de saída:
```
Backend        N           index_time_s    search_time_s   avg_search_ms
------------------------------------------------------------------------
InMemory       1000        0.12            0.05            0.50
Faiss          1000        0.08            0.02            0.20
InMemory       5000        0.58            0.23            2.30
Faiss          5000        0.35            0.08            0.80
InMemory       10000       1.15            0.45            4.50
Faiss          10000       0.68            0.15            1.50
```

O benchmark testa indexação e busca com vetores de 768 dimensões.

---

## 🔪 Chunking Configuration

O sistema usa **chunking baseado em caracteres** para dividir os documentos em pedaços menores antes da indexação.

### Parâmetros Padrão

- **chunk_size_chars**: 1000 caracteres
- **chunk_overlap_chars**: 150 caracteres

Esses valores foram escolhidos como um bom equilíbrio entre:
- Contexto suficiente para embeddings significativos
- Tamanho gerenciável para o modelo de linguagem
- Overlap adequado para manter continuidade entre chunks

### Configuração

Os parâmetros podem ser ajustados via:

**1. Variáveis de ambiente (`.env`):**
```bash
RAG_CHUNK_SIZE_CHARS=1000
RAG_CHUNK_OVERLAP_CHARS=150
```

**2. Arquivo de configuração (`config/settings.yaml`):**
```yaml
rag:
  chunk_size_chars: 1000
  chunk_overlap_chars: 150
```

### Quando Ajustar

- **Documentos técnicos densos**: Aumente `chunk_size` para ~1500 para manter contexto técnico completo
- **Documentos com seções curtas**: Reduza `chunk_size` para ~700 para evitar mistura de tópicos
- **Overlap**: Aumente para ~200 se houver muitas referências cruzadas entre seções

### Teste de Sanidade

Para verificar o comportamento do chunking:
```bash
python backend\scripts\test_chunking.py
```

---
