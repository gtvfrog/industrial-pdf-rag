# 🎯 Quick Reference - Refactored Structure

## Como Rodar

### Opção 1: Windows (Mais Fácil)
```bash
start.bat
```

### Opção 2: Makefile
```bash
make run
```

### Opção 3: Manual
```bash
cd backend
uvicorn app.main:app --reload
```

## Streamlit UI (em outro terminal)
```bash
streamlit run frontend/Chat.py
# OU no Windows:
start_ui.bat
```

## Estrutura de Pastas

```
industrial-rag-assistant/
├── backend/          # Código da API (rode comandos daqui)
│   ├── app/         # Imports: from app.core... (não mudou!)
│   ├── scripts/
│   └── tests/
├── frontend/         # Streamlit UI
├── config/           # .env e configurações
├── documents/        # PDFs (runtime)
├── models_cache/     # HF models (runtime)
└── metrics_history/  # Logs (runtime)
```

## Configuração

1. Copiar .env:
   ```bash
   cp config/.env.example config/.env
   ```

2. Editar `config/.env` se necessário

## Importante

- ✅ Imports NÃO mudaram (ainda são `from app...`)
- ✅ PDFs vão para `documents/` e persistem
- ✅ Modelos vão para `models_cache/` (gitignored)
- ✅ `.env` está em `config/.env` agora
- ✅ Working dir ao rodar é `backend/`
